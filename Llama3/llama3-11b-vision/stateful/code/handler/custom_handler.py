import logging, os
import datetime
from pathlib import Path
import torch

from data_types import (
    BaseRequest,
    CloseSessionRequest,
    StartSessionRequest,
    TextPromptRequest
)
from inference_api import InferenceAPI
from torch.distributed import FileStore

from ts.torch_handler.base_handler import BaseHandler

type_request_mapping = {
    "start_session": StartSessionRequest,
    "close_session": CloseSessionRequest,
    "send_text_prompt": TextPromptRequest
}
request_api_mapping = {str(v): k for k, v in type_request_mapping.items()}


logger = logging.getLogger(__name__)

OPEN_SESSION = "open_sessions"
TIMEOUT = 2


class CustomHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.ctx = ctx
        self.manifest = ctx.manifest

        self.device = torch.device("cuda:" + str(self.ctx.system_properties.get("gpu_id")))
        torch.cuda.set_device(self.device)

        identifier = f"{self.manifest['model']['modelName']}_store"
        # a store for open sessions across worker processes. We use this store to aggregate metrics for a node.  
        # It contains one key and all the sessions on a node. There will be one filestore per SM Node
        self.store_path = Path("/tmp") / identifier
        self.store = FileStore(self.store_path.as_posix(), -1)

        self.current_session = None

        ctx.cache = {}
        self.hf_token = os.environ.get('TS_HF_TOKEN')
        self.inference_api = InferenceAPI(self.hf_token)

        self.inference_api.load_model()

        self.initialized = True
        

    def preprocess(self, data):
        results = []
        for idx, row in enumerate(data):
            sequence_id = self.context.get_sequence_id(idx)

            # SageMaker sticky router relies on response header to identify the sessions
            # The sequence_id from request headers must be set in response headers
            self.context.set_response_header(
                idx, self.context.header_key_sequence_id, sequence_id
            )

            # check if sequence_id exists
            if self.context.get_request_header(
                idx, self.context.header_key_sequence_start
            ):
                self.context.cache[sequence_id] = {
                    "start": True,
                    "cancel": False,
                    "end": False,
                    "num_requests": 0,
                }
            elif sequence_id not in self.context.cache:
                logger.error(
                    f"MYLOGS-HANDLER:Not received sequence_start request for sequence_id:{sequence_id} before."
                )
                raise RuntimeError(
                    f"MYLOGS-HANDLER Not received sequence_start request for sequence_id:{sequence_id} before."
                )
            else:
                self.context.cache[sequence_id]["start"] = False

            req_id = self.context.get_request_id(idx)
            # process a new request
            if req_id not in self.context.cache:
                logger.info(
                    f"MYLOGS-HANDLER: received a new request sequence_id={sequence_id}, request_id={req_id}"
                )
                request = row.get("data") or row.get("body")
                if isinstance(request, (bytes, bytearray)):
                    request = request.decode("utf-8")

                request = load_request_from_json(request)

                self.context.cache[sequence_id]["num_requests"] += 1

                if isinstance(request, StartSessionRequest):
                    request.session_id = sequence_id
                    # Update sequence ID response header to include expiry timestamp
                    session_expiry_minutes = self.context.model_yaml_config["handler"]["sessionExpiryInMinutes"]
                    expiry = datetime.datetime.now() + datetime.timedelta(minutes=session_expiry_minutes)
                    expiry_str = expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
                    self.context.set_response_header(
                        idx, self.context.header_key_sequence_id,
                        sequence_id + f"; Expires={expiry_str}")
                elif isinstance(request, CloseSessionRequest):
                    self.context.cache[sequence_id]["end"] = True
                    self.context.set_response_header(
                        idx, self.context.header_key_sequence_end, sequence_id)
                results.append(request)
            else:
                # continue processing stream
                logger.info(
                    f"MYLOGS-HANDLER: received continuous request sequence_id={sequence_id}, request_id={req_id}"
                )
                results.append({"type": "continue", "session_id": sequence_id})

        return results

    def inference(self, input_batch):
        results = []
        for request in input_batch:
            logger.info(f"MYLOGS-HANDLER: now processing {request}")
            if isinstance(request, StartSessionRequest):
                self.open_session(request.session_id)
                results.append(self.inference_api.start_session(request).to_json())
            elif isinstance(request, CloseSessionRequest):
                self.close_session(request.session_id)
                results.append(self.inference_api.close_session(request).to_json())
            elif isinstance(request, TextPromptRequest):
                results.append(self.inference_api.send_text_prompt(request).to_json())
        return results

    def postprocess(self, inference_output):
        return inference_output

    def open_session(self, session_id):
        if self.current_session is not None:
            # Worker was assigned a new session which means the previous session has times out
            # Lets clean up the model state here. and close the session (If not already happened).
            self.inference_api.clear_session_state(self.current_session)
            self.clean_up(self.current_session, None, True)
            self.close_session(self.current_session)

        # This ID is actually generated in the frontend and will be read from header in ctx
        self.current_session = session_id

        logger.info(f"MYLOGS-HANDLER: Opening Session {session_id}")
        # Try if this is the first session
        new_open_sessions = session_id
        ret = self.store.compare_set(OPEN_SESSION, "", new_open_sessions).decode(
            "utf-8"
        )
        while ret != new_open_sessions:
            # There are other open sessions
            new_open_sessions = ";".join(ret.split(";") + [session_id])
            ret = self.store.compare_set(OPEN_SESSION, ret, new_open_sessions).decode(
                "utf-8"
            )

    def close_session(self, session_id):
        ret = self.store.compare_set(OPEN_SESSION, session_id, "").decode("utf-8")
        logger.info(f"MYLOGS-HANDLER: Closing {session_id=} Sessions open: {ret=}")
        if ret != "":
            # This session was not the only session
            if ret == session_id:
                # For some reason the session was closed before the key was set (should never happen -> error)
                # After the initial session the key should always be present (can be "" if no session is open)
                raise RuntimeError("open_sessions key was not set")

            success = False
            while not success:
                if session_id not in ret:
                    # The session was already closed, maybe through timeout
                    return
                else:
                    # Remove session_id and set in store
                    remaining_open_session = ";".join(
                        filter(lambda x: x != session_id, ret.split(";"))
                    )
                    ret = self.store.compare_set(
                        OPEN_SESSION, ret, remaining_open_session
                    ).decode("utf-8")
                    success = ret == remaining_open_session
        self.current_session = None

    def clean_up(self, seq_id, req_id, del_seq):
        # clean up
        if seq_id in self.context.cache:
            self.context.cache[seq_id]["num_requests"] -= 1
            if self.context.cache[seq_id]["num_requests"] <= 0 and del_seq:
                del self.context.cache[seq_id]

        if req_id in self.context.cache:
            del self.context.cache[req_id]


def load_request_from_json(request):
    base = BaseRequest.from_json(request)
    if base.type not in type_request_mapping:
        raise TypeError("Unsupported type")
    return type_request_mapping[base.type].from_json(request)
