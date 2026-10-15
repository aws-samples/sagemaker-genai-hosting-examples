# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Minimal SageMaker AI hosting server for a sample's model_fn / transform_fn adapter.

Mirrors the parts of the SageMaker PyTorch inference toolkit the decision-model samples use:

1. Install code/requirements.txt, if present (pip honours --no-index lines, so the
   network-isolated notebooks keep working).
2. Import SAGEMAKER_PROGRAM (default inference.py) from SAGEMAKER_SUBMIT_DIRECTORY and load
   the model once with model_fn("/opt/ml/model").
3. Serve GET /ping and POST /invocations on port 8080 with one process. Requests go to
   transform_fn; GenericInferenceToolkitError keeps its HTTP status, so a 422 from the
   adapter still reaches the caller as a ModelError with OriginalStatusCode 422. An error
   message that is itself JSON is returned as application/json.
4. SAGEMAKER_MODEL_SERVER_TIMEOUT, when set, bounds how long a caller waits for
   transform_fn before the server answers 504. See README.md for what it does not bound.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from sagemaker_inference.errors import GenericInferenceToolkitError

MODEL_DIR = "/opt/ml/model"
CODE_DIR = Path(os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", f"{MODEL_DIR}/code"))
PROGRAM = os.environ.get("SAGEMAKER_PROGRAM", "inference.py")


def response_timeout():
    """Seconds a caller waits for transform_fn before the server answers 504.

    The inference toolkit's variable name, with the same meaning. Unset, or 0, waits as long
    as the caller does. A value that is not a number fails here, with the reason, rather than
    leaving an operator believing a bound is in place.
    """
    value = os.environ.get("SAGEMAKER_MODEL_SERVER_TIMEOUT", "0").strip()
    try:
        seconds = float(value or 0)
    except ValueError:
        raise ValueError("SAGEMAKER_MODEL_SERVER_TIMEOUT must be a number of seconds, "
                         f"not {value!r}.") from None
    return seconds or None


RESPONSE_TIMEOUT = response_timeout()


def install_requirements() -> None:
    requirements = CODE_DIR / "requirements.txt"
    if requirements.is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)],
                       check=True)


def load_adapter():
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("inference", CODE_DIR / PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def media_type_of(message: str) -> str:
    """application/json for an error message that is itself a JSON object or array."""
    try:
        return "application/json" if isinstance(json.loads(message), (dict, list)) else "text/plain"
    except (TypeError, ValueError):
        return "text/plain"


install_requirements()
adapter = load_adapter()
model = adapter.model_fn(MODEL_DIR)
# transform_fn is synchronous, so it runs in a worker thread, as FastAPI does for sync routes.
# 40 threads is what starlette's own thread pool allowed; the adapters serialize on a lock,
# so the spare threads only ever hold calls abandoned by a timeout.
executor = ThreadPoolExecutor(max_workers=40, thread_name_prefix="transform")
app = FastAPI()


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")
    accept = request.headers.get("accept", "application/json")
    call = asyncio.get_running_loop().run_in_executor(
        executor, adapter.transform_fn, model, body, content_type, accept)
    done, _ = await asyncio.wait({call}, timeout=RESPONSE_TIMEOUT)
    if not done:
        # A running thread cannot be cancelled, so this bounds the answer, not the work:
        # the call keeps the adapter's lock until it returns. Retrieve its result later
        # so a failure after the timeout is not reported as an unhandled exception.
        call.add_done_callback(lambda finished: finished.exception())
        return Response(content=json.dumps({"error": "Inference timed out."}),
                        status_code=504, media_type="application/json")
    try:
        payload, out_type = call.result()
    except GenericInferenceToolkitError as error:
        return Response(content=error.message, status_code=error.status_code,
                        media_type=media_type_of(error.message))
    return Response(content=payload, media_type=out_type)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
