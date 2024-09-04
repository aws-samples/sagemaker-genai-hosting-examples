import time
import uuid
from typing import Dict, Generator, Union
import logging

from torch import Tensor
import torch
from data_types import (
    CloseSessionRequest,
    CloseSessionResponse,
    InferenceSession,
    StartSessionRequest,
    OpenSessionResponse,
    TextPromptRequest,
    TextPromptResponse    
)
from utils import download_image, measure_time
from transformers import pipeline
import io
from PIL import Image
import torchvision.transforms as transforms
from predict import Predictor


logger = logging.getLogger(__name__)


class InferenceAPI:
    def __init__(self):
        super(InferenceAPI, self).__init__()

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
        self.pil_to_tensor_transform = transforms.Compose([transforms.PILToTensor()])
        self.session_states: Dict[str, InferenceSession] = {}
        self.predictor = Predictor()
    @measure_time
    def load_model(self) -> None:
        logger.info(f"MYLOGS-MODEL: start loading model to {self.device}")
        self.predictor.setup()
        logger.info(f"MYLOGS-MODEL: done loading model to {self.device}")

        
    @measure_time
    def start_session(self, request: StartSessionRequest) -> OpenSessionResponse:
        logger.info(f"MYLOGS-MODEL: start start_session")
        session = self.__create_session(request)
        self.session_states[session.session_id] = session
        logger.info(f"MYLOGS-MODEL: end start_session")
        return OpenSessionResponse(session_id=session.session_id)
        
    @measure_time
    def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        self.clear_session_state(request.session_id)
        return CloseSessionResponse(success=True)
        
    @measure_time
    def send_text_prompt(self, request: TextPromptRequest) -> TextPromptResponse:
        prompt = request.prompt_text
        image_gpu_tensor = self.session_states[request.session_id].state['image']
        response_text = self.predictor.predict(image_gpu_tensor, prompt,  max_tokens=1024, temperature=0.2, top_p=1.0)
        return TextPromptResponse(response_text=response_text)
        
    @measure_time
    def clear_session_state(self, session_id) -> bool:
        if session_id in self.session_states:
            del self.session_states[session_id]
            return True

        return False
        
    @measure_time
    def __create_session(self, request) -> InferenceSession:
        now = time.time()
        state: Dict[str, Tensor] = {}
        key = 'image'
        image_data = download_image(request.path)
        image_tensor = self.predictor.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().to(self.device)
        state[key] = image_tensor

        return InferenceSession(
            start_time=now, last_use_time=now, session_id=request.session_id, state=state
        )
