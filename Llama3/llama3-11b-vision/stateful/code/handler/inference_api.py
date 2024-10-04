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
from transformers import pipeline,MllamaForConditionalGeneration, MllamaProcessor
import io
from PIL import Image
import torch

from accelerate import  Accelerator


logger = logging.getLogger(__name__)

accelerator = Accelerator()

model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

class InferenceAPI:
    def __init__(self,hf_token: str):
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
        self.session_states: Dict[str, InferenceSession] = {}
        self.model = None
        self.processor = None
        self.hf_token = hf_token
    @measure_time
    def load_model(self) -> None:
        """
        Load the model and processor based on the 11B or 90B model.
        """
        logger.info(f"MYLOGS-MODEL: start loading model to {self.device}")
        model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16,use_safetensors=True, device_map=self.device,
                                                            token=self.hf_token)
        processor = MllamaProcessor.from_pretrained(model_name, token=self.hf_token,use_safetensors=True)

        model, processor=accelerator.prepare(model, processor)
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
        image = self.session_states[request.session_id].state['image']
        response_text = self._generate_text_from_image(self.model, self.processor, image, prompt,0.2,1.0)
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
        state: Dict[str, Image.Image] = {}
        key = 'image'
        image_data = download_image(request.path)
        state[key] = image_data

        return InferenceSession(
            start_time=now, last_use_time=now, session_id=request.session_id, state=state
        )

    def _generate_text_from_image(self,model, processor, image, prompt_text: str, temperature: float, top_p: float):
        """
        Generate text from an image using the model and processor.
        """
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(image, prompt, return_tensors="pt").to(self.device)
        output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=512)
        return processor.decode(output[0])[len(prompt):]   

         
if __name__ == "__main__":  
    infer = InferenceAPI("hf_token")
    infer.load_model()
    start_response = infer.start_session(StartSessionRequest("start_session_request", "https://images.pexels.com/photos/1519753/pexels-photo-1519753.jpeg","NEW_SESSION"))
    send_text_response = infer.send_text_prompt(TextPromptRequest("send_text_prompt",start_response.session_id,"describe the picture"))
    print(send_text_response.response_text)