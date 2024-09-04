import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token

from PIL import Image

import requests
from io import BytesIO

from cog import BasePredictor, Input, Path

import os

working_dir = "/home/model-server"
model_name = "llava-v1.5-13b"
model_weights_path = os.path.join(working_dir,model_name)
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(working_dir,"weights")

class Predictor(BasePredictor):
    def setup(self) -> None:
        disable_torch_init()    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_weights_path, model_name=model_name, model_base=None, load_8bit=False, load_4bit=False)

    def predict(
        self,
        image_tensor: torch.Tensor = Input(description="Input image tensor"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
    ) -> str:
        """Run a single prediction on the model"""
    
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
       
        # loop start
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
            )
        
        generated_text = self.tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
        return generated_text
    
if __name__ == "__main__":    
    image_data = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
    predictor = Predictor()
    predictor.setup()
    i_tensor=predictor.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half()
    result = predictor.predict(image_tensor=i_tensor, prompt="tell me what is in the photo",  max_tokens=1024, temperature=0.2, top_p=1.0)
    print(result)