import json
import time
import os
import random
import base64
import logging
from io import BytesIO
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

device = "cuda"
torch_dtype = torch.bfloat16

def encode_images(images):
    encoded_images = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue())
        encoded_images.append(img_str.decode("utf8"))

    return encoded_images


def model_fn(model_dir):
    pipe = QwenImageEditPipeline.from_pretrained(model_dir, 
                                                device_map="balanced",
                                                max_memory={0: "48GB", 1: "48GB", 2: "48GB", 3: "48GB"},
                                                torch_dtype=torch_dtype)
    #pipe.enable_model_cpu_offload()
    #pipe.to(device)

    return pipe


def predict_fn(data, pipe):
    print("inference started")

    positive_magic = ", Ultra HD, 4K, cinematic composition."
    default_prompt =  '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197"'''
    default_neg_prompt = " "
    default_width = 1664
    default_height = 928
    default_num_steps = 20
    default_cfg_scale = 4.0
    default_seed = 42

    base64_encoded = data.pop("image")

    # Convert base64 back to PIL Image
    image_data = base64.b64decode(base64_encoded)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    prompt = data.pop("prompt", default_prompt)
    negative_prompt = data.pop("negative_prompt", default_neg_prompt)
    height = int(data.pop("height", default_height))
    width = int(data.pop("width", default_width))
    num_steps = int(data.pop("num_steps", default_num_steps))
    true_cfg_scale = float(data.pop("true_cfg_scale", default_cfg_scale))
    seed = int(data.pop("seed", default_seed))

    start_time = time.perf_counter()

    images = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=num_steps,
        true_cfg_scale=true_cfg_scale,
        generator=torch.Generator(device=device).manual_seed(seed)
    ).images

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Execution time - in PREDICT : {elapsed_time:.6f} seconds")

    encoded_images = encode_images(images)

    return {"data": encoded_images}

