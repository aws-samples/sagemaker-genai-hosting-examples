import base64
import torch
from io import BytesIO
import base64
from PIL import Image
from pruna import PrunaModel
from diffusers.utils import load_image
import time

def model_fn(model_dir):
    
    print("from pretrained model")
    pipe = PrunaModel.from_pretrained(f"{model_dir}/flux_smashed", torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    print("Running first inference on smashed model")

    prompt = "Add a fun hat to the dog on the right and a top hat to the dog on the left"
    input_image = load_image("https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg")

    start_time = time.perf_counter()
    
    pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=2.5,
        num_inference_steps=50
    )["images"]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"Execution time - SMASHED after loading : {elapsed_time:.6f} seconds")

    return pipe

def decode_base64_image(image_string):
  base64_image = base64.b64decode(image_string)
  buffer = BytesIO(base64_image)
  return Image.open(buffer)

def predict_fn(data, pipe):

    print("inference started")

    prompt = data.pop("inputs", data)
    guidance_scale = data.pop("guidance_scale", 2.5)
    input_image_base64 = data.pop("input_image", None)
    input_image = decode_base64_image(input_image_base64)

    print("imagein generation started")

    start_time = time.perf_counter()

    generated_images = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=50
    )["images"]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"Execution time - in PREDICT : {elapsed_time:.6f} seconds")

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    print(f"response created:${len(encoded_images)}")

    return {"generated_images": encoded_images}
