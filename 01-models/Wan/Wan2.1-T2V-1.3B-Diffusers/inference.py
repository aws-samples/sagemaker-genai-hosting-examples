import os
import time
import boto3
import torch
from botocore.exceptions import ClientError
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


def upload_file(file_name, bucket, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True


def model_fn(model_dir):
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_dir, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    return pipe


def predict_fn(data, pipe):
    print("inference started")

    bucket = data.pop("bucket")
    file_name = data.pop("file_name", "model_output.mp4")
    prompt = data.pop("prompt", "A curious raccoon")
    negative_prompt = data.pop("negative_prompt", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    height = int(data.pop("height", 480))
    width = int(data.pop("width", 832))
    num_frames = int(data.pop("num_frames", 17))
    guidance_scale = float(data.pop("guidance_scale", 5.0))
    fps = int(data.pop("fps", 15))

    start_time = time.perf_counter()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale
    ).frames[0]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Execution time - in PREDICT : {elapsed_time:.6f} seconds")

    file_path = f"/tmp/{os.path.basename(file_name)}"
    export_to_video(output, file_path, fps)

    upload_file(file_path, bucket, file_name)

    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return {"generated_video": f"s3://{bucket}/{file_name}"}
