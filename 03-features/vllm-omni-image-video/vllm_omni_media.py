"""Shared helpers for the vLLM-Omni image-to-video example."""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from PIL import Image, ImageOps, UnidentifiedImageError
from urllib3.filepost import encode_multipart_formdata

DLC_ACCOUNT_ID = "763104351884"
DLC_TAG = "omni-sagemaker-cuda-v1.6"
INFERENCE_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1"

IMAGE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
VIDEO_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
IMAGE_ROUTE = "/v1/images/generations"
VIDEO_ROUTE = "/v1/videos/sync"

DEFAULT_STATE_PATH = Path(__file__).with_name(".vllm_omni_media_state.json")


@dataclass(frozen=True)
class DeploymentState:
    """Names and locations required to invoke or remove the deployment."""

    region: str
    bucket: str
    prefix: str
    image_model_name: str
    image_endpoint_config_name: str
    image_endpoint_name: str
    video_model_name: str
    video_endpoint_config_name: str
    video_endpoint_name: str


def container_image_uri(region: str) -> str:
    """Return the regional vLLM-Omni Deep Learning Container image URI."""

    return (
        f"{DLC_ACCOUNT_ID}.dkr.ecr.{region}.amazonaws.com/"
        f"vllm:{DLC_TAG}"
    )


def build_image_payload(
    prompt: str,
    *,
    size: str = "1024x1024",
    steps: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    """Build a FLUX.2-klein image generation request."""

    return {
        "model": IMAGE_MODEL_ID,
        "prompt": prompt,
        "size": size,
        "num_inference_steps": steps,
        "seed": seed,
    }


def decode_image_response(body: bytes) -> bytes:
    """Decode the first base64 image from an OpenAI-compatible response."""

    try:
        response = json.loads(body)
        encoded = response["data"][0]["b64_json"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise ValueError("Image response did not contain data[0].b64_json") from error

    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Image response contained invalid base64 data") from error


def image_data_url(
    image_bytes: bytes,
    media_type: str = "image/png",
) -> str:
    """Encode image bytes as a data URL accepted by vLLM-Omni."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def prepare_video_reference(
    image_bytes: bytes,
    *,
    width: int,
    height: int,
) -> bytes:
    """Resize the source image to a compact JPEG for the multipart request."""

    if width <= 0 or height <= 0:
        raise ValueError("Video width and height must be positive")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            reference = ImageOps.fit(
                source.convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("Source image could not be decoded") from error

    output = BytesIO()
    reference.save(
        output,
        format="JPEG",
        quality=90,
        optimize=True,
    )
    return output.getvalue()


def build_video_multipart(
    prompt: str,
    image_bytes: bytes,
    *,
    width: int = 480,
    height: int = 320,
    num_frames: int = 17,
    fps: int = 8,
    steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = 42,
    image_media_type: str = "image/png",
    boundary: str | None = None,
) -> tuple[bytes, str]:
    """Build the multipart body required by the vLLM-Omni Videos API."""

    reference = json.dumps(
        {"image_url": image_data_url(image_bytes, image_media_type)},
        separators=(",", ":"),
    )
    fields = {
        "model": VIDEO_MODEL_ID,
        "prompt": prompt,
        "image_reference": reference,
        "width": str(width),
        "height": str(height),
        "num_frames": str(num_frames),
        "fps": str(fps),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance_scale),
        "seed": str(seed),
    }
    return encode_multipart_formdata(
        fields,
        boundary=boundary or f"vllm-omni-{uuid.uuid4().hex}",
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split a complete S3 URI into bucket and key."""

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def save_state(
    state: DeploymentState,
    path: str | Path = DEFAULT_STATE_PATH,
) -> None:
    """Persist deployment state for the generation and cleanup scripts."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(asdict(state), indent=2) + "\n",
        encoding="utf-8",
    )


def load_state(path: str | Path = DEFAULT_STATE_PATH) -> DeploymentState:
    """Load a previously saved deployment state."""

    state_path = Path(path)
    if not state_path.exists():
        raise FileNotFoundError(
            f"Deployment state not found at {state_path}. Run deploy.py first."
        )
    return DeploymentState(**json.loads(state_path.read_text(encoding="utf-8")))


def ensure_bucket(s3_client, bucket: str, region: str) -> None:
    """Create a private, encrypted S3 bucket when it does not already exist."""

    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError as error:
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status not in {403, 404}:
            raise
        if status == 403:
            raise PermissionError(f"Bucket exists but is not accessible: {bucket}") from error

    create_args: dict[str, object] = {"Bucket": bucket}
    if region != "us-east-1":
        create_args["CreateBucketConfiguration"] = {
            "LocationConstraint": region,
        }
    s3_client.create_bucket(**create_args)
    s3_client.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3_client.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256",
                    }
                }
            ]
        },
    )


def wait_for_endpoint(
    sagemaker_client,
    endpoint_name: str,
    *,
    timeout_seconds: int = 3600,
    poll_seconds: int = 30,
) -> None:
    """Wait until a SageMaker endpoint is in service or fails."""

    deadline = time.monotonic() + timeout_seconds
    previous_status = None
    while time.monotonic() < deadline:
        description = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        status = description["EndpointStatus"]
        if status != previous_status:
            print(f"{endpoint_name}: {status}")
            previous_status = status
        if status == "InService":
            return
        if status in {"Failed", "OutOfService"}:
            reason = description.get("FailureReason", "No failure reason returned")
            raise RuntimeError(f"Endpoint {endpoint_name} failed: {reason}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for endpoint {endpoint_name}")


def sagemaker_resource_exists(
    sagemaker_client,
    describe_method: str,
    **kwargs,
) -> bool:
    """Return whether a named SageMaker resource exists."""

    try:
        getattr(sagemaker_client, describe_method)(**kwargs)
        return True
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code == "ValidationException":
            return False
        raise


def create_endpoint(
    sagemaker_client,
    *,
    model_name: str,
    endpoint_config_name: str,
    endpoint_name: str,
    role_arn: str,
    image_uri: str,
    model_id: str,
    instance_type: str,
    startup_timeout_seconds: int,
    async_output_path: str | None = None,
    async_failure_path: str | None = None,
) -> None:
    """Create missing resources for a SageMaker endpoint."""

    environment = {"SM_VLLM_MODEL": model_id}
    if model_id == VIDEO_MODEL_ID:
        environment["SM_VLLM_VAE_USE_TILING"] = "true"

    if sagemaker_resource_exists(
        sagemaker_client,
        "describe_model",
        ModelName=model_name,
    ):
        print(f"Reusing model {model_name}")
    else:
        sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                "Image": image_uri,
                "Environment": environment,
            },
        )

    endpoint_config: dict[str, object] = {
        "EndpointConfigName": endpoint_config_name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
                "ContainerStartupHealthCheckTimeoutInSeconds": (
                    startup_timeout_seconds
                ),
                "InferenceAmiVersion": INFERENCE_AMI_VERSION,
            }
        ],
    }
    if async_output_path:
        output_config = {"S3OutputPath": async_output_path}
        if async_failure_path:
            output_config["S3FailurePath"] = async_failure_path
        endpoint_config["AsyncInferenceConfig"] = {
            "OutputConfig": output_config,
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
        }

    if sagemaker_resource_exists(
        sagemaker_client,
        "describe_endpoint_config",
        EndpointConfigName=endpoint_config_name,
    ):
        print(f"Reusing endpoint configuration {endpoint_config_name}")
    else:
        sagemaker_client.create_endpoint_config(**endpoint_config)

    if sagemaker_resource_exists(
        sagemaker_client,
        "describe_endpoint",
        EndpointName=endpoint_name,
    ):
        print(f"Reusing endpoint {endpoint_name}")
    else:
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )


def invoke_image(
    runtime_client,
    state: DeploymentState,
    prompt: str,
    *,
    size: str = "1024x1024",
    steps: int = 4,
    seed: int = 42,
) -> bytes:
    """Generate an image through the real-time endpoint."""

    response = runtime_client.invoke_endpoint(
        EndpointName=state.image_endpoint_name,
        Body=json.dumps(
            build_image_payload(
                prompt,
                size=size,
                steps=steps,
                seed=seed,
            )
        ),
        ContentType="application/json",
        Accept="application/json",
        CustomAttributes=f"route={IMAGE_ROUTE}",
    )
    return decode_image_response(response["Body"].read())


def submit_video(
    runtime_client,
    s3_client,
    state: DeploymentState,
    prompt: str,
    image_bytes: bytes,
    *,
    width: int = 480,
    height: int = 320,
    num_frames: int = 17,
    fps: int = 8,
    steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = 42,
) -> tuple[str, str | None, str]:
    """Upload a multipart request and submit it to the async video endpoint."""

    reference_image = prepare_video_reference(
        image_bytes,
        width=width,
        height=height,
    )
    body, content_type = build_video_multipart(
        prompt,
        reference_image,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        image_media_type="image/jpeg",
    )
    request_key = f"{state.prefix}/requests/{uuid.uuid4().hex}.multipart"
    s3_client.put_object(
        Bucket=state.bucket,
        Key=request_key,
        Body=body,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )

    response = runtime_client.invoke_endpoint_async(
        EndpointName=state.video_endpoint_name,
        InputLocation=f"s3://{state.bucket}/{request_key}",
        ContentType=content_type,
        Accept="video/mp4",
        CustomAttributes=f"route={VIDEO_ROUTE}",
    )
    return response["OutputLocation"], response.get("FailureLocation"), request_key


def wait_for_s3_object(
    s3_client,
    uri: str,
    *,
    failure_uri: str | None = None,
    timeout_seconds: int = 3600,
    poll_seconds: int = 5,
) -> bytes:
    """Poll an async inference output location and return its bytes."""

    bucket, key = parse_s3_uri(uri)
    failure_bucket, failure_key = (
        parse_s3_uri(failure_uri) if failure_uri else (None, None)
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise

        if failure_bucket and failure_key:
            try:
                response = s3_client.get_object(
                    Bucket=failure_bucket,
                    Key=failure_key,
                )
                detail = response["Body"].read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Video generation failed: {detail}")
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code")
                if code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for {uri}")


def validate_mp4(video_bytes: bytes) -> None:
    """Raise when a response does not have an ISO base media file header."""

    if len(video_bytes) < 12 or video_bytes[4:8] != b"ftyp":
        raise ValueError("Video response is not an MP4 file")


def wait_for_endpoint_deleted(
    sagemaker_client,
    endpoint_name: str,
    *,
    timeout_seconds: int = 1800,
    poll_seconds: int = 15,
) -> None:
    """Wait until SageMaker no longer returns an endpoint description."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code == "ValidationException":
                return
            raise
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out deleting endpoint {endpoint_name}")


def delete_deployment(sagemaker_client, state: DeploymentState) -> None:
    """Delete endpoint resources, ignoring resources already removed."""

    endpoint_names = [
        state.video_endpoint_name,
        state.image_endpoint_name,
    ]
    for endpoint_name in endpoint_names:
        try:
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            message = error.response.get("Error", {}).get("Message", "")
            if code != "ValidationException" or "Could not find" not in message:
                raise
        wait_for_endpoint_deleted(sagemaker_client, endpoint_name)

    operations = [
        (
            sagemaker_client.delete_endpoint_config,
            {"EndpointConfigName": state.video_endpoint_config_name},
        ),
        (
            sagemaker_client.delete_endpoint_config,
            {"EndpointConfigName": state.image_endpoint_config_name},
        ),
        (
            sagemaker_client.delete_model,
            {"ModelName": state.video_model_name},
        ),
        (
            sagemaker_client.delete_model,
            {"ModelName": state.image_model_name},
        ),
    ]
    for operation, parameters in operations:
        try:
            operation(**parameters)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            message = error.response.get("Error", {}).get("Message", "")
            if code != "ValidationException" or "Could not find" not in message:
                raise
