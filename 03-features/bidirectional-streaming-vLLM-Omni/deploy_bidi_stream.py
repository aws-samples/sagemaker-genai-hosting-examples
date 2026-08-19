"""Deploy vLLM-Omni and stream text-to-speech through SageMaker AI.

This sample deploys the vLLM-Omni SageMaker Deep Learning Container, opens a
SageMaker bidirectional stream to the native TTS WebSocket route, sends text,
validates the streamed audio response, and deletes the created resources.

Prerequisites:
    Python 3.12 or newer
    An AWS identity that can create and invoke SageMaker AI endpoints
    A SageMaker AI execution role

Install:
    python -m pip install -r requirements.txt
"""

import argparse
import asyncio
import json
import logging
import os
import time
import wave

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from sagemaker_bidi_tts import collect_pcm

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-east-1")
IMAGE_URI = os.environ.get(
    "VLLM_OMNI_IMAGE_URI",
    (f"763104351884.dkr.ecr.{REGION}.amazonaws.com/vllm:omni-sagemaker-cuda-v1.5"),
)
EXECUTION_ROLE_ARN = os.environ.get(
    "SAGEMAKER_EXECUTION_ROLE_ARN",
    "arn:aws:iam::<account-id>:role/<sagemaker-execution-role>",
)
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
INSTANCE_TYPE = "ml.g6.xlarge"
INFERENCE_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1"


def validate_configuration():
    """Fail before resource creation when required values are placeholders."""
    if "<" in EXECUTION_ROLE_ARN or ">" in EXECUTION_ROLE_ARN:
        raise ValueError(
            "Set SAGEMAKER_EXECUTION_ROLE_ARN to a SageMaker execution role."
        )


def save_wav(path, pcm, sample_rate):
    """Write mono 16-bit PCM as a WAV file."""
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def delete_resources(resource_name):
    """Delete an endpoint, endpoint configuration, and model by name."""
    sagemaker = boto3.client("sagemaker", region_name=REGION)
    try:
        sagemaker.delete_endpoint(EndpointName=resource_name)
        sagemaker.get_waiter("endpoint_deleted").wait(
            EndpointName=resource_name,
            WaiterConfig={"Delay": 15, "MaxAttempts": 80},
        )
        print(f"Deleted endpoint: {resource_name}")
    except ClientError as error:
        if "Could not find" not in str(error):
            raise

    for operation, label, argument in [
        (
            sagemaker.delete_endpoint_config,
            "endpoint configuration",
            {"EndpointConfigName": resource_name},
        ),
        (
            sagemaker.delete_model,
            "model",
            {"ModelName": resource_name},
        ),
    ]:
        try:
            operation(**argument)
            print(f"Deleted {label}: {resource_name}")
        except ClientError as error:
            if "Could not find" not in str(error):
                raise


def main():
    """Deploy the endpoint, run one streaming request, and clean up."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name")
    parser.add_argument("--keep-endpoint", action="store_true")
    parser.add_argument("--delete-endpoint", action="store_true")
    parser.add_argument(
        "--text",
        default=(
            "Hello, this response is streaming from vLLM-Omni on Amazon SageMaker AI."
        ),
    )
    parser.add_argument("--output", default="validation-output.wav")
    args = parser.parse_args()

    resource_name = args.endpoint_name or f"vllm-omni-bidi-{int(time.time())}"
    if args.delete_endpoint:
        if not args.endpoint_name:
            parser.error("--delete-endpoint requires --endpoint-name")
        delete_resources(resource_name)
        return

    validate_configuration()
    model_created = False
    endpoint_config_created = False
    endpoint_created = False
    validation_passed = False

    try:
        print(f"Region: {REGION}")
        print(f"Image: {IMAGE_URI}")
        print(f"Model: {MODEL_ID}")
        print(f"Instance type: {INSTANCE_TYPE}")
        Model.create(
            model_name=resource_name,
            primary_container=ContainerDefinition(
                image=IMAGE_URI,
                environment={"SM_VLLM_MODEL": MODEL_ID},
            ),
            execution_role_arn=EXECUTION_ROLE_ARN,
            region=REGION,
        )
        model_created = True
        EndpointConfig.create(
            endpoint_config_name=resource_name,
            region=REGION,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=resource_name,
                    initial_instance_count=1,
                    instance_type=INSTANCE_TYPE,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                )
            ],
        )
        endpoint_config_created = True
        endpoint = Endpoint.create(
            endpoint_name=resource_name,
            endpoint_config_name=resource_name,
            region=REGION,
        )
        endpoint_created = True
        print(f"Deploying {resource_name}.")
        endpoint.wait_for_status("InService", timeout=2400)

        result = asyncio.run(
            collect_pcm(
                resource_name,
                args.text,
                region=REGION,
            )
        )
        audio_bytes = len(result["audio"])
        print(
            "Stream result:",
            json.dumps(
                {
                    "audio_bytes": audio_bytes,
                    "sample_rate": result["sample_rate"],
                    "saw_start": result["saw_start"],
                    "saw_done": result["saw_done"],
                }
            ),
        )
        if not result["saw_start"] or not result["saw_done"]:
            raise RuntimeError("The expected audio lifecycle events were missing.")
        if audio_bytes <= 2_000:
            raise RuntimeError("The endpoint returned too little audio.")
        save_wav(args.output, result["audio"], result["sample_rate"])
        print(f"Saved validation audio: {args.output}")
        validation_passed = True
    finally:
        if args.keep_endpoint and validation_passed:
            print(f"Kept endpoint: {resource_name}")
            print(
                "Launch the UI with: python sagemaker_bidi_tts_client.py "
                f"--endpoint-name {resource_name} --region {REGION}"
            )
            print(
                "Delete resources with: python deploy_bidi_stream.py "
                f"--endpoint-name {resource_name} --delete-endpoint"
            )
        elif model_created or endpoint_config_created or endpoint_created:
            try:
                delete_resources(resource_name)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Failed to delete %s: %s",
                    resource_name,
                    error,
                )


if __name__ == "__main__":
    main()
