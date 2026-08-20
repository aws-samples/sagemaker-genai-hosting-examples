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
from sagemaker_bidi_tts import collect_pcm

logger = logging.getLogger(__name__)

EXECUTION_ROLE_ARN = os.environ.get(
    "SAGEMAKER_EXECUTION_ROLE_ARN",
    "arn:aws:iam::<account-id>:role/<sagemaker-execution-role>",
)
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
INSTANCE_TYPES = (
    "ml.g6.xlarge",
    "ml.g6e.xlarge",
    "ml.g5.xlarge",
    "ml.g4dn.xlarge",
)
INFERENCE_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1"
INSTANCE_PROVISION_TIMEOUT_SECONDS = 1200


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


def image_uri(region):
    """Return the regional vLLM-Omni DLC image URI."""
    return os.environ.get(
        "VLLM_OMNI_IMAGE_URI",
        (f"763104351884.dkr.ecr.{region}.amazonaws.com/vllm:omni-sagemaker-cuda-v1.5"),
    )


def instance_pools():
    """Return compatible GPU instance types in placement priority order."""
    return [
        {"InstanceType": instance_type, "Priority": priority}
        for priority, instance_type in enumerate(INSTANCE_TYPES, start=1)
    ]


def production_variant(resource_name):
    """Build the endpoint production variant with capacity fallbacks."""
    return {
        "VariantName": "AllTraffic",
        "ModelName": resource_name,
        "InitialInstanceCount": 1,
        "InstancePools": instance_pools(),
        "VariantInstanceProvisionTimeoutInSeconds": (
            INSTANCE_PROVISION_TIMEOUT_SECONDS
        ),
        "InferenceAmiVersion": INFERENCE_AMI_VERSION,
    }


def delete_resources(resource_name, region):
    """Delete an endpoint, endpoint configuration, and model by name."""
    sagemaker = boto3.client("sagemaker", region_name=region)
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
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
    )
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
        delete_resources(resource_name, args.region)
        return

    validate_configuration()
    region = args.region
    regional_image_uri = image_uri(region)
    model_created = False
    endpoint_config_created = False
    endpoint_created = False
    validation_passed = False

    try:
        print(f"Region: {region}")
        print(f"Image: {regional_image_uri}")
        print(f"Model: {MODEL_ID}")
        print(f"Instance pools: {', '.join(INSTANCE_TYPES)}")
        sagemaker = boto3.client("sagemaker", region_name=region)
        sagemaker.create_model(
            ModelName=resource_name,
            PrimaryContainer={
                "Image": regional_image_uri,
                "Environment": {"SM_VLLM_MODEL": MODEL_ID},
            },
            ExecutionRoleArn=EXECUTION_ROLE_ARN,
        )
        model_created = True
        sagemaker.create_endpoint_config(
            EndpointConfigName=resource_name,
            ProductionVariants=[production_variant(resource_name)],
        )
        endpoint_config_created = True
        sagemaker.create_endpoint(
            EndpointName=resource_name,
            EndpointConfigName=resource_name,
        )
        endpoint_created = True
        print(f"Deploying {resource_name}.")
        sagemaker.get_waiter("endpoint_in_service").wait(
            EndpointName=resource_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 80},
        )
        endpoint_description = sagemaker.describe_endpoint(EndpointName=resource_name)
        selected_pools = endpoint_description["ProductionVariants"][0].get(
            "InstancePools",
            [],
        )
        print("Selected instance pools:", json.dumps(selected_pools, default=str))

        result = asyncio.run(
            collect_pcm(
                resource_name,
                args.text,
                region=region,
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
                f"--endpoint-name {resource_name} --region {region}"
            )
            print(
                "Delete resources with: python deploy_bidi_stream.py "
                f"--endpoint-name {resource_name} --region {region} --delete-endpoint"
            )
        elif model_created or endpoint_config_created or endpoint_created:
            try:
                delete_resources(resource_name, region)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Failed to delete %s: %s",
                    resource_name,
                    error,
                )


if __name__ == "__main__":
    main()
