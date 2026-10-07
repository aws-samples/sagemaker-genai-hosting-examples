#!/usr/bin/env python3
"""Deploy a GGUF model with the AWS Deep Learning Container for llama.cpp."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

IMAGE_DIGEST_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def validate_image_uri(image_uri: str) -> None:
    if not IMAGE_DIGEST_PATTERN.fullmatch(image_uri):
        raise ValueError(
            "Image URI must contain a complete lowercase sha256 digest"
        )


def deploy(
    *,
    region: str,
    name: str,
    image_uri: str,
    model_data_url: str,
    role_arn: str,
    instance_type: str,
    context_size: int = 4096,
) -> dict:
    validate_image_uri(image_uri)
    if not model_data_url.startswith("s3://") or not model_data_url.endswith(
        "/model.tar.gz"
    ):
        raise ValueError("Model data URL must end with /model.tar.gz in S3")
    if not instance_type.startswith("ml."):
        raise ValueError("instance_type must be a SageMaker ml.* instance type")
    if context_size < 1:
        raise ValueError("context_size must be a positive integer")

    import boto3

    sm = boto3.Session(region_name=region).client("sagemaker")
    sm.create_model(
        ModelName=name,
        ExecutionRoleArn=role_arn,
        EnableNetworkIsolation=True,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SM_LLAMA_CPP_CTX_SIZE": str(context_size),
            },
        },
    )
    sm.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "ContainerStartupHealthCheckTimeoutInSeconds": 900,
                "ModelDataDownloadTimeoutInSeconds": 900,
            }
        ],
    )
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name)
    return {
        "endpoint_name": name,
        "model_name": name,
        "endpoint_config_name": name,
        "region": region,
        "instance_type": instance_type,
        "status": "Creating",
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--name", default=f"llamacpp-standard-{timestamp}")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--model-data-url", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--instance-type", default="ml.c6i.2xlarge")
    parser.add_argument("--context-size", type=int, default=4096)
    args = parser.parse_args()
    print(
        json.dumps(
            deploy(
                region=args.region,
                name=args.name,
                image_uri=args.image_uri,
                model_data_url=args.model_data_url,
                role_arn=args.role_arn,
                instance_type=args.instance_type,
                context_size=args.context_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
