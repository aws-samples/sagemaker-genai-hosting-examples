#!/usr/bin/env python3
"""Deploy a digest-pinned SemIf image and model archive to SageMaker AI."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone

TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
IMAGE_DIGEST_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def validate_image_uri(image_uri: str) -> None:
    if not IMAGE_DIGEST_PATTERN.fullmatch(image_uri):
        raise ValueError(
            "Image URI must contain a complete lowercase sha256 digest"
        )


def _is_instance_pool_model_lag(error: Exception) -> bool:
    message = str(error)
    return (
        error.__class__.__name__ == "ParamValidationError"
        and "InstancePools" in message
        and "VariantInstanceProvisionTimeoutInSeconds" in message
    )


def _create_endpoint_config_raw(
    *,
    session,
    endpoint_url: str,
    region: str,
    payload: dict,
) -> None:
    """Call the service when Botocore's local model trails the API."""
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    body = json.dumps(payload, separators=(",", ":")).encode()
    request = AWSRequest(
        method="POST",
        url=endpoint_url,
        data=body,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "SageMaker.CreateEndpointConfig",
        },
    )
    credentials = session.get_credentials().get_frozen_credentials()
    SigV4Auth(credentials, "sagemaker", region).add_auth(request)
    prepared = request.prepare()
    url_request = urllib.request.Request(
        prepared.url,
        data=prepared.body,
        headers=dict(prepared.headers.items()),
        method=prepared.method,
    )
    try:
        with urllib.request.urlopen(url_request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"SageMaker CreateEndpointConfig failed: HTTP {error.code}: {detail}"
        ) from error


def deploy(
    *,
    region: str,
    name: str,
    image_uri: str,
    model_data_url: str,
    role_arn: str,
    instance_type: str | None = None,
    instance_pools: list[str] | None = None,
    threads: int | None = None,
    gpu_layers: int = 0,
    variant_instance_provision_timeout_seconds: int = 1200,
) -> dict:
    validate_image_uri(image_uri)
    if not model_data_url.startswith("s3://") or not model_data_url.endswith(
        "/model.tar.gz"
    ):
        raise ValueError("Model data URL must end with /model.tar.gz in S3")
    if threads is not None and threads < 1:
        raise ValueError("threads must be a positive integer")
    if gpu_layers < -1:
        raise ValueError("gpu_layers must be -1 or a non-negative integer")
    if (instance_type is None) == (not instance_pools):
        raise ValueError("provide exactly one of instance_type or instance_pools")
    if instance_pools and any(not value.startswith("ml.") for value in instance_pools):
        raise ValueError("instance_pools must contain SageMaker ml.* instance types")
    if variant_instance_provision_timeout_seconds < 1:
        raise ValueError(
            "variant_instance_provision_timeout_seconds must be positive"
        )

    import boto3

    environment = {
        "SEMIF_TOKENIZER_REVISION": TOKENIZER_REVISION,
        "SEMIF_MAX_TOKENS": "4096",
        "SEMIF_MAX_REQUEST_TOKENS": "8192",
        "SEMIF_MAX_SHARED_DECISIONS": "8",
        "SEMIF_MAX_BODY_BYTES": str(256 * 1024),
        "SEMIF_N_GPU_LAYERS": str(gpu_layers),
    }
    if threads is not None:
        environment["SEMIF_THREADS"] = str(threads)

    session = boto3.Session(region_name=region)
    sm = session.client("sagemaker")
    sm.create_model(
        ModelName=name,
        ExecutionRoleArn=role_arn,
        EnableNetworkIsolation=True,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": environment,
        },
    )
    production_variant = {
        "VariantName": "AllTraffic",
        "ModelName": name,
        "InitialInstanceCount": 1,
        "ContainerStartupHealthCheckTimeoutInSeconds": 900,
        "ModelDataDownloadTimeoutInSeconds": 900,
        "VariantInstanceProvisionTimeoutInSeconds": (
            variant_instance_provision_timeout_seconds
        ),
    }
    if instance_pools:
        production_variant["InstancePools"] = [
            {"InstanceType": value, "Priority": priority}
            for priority, value in enumerate(instance_pools, start=1)
        ]
    else:
        production_variant["InstanceType"] = instance_type
    endpoint_config = {
        "EndpointConfigName": name,
        "ProductionVariants": [production_variant],
    }
    try:
        sm.create_endpoint_config(**endpoint_config)
    except Exception as error:
        if not instance_pools or not _is_instance_pool_model_lag(error):
            raise
        _create_endpoint_config_raw(
            session=session,
            endpoint_url=sm.meta.endpoint_url,
            region=region,
            payload=endpoint_config,
        )
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name)
    return {
        "endpoint_name": name,
        "model_name": name,
        "endpoint_config_name": name,
        "region": region,
        **(
            {"instance_pools": instance_pools}
            if instance_pools
            else {"instance_type": instance_type}
        ),
        "gpu_layers": gpu_layers,
        "status": "Creating",
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--name", default=f"semif-llamacpp-{timestamp}")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--model-data-url", required=True)
    parser.add_argument("--role-arn", required=True)
    placement = parser.add_mutually_exclusive_group(required=True)
    placement.add_argument("--instance-type")
    placement.add_argument(
        "--instance-pool",
        action="append",
        dest="instance_pools",
        help="Ordered SageMaker instance type; repeat for fallbacks",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Positive llama.cpp CPU thread count; omit to use its default",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="llama.cpp layers to offload; -1 offloads all layers",
    )
    parser.add_argument(
        "--provision-timeout",
        type=int,
        default=1200,
        help="Seconds SageMaker may spend provisioning across the instance pool",
    )
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
                instance_pools=args.instance_pools,
                threads=args.threads,
                gpu_layers=args.gpu_layers,
                variant_instance_provision_timeout_seconds=args.provision_timeout,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
