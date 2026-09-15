#!/usr/bin/env python3
"""Deploy FLUX.2-klein and Wan VACE with the vLLM-Omni DLC."""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

import boto3

from vllm_omni_media import (
    DEFAULT_STATE_PATH,
    IMAGE_MODEL_ID,
    VIDEO_MODEL_ID,
    DeploymentState,
    container_image_uri,
    create_endpoint,
    ensure_bucket,
    load_state,
    save_state,
    wait_for_endpoint,
)


def resource_name(prefix: str, kind: str, timestamp: str) -> str:
    """Build a SageMaker-compatible resource name."""

    value = re.sub(r"[^A-Za-z0-9-]", "-", f"{prefix}-{kind}-{timestamp}")
    return value[:63].rstrip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy vLLM-Omni image and video endpoints on SageMaker AI."
    )
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--bucket",
        help="S3 bucket for async requests and outputs. Defaults to the SageMaker convention.",
    )
    parser.add_argument("--prefix", default="vllm-omni-image-video")
    parser.add_argument("--name-prefix", default="vllm-omni-media")
    parser.add_argument("--image-instance-type", default="ml.g6.xlarge")
    parser.add_argument("--video-instance-type", default="ml.g6e.xlarge")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the deployment recorded in --state-file.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = boto3.Session(region_name=args.region)
    sagemaker = session.client("sagemaker")
    s3 = session.client("s3")

    if args.resume:
        state = load_state(args.state_file)
        if state.region != args.region:
            raise ValueError(
                f"State file uses {state.region}, but --region is {args.region}."
            )
        bucket = state.bucket
        print(f"Resuming deployment from {args.state_file}")
    else:
        if args.state_file.exists():
            raise FileExistsError(
                f"State file already exists at {args.state_file}. "
                "Use --resume or remove it after cleanup."
            )
        sts = session.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        bucket = args.bucket or f"sagemaker-{args.region}-{account_id}"
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        state = DeploymentState(
            region=args.region,
            bucket=bucket,
            prefix=args.prefix.strip("/"),
            image_model_name=resource_name(
                args.name_prefix, "image-model", timestamp
            ),
            image_endpoint_config_name=resource_name(
                args.name_prefix, "image-config", timestamp
            ),
            image_endpoint_name=resource_name(
                args.name_prefix, "image-endpoint", timestamp
            ),
            video_model_name=resource_name(
                args.name_prefix, "video-model", timestamp
            ),
            video_endpoint_config_name=resource_name(
                args.name_prefix, "video-config", timestamp
            ),
            video_endpoint_name=resource_name(
                args.name_prefix, "video-endpoint", timestamp
            ),
        )
        save_state(state, args.state_file)

    ensure_bucket(s3, bucket, args.region)

    image_uri = container_image_uri(args.region)
    print(f"Preparing image endpoint {state.image_endpoint_name}")
    create_endpoint(
        sagemaker,
        model_name=state.image_model_name,
        endpoint_config_name=state.image_endpoint_config_name,
        endpoint_name=state.image_endpoint_name,
        role_arn=args.role_arn,
        image_uri=image_uri,
        model_id=IMAGE_MODEL_ID,
        instance_type=args.image_instance_type,
        startup_timeout_seconds=1800,
    )
    wait_for_endpoint(sagemaker, state.image_endpoint_name)

    print(f"Preparing video endpoint {state.video_endpoint_name}")
    create_endpoint(
        sagemaker,
        model_name=state.video_model_name,
        endpoint_config_name=state.video_endpoint_config_name,
        endpoint_name=state.video_endpoint_name,
        role_arn=args.role_arn,
        image_uri=image_uri,
        model_id=VIDEO_MODEL_ID,
        instance_type=args.video_instance_type,
        startup_timeout_seconds=3600,
        async_output_path=f"s3://{bucket}/{state.prefix}/outputs/",
        async_failure_path=f"s3://{bucket}/{state.prefix}/failures/",
    )
    wait_for_endpoint(sagemaker, state.video_endpoint_name)

    print(f"Deployment state written to {args.state_file}")
    print(f"Image endpoint: {state.image_endpoint_name}")
    print(f"Video endpoint: {state.video_endpoint_name}")


if __name__ == "__main__":
    main()
