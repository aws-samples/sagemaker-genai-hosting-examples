#!/usr/bin/env python3
"""Delete the SageMaker resources created by deploy.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3

from vllm_omni_media import (
    DEFAULT_STATE_PATH,
    delete_deployment,
    load_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(args.state_file)
    sagemaker = boto3.client("sagemaker", region_name=state.region)
    delete_deployment(sagemaker, state)
    args.state_file.unlink(missing_ok=True)
    print("Deleted the image and video endpoint resources.")
    print(
        f"Generated artifacts remain under s3://{state.bucket}/{state.prefix}/outputs/."
    )


if __name__ == "__main__":
    main()
