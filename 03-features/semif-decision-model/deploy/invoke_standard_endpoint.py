#!/usr/bin/env python3
"""Invoke an AWS Deep Learning Container for llama.cpp endpoint."""

from __future__ import annotations

import argparse
import json


def invoke(
    *,
    region: str,
    endpoint_name: str,
    prompt: str,
    max_tokens: int = 64,
) -> dict:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")

    import boto3

    runtime = boto3.Session(region_name=region).client("sagemaker-runtime")
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode(),
    )
    return json.loads(response["Body"].read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "A customer cannot receive a password reset email. "
            "Reply with the best support queue."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            invoke(
                region=args.region,
                endpoint_name=args.endpoint_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
