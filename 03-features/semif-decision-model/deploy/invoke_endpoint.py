#!/usr/bin/env python3
"""Invoke a deployed SemIf endpoint with the example decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument(
        "--request",
        type=Path,
        default=Path(__file__).parents[1] / "example-request.json",
    )
    args = parser.parse_args()
    import boto3

    runtime = boto3.Session(region_name=args.region).client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=args.request.read_bytes(),
    )
    print(json.dumps(json.loads(response["Body"].read()), indent=2))


if __name__ == "__main__":
    main()
