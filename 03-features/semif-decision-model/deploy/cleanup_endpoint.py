#!/usr/bin/env python3
"""Delete one explicitly named SageMaker endpoint, config, and model."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    import boto3

    sm = boto3.Session(region_name=args.region).client("sagemaker")
    sm.delete_endpoint(EndpointName=args.name)
    sm.get_waiter("endpoint_deleted").wait(EndpointName=args.name)
    sm.delete_endpoint_config(EndpointConfigName=args.name)
    sm.delete_model(ModelName=args.name)
    print(f"Deleted SageMaker resources named {args.name} in {args.region}")


if __name__ == "__main__":
    main()
