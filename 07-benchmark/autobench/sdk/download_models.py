#!/usr/bin/env python3
"""
Model weight downloader — stages HuggingFace models to S3 for fast loading.

Downloads model weights from HuggingFace and uploads to S3 in the target region.
Run this before benchmarking to avoid timeout issues with large models (100B+).

Usage:
    python download_models.py --region us-east-2                    Download all models to us-east-2
    python download_models.py --region us-west-2 --model=kimi       Download only kimi to us-west-2
    python download_models.py --validate                            Show what would be downloaded
    python download_models.py --submit --region us-east-2           Submit as Processing Job (2TB disk)
    python download_models.py --submit --region us-east-2 --model=kimi

output: s3://sagemaker-benchmark-{region}-{account}/models/{model_key}/
note:   skips models already present in S3 (checks for config.json)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import boto3
import yaml

ENTRYPOINT_MODE = os.environ.get("SM_PROCESSING_MODE") == "model-download"


def load_config(path="../benchmarks.yaml"):
    if ENTRYPOINT_MODE:
        path = "/opt/ml/processing/input/config/benchmarks.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_account():
    return boto3.client("sts", region_name="us-east-2").get_caller_identity()["Account"]


def check_s3_exists(bucket, prefix, region):
    """Check if model already exists in S3."""
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix}config.json")
        return True
    except Exception:
        return False


def ensure_bucket(bucket, region):
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        try:
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
            print(f"  ✓ Created bucket: {bucket}")
        except Exception:
            pass


def download_and_upload(model_key, model_name, region, account, prefix_base="models"):
    """Download from HuggingFace and upload to S3."""
    bucket = f"sagemaker-benchmark-{region}-{account}"
    prefix = f"{prefix_base}/{model_key}/"

    # Skip if already present
    if check_s3_exists(bucket, prefix, region):
        print(f"  ⏭️  {model_key} already in s3://{bucket}/{prefix}")
        return f"s3://{bucket}/{prefix}"

    ensure_bucket(bucket, region)

    # Download
    local_dir = Path(f"/tmp/models/{model_key}")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading {model_name}...")
    subprocess.run(["pip", "install", "-q", "huggingface_hub[cli]"], check=True, capture_output=True)
    subprocess.run(["huggingface-cli", "download", model_name, "--local-dir", str(local_dir)], check=True)
    print(f"  ✓ Downloaded to {local_dir}")

    # Upload
    s3_uri = f"s3://{bucket}/{prefix}"
    print(f"  ⬆️  Uploading to {s3_uri}...")
    subprocess.run(["aws", "s3", "sync", str(local_dir), s3_uri, "--region", region], check=True)
    print(f"  ✓ Uploaded to {s3_uri}")
    return s3_uri


def submit_job(args):
    """Submit as a SageMaker Processing Job."""
    config = load_config(args.config)
    defaults = config.get("sagemaker_defaults", {})
    models = config.get("models", {})
    region = args.region
    role = defaults.get("role_arn")
    if not role:
        raise ValueError("'role_arn' must be set in sagemaker_defaults")
    account = get_account()
    bucket = f"sagemaker-benchmark-{region}-{account}"

    model_suffix = f"-{args.model}" if args.model else ""
    job_name = f"dl{model_suffix}-{region}-{datetime.now().strftime('%m%d-%H%M%S')}"[:63]

    # Upload config + script
    s3 = boto3.client("s3", region_name=region)
    ensure_bucket(bucket, region)
    config_key = f"processing-configs/{job_name}/benchmarks.yaml"
    script_key = f"processing-configs/{job_name}/download_models.py"
    s3.put_object(Bucket=bucket, Key=config_key, Body=open(args.config).read())
    s3.put_object(Bucket=bucket, Key=script_key, Body=open(__file__).read())

    container_args = ["--region", region]
    if args.model:
        container_args += ["--model", args.model]

    sm = boto3.client("sagemaker", region_name=region)

    print(f"\n{'='*60}")
    print(f"Submitting download job: {job_name}")
    print(f"  Region: {region}")
    print(f"  Models: {args.model or 'all'}")
    print(f"  Instance: ml.m7i.8xlarge (2TB disk)")
    print(f"{'='*60}\n")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m7i.8xlarge",
                "VolumeSizeInGB": 2000,
            }
        },
        AppSpecification={
            "ImageUri": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-cpu-py311",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/script/download_models.py"],
            "ContainerArguments": container_args,
        },
        Environment={"SM_PROCESSING_MODE": "model-download"},
        ProcessingInputs=[
            {
                "InputName": "config",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{config_key}",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "script",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{script_key}",
                    "LocalPath": "/opt/ml/processing/input/script",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [{
                "OutputName": "dummy",
                "S3Output": {
                    "S3Uri": f"s3://{bucket}/processing-configs/{job_name}/output/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }]
        },
        RoleArn=role,
        StoppingCondition={"MaxRuntimeInSeconds": 432000},
        NetworkConfig={"EnableNetworkIsolation": False},
    )

    print(f"  ✓ Job submitted: {job_name}")
    print(f"  Monitor: aws sagemaker describe-processing-job --processing-job-name {job_name} --region {region}")


def run_downloads(region, model_filter=None, target="smai", config_path="../benchmarks.yaml"):
    """Download all (or filtered) models to S3."""
    config = load_config(config_path)
    account = get_account()
    prefix_base = "models"

    # All models come from the models section
    targets = []
    for key, model in sorted(config.get("models", {}).items()):
        if model_filter and model_filter not in key:
            continue
        targets.append((key, model["model_name"]))

    print(f"\nDownloading {len(targets)} model(s) to region: {region} (target: {target})")
    print(f"Destination: s3://sagemaker-benchmark-{region}-{account}/{prefix_base}/\n")

    results = []
    for model_key, model_name in targets:
        print(f"\n{'─'*60}")
        print(f"[{model_key}] {model_name}")
        print(f"{'─'*60}")
        try:
            uri = download_and_upload(model_key, model_name, region, account, prefix_base)
            results.append({"model": model_key, "s3_uri": uri, "success": True})
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({"model": model_key, "error": str(e), "success": False})

    # Summary
    print(f"\n{'═'*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'═'*60}")
    ok = sum(1 for r in results if r["success"])
    print(f"Total: {len(results)} | Success: {ok} | Failed: {len(results) - ok}")
    if target == "byom":
        print(f"\nUpdate benchmarks.yaml byom_models weights_s3_uri fields:")
    else:
        print(f"\nUpdate benchmarks.yaml models s3_model_uri fields:")
    for r in results:
        if r["success"]:
            print(f"  {r['model']}: \"{r['s3_uri']}\"")


def main():
    if ENTRYPOINT_MODE:
        parser = argparse.ArgumentParser()
        parser.add_argument("--region", required=True)
        parser.add_argument("--model", default=None)
        args = parser.parse_args()
        subprocess.run(["pip", "install", "-q", "huggingface_hub[cli]", "pyyaml"], check=True, capture_output=True)
        run_downloads(args.region, args.model)
        return

    parser = argparse.ArgumentParser(
        description="Download model weights from HuggingFace and stage in S3 for fast loading.",
        epilog="""
examples:
  %(prog)s --region us-east-2                         Download all models to us-east-2
  %(prog)s --region us-west-2 --model=kimi            Download only kimi models
  %(prog)s --validate --region us-east-2              Show what would be downloaded
  %(prog)s --submit --region us-east-2                Submit as Processing Job (2TB disk)
  %(prog)s --submit --region us-east-2 --model=kimi   Submit single model download

output: s3://sagemaker-benchmark-{region}-{account}/models/{model_key}/
note:   after download, update s3_model_uri in benchmarks.yaml for each model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default="../benchmarks.yaml")
    parser.add_argument("--region", required=True, help="Target S3 region (should match FTP/cluster region)")
    parser.add_argument("--model", help="Filter by model key substring")
    parser.add_argument("--validate", action="store_true", help="Show what would be downloaded (no action)")
    parser.add_argument("--submit", action="store_true", help="Submit as Processing Job")
    args = parser.parse_args()

    if args.submit:
        submit_job(args)
        return

    config = load_config(args.config)
    models = config.get("models", {})
    account = get_account()
    bucket = f"sagemaker-benchmark-{args.region}-{account}"

    if args.validate:
        print(f"Models to download → s3://{bucket}/models/\n")
        for key, model in sorted(models.items()):
            if args.model and args.model not in key:
                continue
            existing = check_s3_exists(bucket, f"models/{key}/", args.region)
            status = "✓ in S3" if existing else "○ pending"
            print(f"  {status}  {key} ({model['model_name']})")
        return

    run_downloads(args.region, args.model, config_path=args.config)


if __name__ == "__main__":
    main()
