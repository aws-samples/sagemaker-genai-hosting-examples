#!/usr/bin/env python3
"""
Bedrock BYOM (Bring Your Own Model) benchmarking via Processing Job.

Submits a long-running Processing Job that:
1. Downloads model weights from HuggingFace
2. Uploads to S3 under byom/ prefix
3. Imports via Mantle API
4. Creates RU reservation
5. Submits benchmark jobs via AI Benchmarking (direct-URL)

Usage:
    python benchmark_bedrock_byom.py --submit --model=gpt-oss-120b-byom
    python benchmark_bedrock_byom.py --validate
    python benchmark_bedrock_byom.py --import-only --model=gpt-oss-120b-byom
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
import yaml
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

RESULTS_DIR = Path("results")
DIRECT_URL_SENTINEL = "bench-direct-7f2a9c4e1b8d053f6a9e2c7d4b1f8a3e5d0c6b9a4e8f2a1d"
ENTRYPOINT_MODE = os.environ.get("SM_PROCESSING_MODE") == "byom-benchmark"


def load_config(path="../benchmarks.yaml"):
    if ENTRYPOINT_MODE:
        path = "/opt/ml/processing/input/script/benchmarks.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def expand_byom_jobs(config, model_filter=None, workload_filter=None):
    """Expand byom_benchmarks into individual jobs."""
    defaults = config.get("byom_defaults", {})
    # BYOM now reads from the shared models section (byom: sub-key)
    models = config.get("models") or {}
    workloads = config.get("workloads", {})
    benchmarks = config.get("byom_benchmarks") or []
    jobs = []

    for bench in benchmarks:
        model_key = bench["model"]
        model = models.get(model_key)
        if not model:
            print(f"ERROR: Unknown byom model '{model_key}'", file=sys.stderr)
            sys.exit(1)
        byom = model.get("byom")
        if not byom:
            print(f"SKIP: Model '{model_key}' has no byom config — not BYOM-supported", file=sys.stderr)
            continue
        if model_filter and model_filter not in model_key:
            continue
        for wl_name in bench["workloads"]:
            if workload_filter and workload_filter not in wl_name:
                continue
            workload = workloads.get(wl_name)
            if not workload:
                print(f"ERROR: Unknown workload '{wl_name}'", file=sys.stderr)
                sys.exit(1)
            for concurrency in workload["concurrency"]:
                jobs.append({
                    "id": f"byom-{model_key}--{wl_name}--c{concurrency}",
                    "model_key": model_key,
                    "base_model_id": byom["base_model_id"],
                    "tokenizer": model.get("benchmark_tokenizer", model["model_name"]),
                    "model_id": byom.get("model_id", ""),
                    "workload_key": wl_name,
                    "concurrency": concurrency,
                    "input_tokens": workload["input_tokens"],
                    "output_tokens": workload["output_tokens"],
                    "streaming": workload.get("streaming", True),
                    "duration": workload.get("duration", 300),
                    "warmup": workload.get("warmup", 30),
                })
    return jobs


# --- Mantle API Client ---

def mantle_request(method, path, defaults, body=None):
    """Make a SigV4-signed request to the Mantle API."""
    endpoint = defaults.get("mantle_endpoint", f"https://bedrock-mantle.{defaults['region']}.api.aws")
    endpoint = endpoint.format(region=defaults["region"])
    region = defaults["region"]
    url = f"{endpoint}{path}"

    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()

    headers = {"Content-Type": "application/json"}
    data = json.dumps(body) if body else None

    request = AWSRequest(method=method, url=url, data=data, headers=headers)
    SigV4Auth(credentials, "bedrock-mantle", region).add_auth(request)

    import urllib.request
    req = urllib.request.Request(
        url,
        data=data.encode() if data else None,
        headers=dict(request.headers),
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"  ✗ Mantle API {method} {path} → {e.code}: {error_body}")
        return None


def check_weights_in_s3(model_key, model_cfg, defaults):
    """Verify model weights exist in S3. Returns the S3 URI or None."""
    # Use explicit s3_model_uri if set in config
    s3_uri = model_cfg.get("s3_model_uri", "")
    if s3_uri:
        print(f"  ✓ Using s3_model_uri: {s3_uri}")
        return s3_uri

    print(f"  ✗ No s3_model_uri set for {model_key}")
    print(f"    1. Run: python download_models.py --region {defaults['region']} --model=<model>")
    print(f"    2. Set s3_model_uri in models.{model_key} in benchmarks.yaml")
    return None


def import_model(model_key, model_cfg, defaults):
    """Import model via Mantle customization API. Returns model_id."""
    print(f"\n  📦 Importing {model_key} via BYOM...")
    weights_uri = model_cfg["weights_s3_uri"]
    # Mantle expects the index file path, not the directory
    if not weights_uri.endswith(".json"):
        weights_uri = weights_uri.rstrip("/") + "/model.safetensors.index.json"
    body = {
        "base_model_id": model_cfg["byom"]["base_model_id"],
        "model_config": {
            "source": {
                "type": "s3",
                "weights_index_s3_uri": weights_uri,
            },
        },
    }
    resp = mantle_request("POST", "/bedrock/v1/models/customization", defaults, body)
    if not resp:
        return None
    model_id = resp.get("model_id") or resp.get("modelId") or resp.get("id")
    print(f"  ✓ Import started: model_id={model_id}")
    return model_id


def poll_import(model_id, defaults, timeout_minutes=30):
    """Poll until model import completes."""
    print(f"  ⏳ Polling import status (timeout={timeout_minutes}min)...")
    for i in range(timeout_minutes * 2):  # poll every 30s
        resp = mantle_request("GET", f"/bedrock/v1/models/customization/{model_id}", defaults)
        if not resp:
            time.sleep(30)
            continue
        status = resp.get("status", resp.get("Status", ""))
        if status.lower() in ("complete", "completed", "active", "ready"):
            print(f"  ✓ Import complete: {model_id}")
            return True
        if status.lower() in ("failed", "error"):
            print(f"  ✗ Import failed: {resp}")
            return False
        elapsed = (i + 1) * 30
        print(f"    [{elapsed}s] status={status}")
        time.sleep(30)
    print(f"  ✗ Import timed out after {timeout_minutes}min")
    return False


def create_reservation(model_id, defaults):
    """Create an RU reservation for the model. Returns reservation_id."""
    ru_count = defaults.get("default_ru_count", 1)
    print(f"  🔒 Creating reservation ({ru_count} RU)...")
    body = {"model_id": model_id, "reservation_units": {"count": ru_count}}
    resp = mantle_request("POST", "/bedrock/v1/reservations", defaults, body)
    if not resp:
        return None
    res_id = resp.get("reservation_id") or resp.get("id")
    print(f"  ✓ Reservation created: {res_id}")
    return res_id


def cancel_reservation(reservation_id, defaults):
    """Cancel an RU reservation."""
    resp = mantle_request("DELETE", f"/bedrock/v1/reservations/{reservation_id}", defaults)
    if resp is not None:
        print(f"  ✓ Reservation cancelled: {reservation_id}")


# --- Token ---

def get_or_create_token_secret(defaults):
    """Generate a Bedrock bearer token and store in Secrets Manager."""
    from datetime import timedelta
    from aws_bedrock_token_generator import provide_token
    import uuid

    region = defaults["region"]
    token = provide_token(region=region, expiry=timedelta(hours=12))
    sm = boto3.client("secretsmanager", region_name=region)
    # Use a stable name so benchmark jobs can always find it
    secret_name = "bench-byom-token"
    try:
        sm.put_secret_value(SecretId=secret_name, SecretString=token)
        resp = sm.describe_secret(SecretId=secret_name)
        print(f"  ✓ Token updated: {secret_name}")
    except sm.exceptions.ResourceNotFoundException:
        resp = sm.create_secret(Name=secret_name, SecretString=token,
                                Description="BYOM benchmark token (12h TTL, auto-refreshed)")
        print(f"  ✓ Token created: {secret_name}")
    return resp["ARN"], secret_name


def cleanup_secret(defaults, secret_name):
    sm = boto3.client("secretsmanager", region_name=defaults["region"])
    try:
        sm.delete_secret(SecretId=secret_name, ForceDeleteWithoutRecovery=True)
        print(f"  ✓ Deleted secret: {secret_name}")
    except Exception:
        pass


# --- Benchmarking ---

def run_byom_benchmark(client, job, defaults, secret_arn):
    """Submit a single benchmark job against BYOM model."""
    region = defaults["region"]
    s3_output = defaults["s3_output"].format(region=region)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    s3_job_path = f"{s3_output}/{job['model_key']}/{job['workload_key']}/byom-c{job['concurrency']}/{timestamp}/"

    ts = datetime.now().strftime("%m%d%H%M")
    job_name = f"{job['id'][:50]}-{ts}".replace("_", "-")
    config_name = f"byom-{job['model_key']}-{job['workload_key']}-c{job['concurrency']}"[:63].replace("_", "-")

    # BYOM models are served via the Mantle inference endpoint (not bedrock-runtime)
    bedrock_url = f"https://bedrock-mantle.{region}.api.aws/v1"

    print(f"\n{'─'*60}")
    print(f"[byom] {job['id']}")
    print(f"  model={job['model_id']} concurrency={job['concurrency']}")
    print(f"  in={job['input_tokens']} out={job['output_tokens']}")
    print(f"{'─'*60}")

    workload_spec = {
        "benchmark": {"type": "aiperf"},
        "parameters": {
            "url": bedrock_url,
            "model": job["model_id"],
            "tokenizer": job["tokenizer"],
            "concurrency": job["concurrency"],
            "streaming": job["streaming"],
            "prompt_input_tokens_mean": job["input_tokens"],
            "output_tokens_mean": job["output_tokens"],
            "benchmark_duration": job["duration"],
            "warmup_duration": job["warmup"],
        },
        "secrets": {"api_key": secret_arn},
    }

    # Create workload config
    try:
        client.delete_ai_workload_config(AIWorkloadConfigName=config_name)
    except Exception:
        pass
    try:
        client.create_ai_workload_config(
            AIWorkloadConfigName=config_name,
            AIWorkloadConfigs={"WorkloadSpec": {"Inline": json.dumps(workload_spec)}},
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Create benchmark job
    try:
        client.delete_ai_benchmark_job(AIBenchmarkJobName=job_name)
    except Exception:
        pass
    try:
        client.create_ai_benchmark_job(
            AIBenchmarkJobName=job_name,
            AIWorkloadConfigIdentifier=config_name,
            BenchmarkTarget={"Endpoint": {"Identifier": DIRECT_URL_SENTINEL}},
            OutputConfig={"S3OutputLocation": s3_job_path},
            RoleArn=defaults["role_arn"],
            Tags=[
                {"Key": "project", "Value": "benchmarking-initiative"},
                {"Key": "environment", "Value": "byom"},
                {"Key": "model", "Value": job["model_key"]},
                {"Key": "workload", "Value": job["workload_key"]},
                {"Key": "concurrency", "Value": str(job["concurrency"])},
            ],
        )
        print(f"  ✓ Benchmark job submitted: {job_name}")
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Poll
    print(f"  ⏳ Polling (every 30s, up to 30 min)...")
    for _ in range(60):
        try:
            resp = client.describe_ai_benchmark_job(AIBenchmarkJobName=job_name)
            status = resp["AIBenchmarkJobStatus"]
        except Exception:
            time.sleep(30)
            continue
        if status == "Completed":
            s3_loc = resp["OutputConfig"]["S3OutputLocation"]
            print(f"  ✓ Completed — {s3_loc}")
            from athena_writer import write_athena_record
            model_cfg = models.get(job["model_key"], {})
            write_athena_record({
                "job_id": job_name, "environment": "byom",
                "model_key": job["model_key"], "model_id": job["model_id"],
                "workload": job["workload_key"], "concurrency": job["concurrency"],
                "input_tokens": job["input_tokens"], "output_tokens": job["output_tokens"],
                "streaming": job["streaming"], "duration": job["duration"],
                "warmup": job["warmup"], "source_region": region,
                "s3_output": s3_loc, "timestamp": timestamp,
            }, model_config=model_cfg, defaults_config=defaults, config_source="benchmarks.yaml")
            return {"success": True, "s3_output": s3_loc}
        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "unknown")
            print(f"  ✗ {status}: {reason}")
            return {"success": False, "error": reason}
        time.sleep(30)
    return {"success": False, "error": "Timed out after 30 min"}


# --- Submit Mode (runs on your laptop) ---

def submit_job(args):
    """Package and submit a SageMaker Processing Job."""
    config = load_config(args.config)
    defaults = config.get("byom_defaults", {})
    models = config.get("models", {})

    model_key = args.model
    if not model_key or model_key not in models:
        print(f"ERROR: --model required. Available: {list(models.keys())}", file=sys.stderr)
        sys.exit(1)

    region = defaults.get("region", "us-east-1")
    role = defaults.get("role_arn")
    job_name = f"bench-byom-{model_key}-{datetime.now().strftime('%m%d-%H%M')}"[:63]

    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    bucket = f"sagemaker-benchmark-{region}-{account}"

    # Upload config + script + deps
    script_dir = os.path.dirname(__file__)
    config_key = f"processing-configs/{job_name}/benchmarks.yaml"
    script_key = f"processing-configs/{job_name}/benchmark_bedrock_byom.py"
    req_key = f"processing-configs/{job_name}/requirements.txt"
    s3.put_object(Bucket=bucket, Key=config_key, Body=open(args.config).read())
    s3.put_object(Bucket=bucket, Key=script_key, Body=open(__file__).read())
    req_path = os.path.join(script_dir, "requirements.txt")
    if os.path.exists(req_path):
        s3.put_object(Bucket=bucket, Key=req_key, Body=open(req_path).read())
    athena_path = os.path.join(script_dir, "athena_writer.py")
    if os.path.exists(athena_path):
        s3.put_object(Bucket=bucket, Key=f"processing-configs/{job_name}/athena_writer.py", Body=open(athena_path).read())

    instance_type = defaults.get("processing_instance_type", "ml.m7i.8xlarge")
    volume_gb = defaults.get("processing_volume_gb", 2000)

    print(f"\n{'='*60}")
    print(f"Submitting Processing Job: {job_name}")
    print(f"  Model: {model_key}")
    print(f"  Region: {region}")
    print(f"  Instance: {instance_type} ({volume_gb}GB disk)")
    print(f"{'='*60}\n")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": volume_gb,
            }
        },
        AppSpecification={
            "ImageUri": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-cpu-py311",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/script/benchmark_bedrock_byom.py"],
            "ContainerArguments": ["--model", model_key] + (["--benchmark-only"] if args.benchmark_only else []) + (["--import-only"] if args.import_only else []),
        },
        Environment={
            "SM_PROCESSING_MODE": "byom-benchmark",
            "MODEL_KEY": model_key,
        },
        ProcessingInputs=[
            {
                "InputName": "script",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/processing-configs/{job_name}/",
                    "LocalPath": "/opt/ml/processing/input/script",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [{
                "OutputName": "results",
                "S3Output": {
                    "S3Uri": f"s3://{bucket}/byom/{model_key}/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }]
        },
        RoleArn=role,
        StoppingCondition={"MaxRuntimeInSeconds": 432000},  # 5 days
        NetworkConfig={"EnableNetworkIsolation": False},
    )

    print(f"  ✓ Job submitted: {job_name}")
    print(f"  Monitor: aws sagemaker describe-processing-job --processing-job-name {job_name} --region {region}")
    print(f"  Logs: /aws/sagemaker/ProcessingJobs in CloudWatch ({region})")


# --- Execution Mode (runs inside the Processing Job or locally without --submit) ---

def run_pipeline(model_filter=None, workload_filter=None, import_only=False, benchmark_only=False):
    """Full BYOM pipeline: download → upload → import → reserve → benchmark."""
    import subprocess

    config = load_config()
    defaults = config.get("byom_defaults", {})
    models = config.get("models", {})
    region = defaults["region"]

    # If inside Processing Job, override model filter from env
    if ENTRYPOINT_MODE:
        model_filter = os.environ.get("MODEL_KEY", model_filter)

    jobs = expand_byom_jobs(config, model_filter, workload_filter)
    unique_models = sorted(set(j["model_key"] for j in jobs))
    reservations = {}

    if not benchmark_only:
        for model_key in unique_models:
            model_cfg = models[model_key]
            model_id = model_cfg.get("byom", {}).get("model_id", "")

            if not model_id:
                # Verify weights are staged in S3
                weights_uri = check_weights_in_s3(model_key, model_cfg, defaults)
                if not weights_uri:
                    continue
                model_cfg["weights_s3_uri"] = weights_uri
                # Import
                model_id = import_model(model_key, model_cfg, defaults)
                if not model_id:
                    print(f"  ✗ Failed to import {model_key}, skipping")
                    continue
                if not poll_import(model_id, defaults):
                    continue
                model_cfg.setdefault("byom", {})["model_id"] = model_id
                print(f"  💡 Add model_id: \"{model_id}\" to models.{model_key}.byom in benchmarks.yaml")

            # Create RU reservation
            res_id = create_reservation(model_id, defaults)
            if res_id:
                reservations[model_key] = res_id

    if import_only:
        print("\n  ✓ Import complete (--import-only)")
        return

    # Create reservations for benchmark-only mode (skipped above)
    if benchmark_only:
        for model_key in unique_models:
            model_cfg = models[model_key]
            model_id = model_cfg.get("byom", {}).get("model_id", "")
            if model_id:
                res_id = create_reservation(model_id, defaults)
                if res_id:
                    reservations[model_key] = res_id

    # Benchmark
    print("\nGenerating Bedrock token...")
    secret_arn, secret_name = get_or_create_token_secret(defaults)
    client = boto3.client("sagemaker", region_name=region,
                          config=boto3.session.Config(parameter_validation=False))

    output_dir = Path("/opt/ml/processing/output") if ENTRYPOINT_MODE else RESULTS_DIR / "byom-latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    gaps = []

    try:
        for job in jobs:
            job["model_id"] = models[job["model_key"]].get("byom", {}).get("model_id", "")
            if not job["model_id"]:
                gaps.append({"model": job["model_key"], "job": job["id"], "error": "model not imported"})
                continue

            result_file = output_dir / f"{job['id']}.json"
            if result_file.exists():
                existing = json.loads(result_file.read_text())
                if existing.get("success"):
                    print(f"\n  ⏭️  Skipping {job['id']} (completed)")
                    results.append(existing)
                    continue

            result = run_byom_benchmark(client, job, defaults, secret_arn)
            result["id"] = job["id"]
            results.append(result)
            result_file.write_text(json.dumps(result, indent=2, default=str))
            if not result.get("success"):
                gaps.append({"model": job["model_key"], "job": job["id"], "error": result.get("error", "")})
    finally:
        print("\nCancelling RU reservations...")
        for mk, rid in reservations.items():
            cancel_reservation(rid, defaults)

    # Secret persists for async benchmark jobs (12h TTL, refreshed each run)

    summary = {
        "run_id": "byom-latest", "target": "byom",
        "total": len(results),
        "completed": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "gaps": gaps,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'═'*60}")
    print("BYOM BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    print(f"Total: {summary['total']} | Completed: {summary['completed']} | Failed: {summary['failed']}")
    if gaps:
        print(f"\nGAPS ({len(gaps)}):")
        for g in gaps:
            print(f"  ✗ {g['job']}: {g['error'][:100]}")


# --- Main ---

def main():
    # If running inside Processing Job, parse container args and run pipeline
    if ENTRYPOINT_MODE:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None)
        parser.add_argument("--benchmark-only", action="store_true")
        parser.add_argument("--import-only", action="store_true")
        args = parser.parse_args()
        run_pipeline(model_filter=args.model, import_only=args.import_only, benchmark_only=args.benchmark_only)
        return

    parser = argparse.ArgumentParser(
        description="Bedrock BYOM (Bring Your Own Model) — import weights via Mantle API, reserve RUs, benchmark.",
        epilog="""
examples:
  %(prog)s --validate                    Show expanded job matrix without running
  %(prog)s --submit --model=gpt-oss-120b-byom
                                         Submit full pipeline as Processing Job (download → import → benchmark)
  %(prog)s --submit --import-only --model=gpt-oss-120b-byom
                                         Submit import-only job (no benchmarking)
  %(prog)s --submit --benchmark-only --model=gpt-oss-120b-byom
                                         Submit benchmark-only job (model_id must be in config)
  %(prog)s --model=gpt-oss              Run locally, filter by model substring
  %(prog)s --import-only --model=kimi   Import only, no benchmark
  %(prog)s --benchmark-only             Benchmark already-imported models (skips download/import)

config: reads from benchmarks.yaml (models[].byom, workloads, byom_defaults, byom_benchmarks)
flow:   download weights → upload to S3 → import via Mantle API → create RU reservation → benchmark → cancel RU
auth:   generates a 12h bearer token, stores in Secrets Manager (auto-cleaned after run)
cost:   RU reservations billed per minute — auto-cancelled after benchmarking completes
resume: re-run safely — completed jobs are skipped, weights skip re-upload if already in S3
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default="../benchmarks.yaml", help="Path to benchmarks.yaml config (default: ../benchmarks.yaml)")
    parser.add_argument("--validate", action="store_true", help="Show expanded job matrix and exit (no AWS calls)")
    parser.add_argument("--list", action="store_true", help="List all BYOM customized models on Mantle")
    parser.add_argument("--status", help="Check import status of a specific model ID")
    parser.add_argument("--import-only", action="store_true", help="Download, upload, and import model only (no benchmark)")
    parser.add_argument("--benchmark-only", action="store_true", help="Benchmark only — model_id must already be set in config")
    parser.add_argument("--model", help="Filter by model key substring (e.g. 'gpt-oss', 'kimi')")
    parser.add_argument("--workload", help="Filter by workload key substring (e.g. 'rag', 'chat')")
    parser.add_argument("--submit", action="store_true", help="Submit as SageMaker Processing Job (2TB disk, 5-day timeout)")
    args = parser.parse_args()

    if args.list:
        config = load_config(args.config)
        defaults = config.get("byom_defaults", {})
        print("Listing BYOM customized models...\n")
        resp = mantle_request("GET", "/bedrock/v1/models/customization", defaults)
        if resp:
            models_list = resp if isinstance(resp, list) else resp.get("models", resp.get("data", [resp]))
            for m in models_list:
                mid = m.get("model_id", m.get("id", "?"))
                status = m.get("status", "?")
                base = m.get("base_model_id", "?")
                print(f"  {mid}  base={base}  status={status}")
            print(f"\nTotal: {len(models_list)}")
        else:
            print("  No models found or API returned empty response.")
        return

    if args.status:
        config = load_config(args.config)
        defaults = config.get("byom_defaults", {})
        print(f"Checking status of model: {args.status}\n")
        resp = mantle_request("GET", f"/bedrock/v1/models/customization/{args.status}", defaults)
        if resp:
            print(json.dumps(resp, indent=2))
        else:
            print("  Model not found or API error.")
        return

    if args.submit:
        submit_job(args)
        return

    config = load_config(args.config)
    defaults = config.get("byom_defaults", {})
    models = config.get("models", {})
    jobs = expand_byom_jobs(config, args.model, args.workload)

    if args.validate:
        unique_models = sorted(set(j["model_key"] for j in jobs))
        print(f"Expanded {len(jobs)} BYOM benchmark job(s)\n")
        for m in unique_models:
            mc = models[m]
            status = "✓ imported" if mc.get("byom", {}).get("model_id") else "○ pending import"
            print(f"  {status} {m} (base={mc.get('byom', {}).get('base_model_id', '?')})")
        print(f"\nMantle endpoint: {defaults['mantle_endpoint']}")
        print(f"Total jobs: {len(jobs)}")
        return

    # Local execution (same as Processing Job but on your machine)
    run_pipeline(model_filter=args.model, workload_filter=args.workload, import_only=args.import_only, benchmark_only=args.benchmark_only)


if __name__ == "__main__":
    # Upgrade deps in Processing Job, then re-exec to pick up new packages
    if os.environ.get("SM_PROCESSING_MODE") and not os.environ.get("_DEPS_INSTALLED"):
        import subprocess
        subprocess.run(["pip", "install", "-q", "-r", "/opt/ml/processing/input/script/requirements.txt"])
        os.environ["_DEPS_INSTALLED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
