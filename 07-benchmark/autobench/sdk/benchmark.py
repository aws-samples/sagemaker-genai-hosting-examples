#!/usr/bin/env python3
"""
SDK-based benchmarking execution path.

Deploys models and runs benchmark sweeps directly via boto3, bypassing
ml-container-creator for faster iteration. Reads the same benchmarks.yaml
as the Node.js orchestrator.

Usage:
    python sdk/benchmark.py --validate
    python sdk/benchmark.py --model=gemma --workload=multi_turn
    python sdk/benchmark.py --deploy-only --model=gemma
    python sdk/benchmark.py --benchmark-only --model=gemma
    python sdk/benchmark.py --cleanup --model=gemma
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

REGION = os.getenv("AWS_REGION", "us-west-2")
RESULTS_DIR = Path("results")

# --- Config Loading ---

def load_config(path="benchmarks.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def expand_jobs(config, model_filter=None, workload_filter=None):
    """Expand benchmarks into individual jobs: (model × workload × concurrency)."""
    jobs = []
    for bench in config["sagemaker_benchmarks"]:
        model_key = bench["model"]
        model = config["models"].get(model_key)
        if not model:
            print(f"ERROR: Unknown model '{model_key}'", file=sys.stderr)
            sys.exit(1)
        if model_filter and model_filter not in model_key:
            continue
        for wl_name in bench["workloads"]:
            if workload_filter and workload_filter not in wl_name:
                continue
            workload = config["workloads"].get(wl_name)
            if not workload:
                print(f"ERROR: Unknown workload '{wl_name}'", file=sys.stderr)
                sys.exit(1)
            for concurrency in workload["concurrency"]:
                jobs.append({
                    "id": f"{model_key}--{wl_name}--c{concurrency}",
                    "model_key": model_key,
                    "workload_key": wl_name,
                    "model_name": model["model_name"],
                    "benchmark_tokenizer": model.get("benchmark_tokenizer"),
                    "instance_type": model["instance_type"],
                    "num_gpus": model.get("num_gpus", 1),
                    "base_image": model.get("base_image"),
                    "variant": model.get("variant", "baseline"),
                    "optimize_model": model.get("optimize_model", False),
                    "concurrency": concurrency,
                    "input_tokens": workload["input_tokens"],
                    "output_tokens": workload["output_tokens"],
                    "streaming": workload.get("streaming", True),
                    "duration": workload.get("duration", 300),
                    "warmup": workload.get("warmup", 30),
                    "dataset": workload.get("dataset"),
                })
    return jobs


# --- ECR Image Management ---

def get_ecr_image(base_image):
    """Ensure the vLLM image is in ECR. Uses CodeBuild to mirror if needed (no local Docker)."""
    sts = boto3.client("sts", region_name="us-east-2")
    account = sts.get_caller_identity()["Account"]
    ecr_client = boto3.client("ecr", region_name=REGION)
    repo_name = "benchmarking-initiative"
    tag = base_image.split(":")[-1] if ":" in base_image else "latest"
    ecr_uri = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{repo_name}:{tag}"

    # Check if image already exists in ECR
    try:
        ecr_client.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])
        print(f"  ✓ ECR image exists: {ecr_uri}")
        return ecr_uri
    except ecr_client.exceptions.ImageNotFoundException:
        pass
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=repo_name)
        print(f"  ✓ Created ECR repo: {repo_name}")

    # Use CodeBuild to pull from Docker Hub and push to ECR (no local Docker needed)
    print(f"  ⏳ Mirroring {base_image} → ECR via CodeBuild...")
    cb_client = boto3.client("codebuild", region_name=REGION)
    project_name = "bench-image-mirror"

    # Ensure CodeBuild project exists
    try:
        cb_client.batch_get_projects(names=[project_name])["projects"][0]
    except (IndexError, KeyError):
        # Create a minimal CodeBuild project for image mirroring
        iam = boto3.client("iam")
        role_arn = _ensure_codebuild_role(iam, account)
        cb_client.create_project(
            name=project_name,
            source={"type": "NO_SOURCE", "buildspec": "version: 0.2\nphases:\n  build:\n    commands:\n      - echo placeholder"},
            artifacts={"type": "NO_ARTIFACTS"},
            environment={
                "type": "LINUX_CONTAINER",
                "computeType": "BUILD_GENERAL1_MEDIUM",
                "image": "aws/codebuild/standard:7.0",
                "privilegedMode": True,
            },
            serviceRole=role_arn,
        )
        print(f"  ✓ Created CodeBuild project: {project_name}")

    # Start build with inline buildspec that mirrors the image
    buildspec = f"""version: 0.2
phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {account}.dkr.ecr.{REGION}.amazonaws.com
  build:
    commands:
      - docker pull --platform linux/amd64 {base_image}
      - docker tag {base_image} {ecr_uri}
      - docker push {ecr_uri}
"""
    build = cb_client.start_build(
        projectName=project_name,
        buildspecOverride=buildspec,
    )
    build_id = build["build"]["id"]
    print(f"  ⏳ CodeBuild started: {build_id}")

    # Poll for completion
    for _ in range(60):
        resp = cb_client.batch_get_builds(ids=[build_id])
        status = resp["builds"][0]["buildStatus"]
        if status == "SUCCEEDED":
            print(f"  ✓ Image mirrored to ECR: {ecr_uri}")
            return ecr_uri
        if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
            logs = resp["builds"][0].get("logs", {}).get("deepLink", "")
            print(f"  ✗ CodeBuild {status}. Logs: {logs}")
            sys.exit(1)
        time.sleep(15)
    print("  ✗ CodeBuild timed out")
    sys.exit(1)


def _ensure_codebuild_role(iam, account):
    """Create or return the CodeBuild service role for image mirroring."""
    role_name = "bench-codebuild-mirror-role"
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    })
    role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust)
    iam.attach_role_policy(RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser")
    iam.attach_role_policy(RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/CloudWatchLogsFullAccess")
    time.sleep(10)  # wait for role propagation
    return role["Role"]["Arn"]


# --- Deployment (boto3) ---

def get_client(region=None):
    return boto3.client("sagemaker", region_name=region or REGION,
                        config=boto3.session.Config(parameter_validation=False))


def get_role_arn():
    """Get the SageMaker execution role from environment."""
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        raise ValueError("No role ARN configured. Set 'role_arn' in sagemaker_defaults or SAGEMAKER_ROLE_ARN env var.")
    return role


def endpoint_name(model_key):
    return f"bench-{model_key}"[:63]


def ic_name(model_key):
    return f"bench-ic-{model_key}"[:63]


def deploy_model(client, model_key, model_cfg, defaults):
    """Deploy using the AWS vLLM SageMaker DLC.
    Uses SM_VLLM_* env vars which map directly to vLLM CLI args.
    Image is already in ECR — no build needed.
    """
    ep_name = endpoint_name(model_key)
    ep_config_name = f"{ep_name}-config"
    sm_model_name = f"bench-mdl-{model_key}"[:63]
    role = defaults.get("role_arn") or get_role_arn()
    instance_type = model_cfg["instance_type"]
    num_gpus = model_cfg.get("num_gpus", 1)
    region = defaults.get("region", REGION)
    image = defaults["sagemaker_image"].format(region=region)
    ami_version = defaults.get("inference_ami_version", "al2-ami-sagemaker-inference-gpu-3-1")
    training_plan_arn = defaults.get("ml_reservation_arn")
    model_id = model_cfg["model_name"]
    s3_model_uri = model_cfg.get("s3_model_uri", "")

    env_vars = {
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(num_gpus),
        "SM_VLLM_TRUST_REMOTE_CODE": "true",
    }
    # When s3_model_uri is set, use ModelDataSource (SageMaker mounts to /opt/ml/model)
    # Otherwise, use SM_VLLM_MODEL with the HuggingFace ID
    if not s3_model_uri:
        env_vars["SM_VLLM_MODEL"] = model_id
    vllm_config = defaults.get("vllm_config", {})
    if vllm_config.get("max_model_len"):
        env_vars["SM_VLLM_MAX_MODEL_LEN"] = str(vllm_config["max_model_len"])
    if vllm_config.get("gpu_memory_utilization"):
        env_vars["SM_VLLM_GPU_MEMORY_UTILIZATION"] = str(vllm_config["gpu_memory_utilization"])
    # Allow per-model overrides
    for k, v in model_cfg.get("env", {}).items():
        env_vars[k] = str(v)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    print(f"\n{'='*60}")
    print(f"[deploy] {model_key}")
    print(f"  image: {image}")
    print(f"  instance: {instance_type} ({num_gpus} GPUs)")
    print(f"  model: {model_id}")
    if s3_model_uri:
        print(f"  source: {s3_model_uri} (S3)")
    else:
        print(f"  source: HuggingFace (direct download)")
    print(f"{'='*60}")

    # 1. Create Model
    try:
        client.delete_model(ModelName=sm_model_name)
    except Exception:
        pass
    container_def = {
        "Image": image,
        "Environment": env_vars,
    }
    if s3_model_uri:
        container_def["ModelDataSource"] = {
            "S3DataSource": {
                "S3Uri": s3_model_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }
    client.create_model(
        ModelName=sm_model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=container_def,
    )
    print(f"  ✓ Created model: {sm_model_name}")

    # 2. Create endpoint config (with training plan for capacity)
    try:
        client.delete_endpoint_config(EndpointConfigName=ep_config_name)
    except Exception:
        pass
    variant = {
        "VariantName": "default",
        "ModelName": sm_model_name,
        "InstanceType": instance_type,
        "InitialInstanceCount": 1,
        "InferenceAmiVersion": ami_version,
        "ModelDataDownloadTimeoutInSeconds": 1800,
        "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
    }
    if training_plan_arn:
        variant["CapacityReservationConfig"] = {
            "MlReservationArn": training_plan_arn,
            "CapacityReservationPreference": "capacity-reservations-only",
        }
    client.create_endpoint_config(
        EndpointConfigName=ep_config_name,
        ProductionVariants=[variant],
    )
    print(f"  ✓ Created endpoint config: {ep_config_name}")

    # 3. Create or update endpoint
    try:
        client.create_endpoint(
            EndpointName=ep_name,
            EndpointConfigName=ep_config_name,
        )
        print(f"  ✓ Created endpoint: {ep_name}")
    except client.exceptions.ClientError as e:
        if "Cannot create already existing" in str(e):
            client.update_endpoint(EndpointName=ep_name, EndpointConfigName=ep_config_name)
            print(f"  → Updated endpoint: {ep_name}")
        else:
            raise

    # 4. Wait for InService
    print(f"  ⏳ Waiting for endpoint InService...")
    waiter = client.get_waiter("endpoint_in_service")
    try:
        waiter.wait(EndpointName=ep_name, WaiterConfig={"Delay": 30, "MaxAttempts": 60})
    except Exception as e:
        try:
            status = client.describe_endpoint(EndpointName=ep_name)["EndpointStatus"]
        except Exception:
            status = "Unknown"
        return {"success": False, "status": status, "error": f"Endpoint reached {status}: {e}"}

    print(f"  ✓ Endpoint InService: {ep_name}")
    return {"success": True, "endpoint": ep_name, "model_name": sm_model_name}


# --- Inference Recommender (Advanced) ---

def run_recommendation_job(client, model_key, model_cfg, ep_name, defaults):
    """Run an Advanced Inference Recommender job with staircase traffic pattern.
    Tests concurrency scaling and can optionally tune vLLM env vars.
    """
    role = get_role_arn()
    sm_model_name = f"bench-mdl-{model_key}"[:63]
    job_name = f"reco-{model_key}-{int(time.time())}"[:63]
    model_id = model_cfg["model_name"]

    print(f"\n{'='*60}")
    print(f"[recommendation] {model_key}")
    print(f"  endpoint: {ep_name}")
    print(f"  model: {model_id}")
    print(f"{'='*60}")

    try:
        client.create_inference_recommendations_job(
            JobName=job_name,
            JobType="Advanced",
            RoleArn=role,
            InputConfig={
                "ModelName": sm_model_name,
                "Endpoints": [{"EndpointName": ep_name}],
                "JobDurationInSeconds": 600,
                "TrafficPattern": {
                    "TrafficType": "STAIRS",
                    "Stairs": {
                        "DurationInSeconds": 120,
                        "NumberOfSteps": 5,
                        "UsersPerStep": 16,
                    },
                },
                "TokenizerConfig": {
                    "ModelId": model_id,
                    "AcceptEula": True,
                },
            },
            StoppingConditions={
                "MaxInvocations": 5000,
            },
        )
        print(f"  ✓ Created recommendation job: {job_name}")
    except Exception as e:
        print(f"  ✗ Failed to create recommendation job: {e}")
        return {"success": False, "error": str(e), "gap": classify_gap(str(e))}

    # Poll for completion
    print(f"  ⏳ Polling (every 30s, up to 30 min)...")
    for _ in range(60):
        resp = client.describe_inference_recommendations_job(JobName=job_name)
        status = resp["Status"]
        if status == "COMPLETED":
            results = resp.get("InferenceRecommendations", [])
            print(f"  ✓ Completed — {len(results)} recommendation(s)")
            return {"success": True, "status": "completed", "job_name": job_name, "recommendations": results}
        if status in ("FAILED", "STOPPED"):
            reason = resp.get("FailureReason", "unknown")
            print(f"  ✗ {status}: {reason}")
            return {"success": False, "error": reason, "gap": classify_gap(reason)}
        time.sleep(30)
    return {"success": False, "error": "Recommendation job timed out", "gap": {"category": "timeout"}}


# --- Benchmarking (boto3) ---

def run_benchmark_job(client, job, ep_name, defaults, models=None):
    """Create a workload config and benchmark job for a single concurrency level."""
    role = defaults.get("role_arn") or get_role_arn()
    s3_output = defaults.get("s3_output", "").format(region=REGION)
    if not s3_output:
        account = boto3.client("sts", region_name="us-east-2").get_caller_identity()["Account"]
        s3_output = f"s3://sagemaker-benchmark-{REGION}-{account}/managed-inference"

    # Hierarchical path: {env}/{model}/{workload}/{instance}/{timestamp}/
    instance_short = defaults.get("ml_reservation_arn", "").split("/")[-1][:8] or "p6-b200"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    s3_job_path = f"{s3_output}/{job['model_key']}/{job['workload_key']}/p6-b200-c{job['concurrency']}/{timestamp}/"

    ts = datetime.now().strftime("%m%d%H%M")
    job_name = f"{job['id'][:50]}-{ts}".replace("_", "-")
    config_name = f"wl-{job['workload_key']}-c{job['concurrency']}"[:63].replace("_", "-")

    print(f"\n{'─'*60}")
    print(f"[benchmark] {job['id']}")
    print(f"  concurrency={job['concurrency']} in={job['input_tokens']} out={job['output_tokens']}")
    print(f"{'─'*60}")

    # Build workload spec
    workload_spec = {
        "benchmark": {"type": "aiperf"},
        "parameters": {
            "tokenizer": job.get("benchmark_tokenizer") or job["model_name"],
            "concurrency": job["concurrency"],
            "prompt_input_tokens_mean": job["input_tokens"],
            "prompt_input_tokens_stddev": 0,
            "output_tokens_mean": job["output_tokens"],
            "output_tokens_stddev": 0,
            "streaming": job["streaming"],
            "benchmark_duration": job["duration"],
            "warmup_duration": job["warmup"],
            "extra_inputs": "ignore_eos:true temperature:0",
        },
        "tooling": {"api_standard": "openai"},
    }
    # Only set public_dataset if a real dataset is specified (not "synthetic" or empty)
    if job.get("dataset") and job["dataset"].lower() != "synthetic":
        workload_spec["parameters"]["public_dataset"] = job["dataset"]

    hf_secret = os.environ.get("HF_TOKEN_SECRET_ARN")
    if hf_secret:
        workload_spec["secrets"] = {"hf_token": hf_secret}

    # Create or update workload config
    try:
        client.delete_ai_workload_config(AIWorkloadConfigName=config_name)
    except Exception:
        pass
    try:
        client.create_ai_workload_config(
            AIWorkloadConfigName=config_name,
            AIWorkloadConfigs={"WorkloadSpec": {"Inline": json.dumps(workload_spec)}},
        )
        print(f"  ✓ Created workload config: {config_name}")
    except Exception as e:
        print(f"  ✗ Failed to create workload config: {e}")
        return {"success": False, "error": f"Failed to create workload config: {e}", "gap": classify_gap(str(e))}

    # Delete previous job with same name if exists
    try:
        client.delete_ai_benchmark_job(AIBenchmarkJobName=job_name)
    except Exception:
        pass

    # Create benchmark job
    try:
        client.create_ai_benchmark_job(
            AIBenchmarkJobName=job_name,
            AIWorkloadConfigIdentifier=config_name,
            BenchmarkTarget={
                "Endpoint": {
                    "Identifier": ep_name,
                }
            },
            OutputConfig={"S3OutputLocation": s3_job_path},
            RoleArn=role,
            Tags=[
                {"Key": "project", "Value": "benchmarking-initiative"},
                {"Key": "environment", "Value": "managed-inference"},
                {"Key": "model", "Value": job["model_key"]},
                {"Key": "workload", "Value": job["workload_key"]},
                {"Key": "concurrency", "Value": str(job["concurrency"])},
                {"Key": "instance-type", "Value": job["instance_type"]},
                {"Key": "ml-reservation-arn", "Value": defaults.get("ml_reservation_arn", "")},
            ],
        )
        print(f"  ✓ Created benchmark job: {job_name}")

        # Write manifest describing benchmark conditions
        manifest = {
            "model": job["model_name"],
            "model_key": job["model_key"],
            "workload": job["workload_key"],
            "concurrency": job["concurrency"],
            "input_tokens": job["input_tokens"],
            "output_tokens": job["output_tokens"],
            "streaming": job["streaming"],
            "duration": job["duration"],
            "instance_type": "ml.p6-b200.48xlarge",
            "num_gpus": job["num_gpus"],
            "vllm_config": defaults.get("vllm_config", {}),
            "image": defaults.get("sagemaker_image", "").format(region=REGION),
            "timestamp": timestamp,
            "environment": "managed-inference",
        }
        s3_parts = s3_job_path.replace("s3://", "").split("/", 1)
        s3_client = boto3.client("s3", region_name=REGION)
        try:
            s3_client.put_object(
                Bucket=s3_parts[0],
                Key=f"{s3_parts[1]}manifest.json",
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json",
            )
        except Exception:
            pass  # non-fatal
    except Exception as e:
        return {"success": False, "error": f"Failed to create benchmark job: {e}", "gap": classify_gap(str(e))}

    # Poll for completion
    print(f"  ⏳ Polling (every 30s, up to 30 min)...")
    for i in range(60):
        resp = client.describe_ai_benchmark_job(AIBenchmarkJobName=job_name)
        status = resp["AIBenchmarkJobStatus"]
        if status == "Completed":
            s3_loc = resp["OutputConfig"]["S3OutputLocation"]
            print(f"  ✓ Completed — results at {s3_loc}")
            from athena_writer import write_athena_record
            model_cfg = (models or {}).get(job["model_key"], {})
            write_athena_record({
                "job_id": job_name,
                "environment": "managed-inference",
                "model_key": job["model_key"],
                "model_name": job["model_name"],
                "workload": job["workload_key"],
                "concurrency": job["concurrency"],
                "input_tokens": job["input_tokens"],
                "output_tokens": job["output_tokens"],
                "streaming": job["streaming"],
                "duration": job["duration"],
                "warmup": job["warmup"],
                "instance_type": job["instance_type"],
                "num_gpus": job["num_gpus"],
                "dataset": job.get("dataset"),
                "vllm_config": defaults.get("vllm_config", {}),
                "source_region": REGION,
                "s3_output": s3_loc,
                "timestamp": timestamp,
            }, model_config=model_cfg, defaults_config=defaults, config_source="benchmarks.yaml")
            return {"success": True, "status": "completed", "s3_output": s3_loc}
        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "unknown")
            print(f"  ✗ {status}: {reason}")
            return {"success": False, "error": reason, "gap": classify_gap(reason)}
        time.sleep(30)
    return {"success": False, "error": "Benchmark timed out after 30 min", "gap": {"category": "timeout"}}


# --- Gap Classification ---

def classify_gap(error_str):
    err = str(error_str).lower()
    if "unrecognizedclient" in err:
        return {"category": "api_availability", "action": "Confirm API availability in us-west-2"}
    if "capacity" in err or "resourcelimit" in err:
        return {"category": "capacity", "action": "Request capacity or try different instance"}
    if "tokenizer" in err or "hf_token" in err or "access" in err:
        return {"category": "model_access", "action": "Set HF_TOKEN or HF_TOKEN_SECRET_ARN"}
    if "timeout" in err:
        return {"category": "timeout", "action": "Increase timeout or reduce concurrency"}
    if "throttl" in err:
        return {"category": "throttling", "action": "Add backoff or reduce request rate"}
    return {"category": "unknown", "detail": error_str[:300]}


# --- Cleanup ---

def cleanup_model(client, model_key):
    """Delete endpoint, endpoint config, and model."""
    ep_name = endpoint_name(model_key)
    ep_config_name = f"{ep_name}-config"
    sm_model_name = f"bench-mdl-{model_key}"[:63]

    print(f"\n[cleanup] {model_key}")
    try:
        client.delete_endpoint(EndpointName=ep_name)
        print(f"  ✓ Deleted endpoint: {ep_name}")
    except Exception:
        pass
    try:
        client.delete_endpoint_config(EndpointConfigName=ep_config_name)
        print(f"  ✓ Deleted endpoint config: {ep_config_name}")
    except Exception:
        pass
    try:
        client.delete_model(ModelName=sm_model_name)
        print(f"  ✓ Deleted model: {sm_model_name}")
    except Exception:
        pass


def wait_for_endpoint_deleted(client, ep_name, timeout=300):
    """Wait for endpoint to be fully deleted before releasing FTP capacity."""
    print(f"  ⏳ Waiting for endpoint deletion to complete...")
    for i in range(timeout // 15):
        try:
            resp = client.describe_endpoint(EndpointName=ep_name)
            status = resp["EndpointStatus"]
            if status == "Deleting":
                time.sleep(15)
                continue
            # Still exists in non-Deleting state — unexpected
            time.sleep(15)
        except client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e) or "ValidationException" in str(e):
                print(f"  ✓ Endpoint deleted: {ep_name}")
                return
            time.sleep(15)
        except Exception:
            time.sleep(15)
    print(f"  ⚠️  Endpoint deletion not confirmed after {timeout}s — proceeding")


def wait_for_ftp_available(client, defaults):
    """Poll the training plan until the reserved capacity instance is available."""
    training_plan_arn = defaults.get("ml_reservation_arn", "")
    if not training_plan_arn:
        time.sleep(60)
        return
    plan_name = training_plan_arn.split("/")[-1]
    region = defaults.get("region", "us-east-2")
    sm = boto3.client("sagemaker", region_name=region)

    print(f"  ⏳ Waiting for FTP instance to become available...")
    for attempt in range(120):  # up to 60 min
        try:
            resp = sm.describe_training_plan(TrainingPlanName=plan_name)
            available = resp.get("AvailableInstanceCount", 0)
            total = resp.get("TotalInstanceCount", 1)
            if available >= 1:
                print(f"  ✓ FTP available ({available}/{total} instances free)")
                return
            print(f"    [{(attempt+1)*30}s] FTP in use ({available}/{total} available)")
        except Exception as e:
            print(f"    [{(attempt+1)*30}s] Could not check FTP: {e}")
        time.sleep(30)
    print(f"  ⚠️  FTP not available after 60 min — proceeding anyway")


# --- Remote Execution (Processing Job) ---

def submit_processing_job(args, script_name, job_prefix, config_path):
    """Submit this script as a SageMaker Processing Job for unattended execution."""
    config = load_config(config_path)
    defaults = config.get("sagemaker_defaults", {})
    region = defaults.get("region", "us-east-2")
    role = defaults.get("role_arn")
    if not role:
        raise ValueError("'role_arn' must be set in sagemaker_defaults")
    s3_output = defaults.get("s3_output", "").format(region=region)
    bucket = s3_output.replace("s3://", "").split("/")[0] if s3_output else f"sagemaker-benchmark-{region}-{boto3.client('sts', region_name=region).get_caller_identity()['Account']}"
    job_name = f"{job_prefix}-{datetime.now().strftime('%m%d-%H%M')}"[:63]

    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name="us-east-2")

    # Upload script + config + requirements + athena_writer (all under same prefix)
    prefix = f"processing-configs/{job_name}"
    script_dir = os.path.dirname(script_name)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/script.py", Body=open(script_name).read())
    s3.put_object(Bucket=bucket, Key=f"{prefix}/benchmarks.yaml", Body=open(config_path).read())
    req_path = os.path.join(script_dir, "requirements.txt")
    if os.path.exists(req_path):
        s3.put_object(Bucket=bucket, Key=f"{prefix}/requirements.txt", Body=open(req_path).read())
    athena_path = os.path.join(script_dir, "athena_writer.py")
    if os.path.exists(athena_path):
        s3.put_object(Bucket=bucket, Key=f"{prefix}/athena_writer.py", Body=open(athena_path).read())

    # Build container args (pass through filters)
    container_args = []
    if args.model:
        container_args += ["--model", args.model]
    if args.workload:
        container_args += ["--workload", args.workload]
    if hasattr(args, 'endpoint') and args.endpoint:
        container_args += ["--endpoint", args.endpoint]

    print(f"\n{'='*60}")
    print(f"Submitting Processing Job: {job_name}")
    print(f"  Region: {region}")
    print(f"  Instance: ml.m7i.4xlarge (Intel)")
    print(f"  Max runtime: 5 days")
    if args.model:
        print(f"  Filter: --model={args.model}")
    print(f"{'='*60}\n")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m7i.4xlarge",
                "VolumeSizeInGB": 50,
            }
        },
        AppSpecification={
            "ImageUri": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-cpu-py311",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/script/script.py"],
            "ContainerArguments": ["/opt/ml/processing/input/script/benchmarks.yaml"] + container_args,
        },
        ProcessingInputs=[
            {
                "InputName": "script",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{prefix}/",
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
                    "S3Uri": f"s3://{bucket}/processing-results/{job_name}/",
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


# --- Main ---

def main():
    # Upgrade dependencies if running inside a Processing Job (DLC has old SDK)
    if os.environ.get("PROCESSING_JOB_NAME") or os.path.exists("/opt/ml/processing"):
        if not os.environ.get("_DEPS_INSTALLED"):
            import subprocess
            subprocess.run(["pip", "install", "-q", "-r", "/opt/ml/processing/input/script/requirements.txt"])
            # Re-exec with updated packages
            os.environ["_DEPS_INSTALLED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(
        description="SageMaker Managed Inference benchmarking using AI Benchmarking (NVIDIA AIPerf).",
        epilog="""
examples:
  %(prog)s --validate                    Show expanded job matrix without running
  %(prog)s                               Run all benchmarks (deploy + benchmark)
  %(prog)s --model=gemma                 Run only models matching 'gemma'
  %(prog)s --workload=rag               Run only workloads matching 'rag'
  %(prog)s --deploy-only                 Deploy endpoints, skip benchmarking
  %(prog)s --benchmark-only              Benchmark existing endpoints (skip deploy)
  %(prog)s --cleanup                     Delete all deployed endpoints
  %(prog)s --submit                      Submit as unattended Processing Job (5-day timeout)
  %(prog)s --submit --model=gemma        Submit filtered job remotely

config: reads from benchmarks.yaml (models, workloads, sagemaker_defaults, sagemaker_benchmarks)
output: results written to S3 and local results/ directory
resume: re-run safely — completed jobs are skipped automatically
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default="../benchmarks.yaml", help="Path to benchmarks.yaml config (default: ../benchmarks.yaml)")
    parser.add_argument("--validate", action="store_true", help="Show expanded job matrix and exit (no AWS calls)")
    parser.add_argument("--model", help="Filter by model key substring (e.g. 'gemma', 'qwen3')")
    parser.add_argument("--workload", help="Filter by workload key substring (e.g. 'rag', 'chat')")
    parser.add_argument("--deploy-only", action="store_true", help="Deploy model endpoints only, skip benchmarking")
    parser.add_argument("--benchmark-only", action="store_true", help="Run benchmarks against already-deployed endpoints")
    parser.add_argument("--endpoint", help="Run benchmarks against this specific endpoint name (skips deploy/cleanup, requires --model)")
    parser.add_argument("--cleanup", action="store_true", help="Delete all deployed endpoints and models")
    parser.add_argument("--submit", action="store_true", help="Submit as a remote SageMaker Processing Job (no credential expiry, 5-day timeout)")
    args = parser.parse_args()

    if args.submit:
        submit_processing_job(args, __file__, "bench-smai", args.config)
        return

    config = load_config(args.config)
    jobs = expand_jobs(config, args.model, args.workload)

    if args.validate:
        models = sorted(set(j["model_key"] for j in jobs))
        workloads = sorted(set(j["workload_key"] for j in jobs))
        print(f"Expanded {len(jobs)} job(s)\n")
        print(f"Models ({len(models)}):")
        for m in models:
            mc = config["models"][m]
            print(f"  ✓ {m} ({mc['instance_type']}, {mc.get('num_gpus',1)} GPUs)")
        print(f"\nWorkloads ({len(workloads)}):")
        for w in workloads:
            wl = config["workloads"][w]
            print(f"  ✓ {w}: in={wl['input_tokens']} out={wl['output_tokens']} c=[{','.join(map(str, wl['concurrency']))}]")
        print(f"\nTotal jobs: {len(jobs)}")
        return

    client = get_client(config.get("sagemaker_defaults", {}).get("region"))

    # Set global region from config
    global REGION
    REGION = config.get("sagemaker_defaults", {}).get("region", REGION)

    # Ensure S3 bucket for benchmark output exists
    defaults_cfg = config.get("sagemaker_defaults", {})
    s3_output = defaults_cfg.get("s3_output", "").format(region=REGION)
    if s3_output:
        s3_bucket = s3_output.replace("s3://", "").split("/")[0]
    else:
        account = boto3.client("sts", region_name="us-east-2").get_caller_identity()["Account"]
        s3_bucket = f"sagemaker-benchmark-{REGION}-{account}"
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.head_bucket(Bucket=s3_bucket)
    except Exception:
        try:
            s3.create_bucket(Bucket=s3_bucket, CreateBucketConfiguration={"LocationConstraint": REGION})
            print(f"  ✓ Created S3 bucket: {s3_bucket}")
        except Exception:
            pass

    if args.cleanup:
        models = sorted(set(j["model_key"] for j in jobs))
        for m in models:
            cleanup_model(client, m)
        return

    # Group jobs by model
    by_model = {}
    for j in jobs:
        by_model.setdefault(j["model_key"], []).append(j)

    run_id = "sagemaker-latest"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    gaps = []

    for model_key, model_jobs in by_model.items():
        model_cfg = config["models"][model_key]

        # Use explicit endpoint if provided (skips deploy/cleanup)
        if args.endpoint:
            ep_name_val = args.endpoint
        elif not args.benchmark_only:
            deploy_result = deploy_model(client, model_key, model_cfg, config.get("sagemaker_defaults", {}))
            if not deploy_result["success"]:
                gap = classify_gap(deploy_result.get("error", ""))
                gaps.append({"model": model_key, "step": "deploy", **gap})
                print(f"  ✗ Deploy failed ({deploy_result.get('status', 'unknown')}) — skipping all workloads for {model_key}")
                for j in model_jobs:
                    results.append({"id": j["id"], "status": "skipped", "reason": deploy_result["error"]})
                # Always cleanup failed endpoint and wait for FTP before next model
                cleanup_model(client, model_key)
                wait_for_endpoint_deleted(client, endpoint_name(model_key))
                wait_for_ftp_available(client, config.get("sagemaker_defaults", {}))
                continue
            ep_name_val = endpoint_name(model_key)
        else:
            ep_name_val = endpoint_name(model_key)

        if args.deploy_only:
            continue

        # Run benchmarks
        for job in model_jobs:
            # Resume logic: skip jobs that already completed
            result_file = run_dir / f"{job['id']}.json"
            if result_file.exists():
                existing = json.loads(result_file.read_text())
                if existing.get("success"):
                    print(f"\n  ⏭️  Skipping {job['id']} (already completed)")
                    results.append(existing)
                    continue

            result = run_benchmark_job(client, job, ep_name_val, config.get("sagemaker_defaults", {}), models=config.get("models", {}))
            result["id"] = job["id"]
            result["model_key"] = job["model_key"]
            result["workload_key"] = job["workload_key"]
            result["concurrency"] = job["concurrency"]
            results.append(result)
            # Write per-job result
            (run_dir / f"{job['id']}.json").write_text(json.dumps(result, indent=2, default=str))
            if result.get("gap"):
                gaps.append({"model": model_key, "job": job["id"], **result["gap"]})

        # Cleanup after all workloads for this model (required for single-instance FTP)
        if config.get("sagemaker_defaults", {}).get("cleanup", False) and not args.benchmark_only and not args.endpoint:
            cleanup_model(client, model_key)
            wait_for_endpoint_deleted(client, endpoint_name(model_key))
            wait_for_ftp_available(client, config.get("sagemaker_defaults", {}))

    # Write summary
    summary = {"run_id": run_id, "total": len(results),
               "completed": sum(1 for r in results if r.get("success")),
               "failed": sum(1 for r in results if not r.get("success") and r.get("status") != "skipped"),
               "skipped": sum(1 for r in results if r.get("status") == "skipped"),
               "gaps": gaps, "results": results}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if gaps:
        (run_dir / "gaps.json").write_text(json.dumps(gaps, indent=2, default=str))

    # Print summary
    print(f"\n{'═'*60}")
    print("BENCHMARK RUN SUMMARY")
    print(f"{'═'*60}")
    print(f"Run ID: {run_id}")
    print(f"Total: {summary['total']} | Completed: {summary['completed']} | Failed: {summary['failed']} | Skipped: {summary['skipped']}")
    if gaps:
        print(f"\nGAPS ({len(gaps)}):")
        for g in gaps:
            print(f"  [{g.get('category','?')}] {g.get('model','')}: {g.get('action', g.get('detail',''))}")
    print(f"\nResults: {run_dir}/")


if __name__ == "__main__":
    main()
