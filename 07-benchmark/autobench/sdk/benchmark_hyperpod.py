#!/usr/bin/env python3
"""
HyperPod EKS benchmarking via AI Benchmarking direct-URL feature.

Deploys vLLM on a HyperPod cluster's p6-b200 node, exposes via ELB,
then submits benchmark jobs targeting the ELB URL.

Usage:
    python sdk/benchmark_hyperpod.py --check
    python sdk/benchmark_hyperpod.py --validate
    python sdk/benchmark_hyperpod.py --deploy-only
    python sdk/benchmark_hyperpod.py
    python sdk/benchmark_hyperpod.py --model=gemma
    python sdk/benchmark_hyperpod.py --cleanup

Prerequisites:
    - kubectl configured for the HyperPod EKS cluster
    - p6-b200 node available in the cluster (via FTP node group)
    - Helm (optional, or raw manifests)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import yaml

RESULTS_DIR = Path("results")
DIRECT_URL_SENTINEL = "bench-direct-7f2a9c4e1b8d053f6a9e2c7d4b1f8a3e5d0c6b9a4e8f2a1d"

VLLM_K8S_MANIFEST = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-{name}
  namespace: {namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-{name}
  template:
    metadata:
      labels:
        app: vllm-{name}
    spec:
      # Pin to the FTP node via instance type
      serviceAccountName: vllm-s3-reader
      nodeSelector:
        node.kubernetes.io/instance-type: {instance_type}
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: vllm
          image: {image}
          args:
            - "--model"
            - "{model_id}"
            - "--tensor-parallel-size"
            - "{num_gpus}"
            - "--gpu-memory-utilization"
            - "{gpu_memory_utilization}"
            - "--max-model-len"
            - "{max_model_len}"
            - "--trust-remote-code"
            - "--port"
            - "8000"
            - "--host"
            - "0.0.0.0"
            - "--served-model-name"
            - "{served_model_name}"
{extra_args}
          env:
            - name: HF_TOKEN
              value: "{hf_token}"
{extra_env}
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              nvidia.com/gpu: "{num_gpus}"
            requests:
              nvidia.com/gpu: "{num_gpus}"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 300
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 1800
            periodSeconds: 30
            failureThreshold: 5
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-{name}
  namespace: {namespace}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-scheme: internal
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: ip
spec:
  type: LoadBalancer
  selector:
    app: vllm-{name}
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
"""


# --- Config ---

def load_config(path="benchmarks.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def expand_hyperpod_jobs(config, model_filter=None, workload_filter=None):
    defaults = config.get("hyperpod_defaults", {})
    models = config.get("models", {})
    workloads = config.get("workloads", {})
    benchmarks = config.get("hyperpod_benchmarks", config.get("sagemaker_benchmarks", []))
    jobs = []

    for bench in benchmarks:
        model_key = bench["model"]
        model = models.get(model_key)
        if not model:
            print(f"ERROR: Unknown model '{model_key}'", file=sys.stderr)
            sys.exit(1)
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
                    "id": f"hp-{model_key}--{wl_name}--c{concurrency}",
                    "model_key": model_key,
                    "model_name": model["model_name"],
                    "num_gpus": model.get("num_gpus", 8),
                    "workload_key": wl_name,
                    "concurrency": concurrency,
                    "input_tokens": workload["input_tokens"],
                    "output_tokens": workload["output_tokens"],
                    "streaming": workload.get("streaming", True),
                    "duration": workload.get("duration", 300),
                    "warmup": workload.get("warmup", 30),
                    "dataset": workload.get("dataset"),
                })
    return jobs


# --- Cluster Configuration ---

def configure_kubectl(defaults):
    """Auto-configure kubectl for the HyperPod cluster's underlying EKS cluster."""
    region = defaults.get("region", "us-east-2")
    eks_cluster = defaults.get("eks_cluster_name")

    if not eks_cluster:
        # Resolve EKS cluster from HyperPod cluster name
        hyperpod_name = defaults.get("hyperpod_cluster_name")
        if not hyperpod_name:
            print("  ⚠️  No cluster configured — assuming kubectl is already set up")
            return
        print(f"  Resolving EKS cluster from HyperPod: {hyperpod_name}")
        sm = boto3.client("sagemaker", region_name=region)
        resp = sm.describe_cluster(ClusterName=hyperpod_name)
        # The EKS cluster ARN is in the orchestrator config
        eks_arn = resp.get("Orchestrator", {}).get("Eks", {}).get("ClusterArn", "")
        if not eks_arn:
            print(f"  ✗ Could not resolve EKS cluster from HyperPod '{hyperpod_name}'")
            sys.exit(1)
        eks_cluster = eks_arn.split("/")[-1]
        print(f"  ✓ Resolved EKS cluster: {eks_cluster}")

    # Update kubeconfig
    print(f"  Configuring kubectl for {eks_cluster} in {region}...")
    r = subprocess.run(
        ["aws", "eks", "update-kubeconfig", "--name", eks_cluster, "--region", region],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  ✗ Failed to configure kubectl: {r.stderr}")
        sys.exit(1)
    print(f"  ✓ kubectl configured for {eks_cluster}")


# --- Dependency Checks ---

def check_dependencies(defaults):
    errors = []

    # kubectl
    try:
        r = subprocess.run(["kubectl", "cluster-info"], capture_output=True, timeout=10)
        if r.returncode == 0:
            print(f"  ✓ kubectl connected to cluster")
        else:
            errors.append("kubectl cannot reach cluster. Configure kubeconfig for HyperPod EKS.")
    except FileNotFoundError:
        errors.append("kubectl not installed")

    # Check FTP node exists
    try:
        # Get instance type from first model in config (strip ml. prefix for K8s label)
        instance_type = "p6-b200.48xlarge"  # default
        r = subprocess.run(
            ["kubectl", "get", "nodes", "-l", f"node.kubernetes.io/instance-type={instance_type}", "-o", "name"],
            capture_output=True, text=True, timeout=10
        )
        nodes = [n for n in r.stdout.strip().split("\n") if n]
        if nodes:
            print(f"  ✓ {instance_type} node(s) found: {len(nodes)}")
        else:
            errors.append(f"No {instance_type} nodes found in cluster. Check FTP node group.")
    except Exception as e:
        errors.append(f"Cannot check nodes: {e}")

    # AWS credentials
    try:
        sts = boto3.client("sts", region_name=defaults.get("region", "us-east-2"))
        identity = sts.get_caller_identity()
        print(f"  ✓ AWS credentials valid (account: {identity['Account']})")
    except Exception as e:
        errors.append(f"AWS credentials invalid: {e}")

    if errors:
        print(f"\n✗ Dependency check FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print(f"\n✓ All checks passed.")


# --- Deployment (kubectl) ---

def deploy_model_k8s(model_key, model_cfg, defaults):
    """Deploy vLLM on the p6-b200 node via kubectl."""
    namespace = defaults.get("namespace", "default")
    # Ensure namespace exists
    subprocess.run(["kubectl", "create", "namespace", namespace], capture_output=True)
    image = defaults.get("vllm_image", "vllm/vllm-openai:v0.20.2")
    model_id = model_cfg["model_name"]
    s3_model_uri = model_cfg.get("s3_model_uri", "")
    num_gpus = str(model_cfg.get("num_gpus", 8))
    hf_token = os.environ.get("HF_TOKEN", "")
    name = model_key[:40]
    vllm_config = defaults.get("vllm_config", {})
    max_model_len = str(vllm_config.get("max_model_len", "16384"))
    gpu_memory_utilization = str(vllm_config.get("gpu_memory_utilization", "0.9"))
    # Per-model vLLM CLI args for HyperPod (EC2 image doesn't have SM_VLLM_* translation)
    extra_args = model_cfg.get("hyperpod_args", [])
    # Format extra args as YAML list items for the manifest template
    extra_args_yaml = "\n".join(f'            - "{arg}"' for arg in extra_args) if extra_args else ""

    # Per-model env vars for HyperPod pod (e.g., VLLM_USE_FLASHINFER_MOE_FP4)
    hyperpod_env = model_cfg.get("hyperpod_env", {})
    extra_env_yaml = "\n".join(
        f'            - name: {k}\n              value: "{v}"' for k, v in hyperpod_env.items()
    ) if hyperpod_env else ""

    # K8s node label uses the full instance type (including ml. prefix on HyperPod)
    instance_type = model_cfg.get("instance_type", "ml.p6-b200.48xlarge")
    # Use S3 URI as model path if set, otherwise HuggingFace ID
    model_source = s3_model_uri if s3_model_uri else model_id
    served_model_name=model_id

    print(f"\n{'='*60}")
    print(f"[deploy-k8s] {model_key}")
    print(f"  image: {image}")
    print(f"  model: {model_id}")
    if s3_model_uri:
        print(f"  source: {s3_model_uri} (S3)")
    else:
        print(f"  source: HuggingFace (direct download)")
    print(f"  gpus: {num_gpus}")
    print(f"  namespace: {namespace}")
    print(f"  served model: {served_model_name}")
    if extra_args:
        print(f"  extra vllm args: {extra_args}")
    print(f"{'='*60}")

    manifest = VLLM_K8S_MANIFEST.format(
        name=name, namespace=namespace, image=image,
        model_id=model_source, num_gpus=num_gpus, hf_token=hf_token,
        max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
        instance_type=instance_type, served_model_name=model_id,
        extra_args=extra_args_yaml,
        extra_env=extra_env_yaml,
    )

    # Delete stale service to force NLB recreation
    subprocess.run(["kubectl", "delete", "svc", f"vllm-{name}", "-n", namespace, "--ignore-not-found"], capture_output=True)
    time.sleep(5)

    # Apply manifest
    r = subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest, text=True, capture_output=True)
    if r.returncode != 0:
        return {"success": False, "error": f"kubectl apply failed: {r.stderr}"}
    print(f"  ✓ Applied deployment + service")

    # Wait for pod ready
    print(f"  ⏳ Waiting for pod ready (model download + load)...")
    for _ in range(60):
        r = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-l", f"app=vllm-{name}",
             "-o", "jsonpath={.items[0].status.conditions[?(@.type=='Ready')].status}"],
            capture_output=True, text=True
        )
        if r.stdout.strip() == "True":
            print(f"  ✓ Pod ready")
            break
        time.sleep(30)
    else:
        return {"success": False, "error": "Pod not ready after 30 min"}

    # Get Pod IP directly (benchmark runner is in same VPC)
    # print(f"  ⏳ Getting pod IP...")
    # for _ in range(10):
    #     r = subprocess.run(
    #         ["kubectl", "get", "pod", "-n", namespace, "-l", f"app=vllm-{name}",
    #         "-o", "jsonpath={.items[0].status.podIP}"],
    #         capture_output=True, text=True
    #     )
    #     pod_ip = r.stdout.strip()
    #     if pod_ip:
    #         url = f"http://{pod_ip}:8080/v1"
    #         print(f"  ✓ Pod URL: {url}")
    #         return {"success": True, "url": url, "name": name}
    #     time.sleep(10)
    # return {"success": False, "error": "Pod IP not available"}

    # Get ELB URL
    print(f"  ⏳ Waiting for LoadBalancer URL...")
    for _ in range(20):
        r = subprocess.run(
            ["kubectl", "get", "svc", f"vllm-{name}", "-n", namespace,
             "-o", "jsonpath={.status.loadBalancer.ingress[0].hostname}"],
            capture_output=True, text=True
        )
        hostname = r.stdout.strip()
        # if hostname:
        #     url = f"http://{hostname}/v1"
        #     print(f"  ✓ ELB URL: {url}")
        #     return {"success": True, "url": url, "name": name}
        if hostname:
              url = f"http://{hostname}/v1"
              print(f"  ✓ ELB URL: {url}")
              # Wait for NLB target to be healthy
              print(f"  ⏳ Waiting for NLB target registration...")
              for _ in range(30):
                  r = subprocess.run(
                      ["kubectl", "get", "endpoints", f"vllm-{name}", "-n", namespace,
                       "-o", "jsonpath={.subsets[0].addresses[0].ip}"],
                      capture_output=True, text=True
                  )
                  if r.stdout.strip():
                      print(f"  ✓ Endpoint registered: {r.stdout.strip()}")
                      break
                  time.sleep(10)
              time.sleep(30)  # Extra wait for NLB health check to pass
              return {"success": True, "url": url, "name": name}
        time.sleep(15)
    return {"success": False, "error": "LoadBalancer hostname not assigned"}


def cleanup_model_k8s(model_key, defaults):
    namespace = defaults.get("namespace", "default")
    name = model_key[:40]
    print(f"\n[cleanup-k8s] {model_key}")
    subprocess.run(["kubectl", "delete", "deployment", f"vllm-{name}", "-n", namespace, "--ignore-not-found"], capture_output=True)
    subprocess.run(["kubectl", "delete", "svc", f"vllm-{name}", "-n", namespace, "--ignore-not-found"], capture_output=True)
    # Wait for pod to fully terminate before returning
    print(f"  ⏳ Waiting for pod termination...")
    for _ in range(30):
        r = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-l", f"app=vllm-{name}", "-o", "name"],
            capture_output=True, text=True
        )
        if not r.stdout.strip():
            break
        time.sleep(10)
    print(f"  ✓ Cleanup complete")


# --- Benchmarking (direct-URL via AIPerf) ---

def get_sagemaker_client(defaults):
    return boto3.client("sagemaker", region_name=defaults.get("region", "us-east-2"),
                          config=boto3.session.Config(parameter_validation=False))


def run_hyperpod_benchmark(client, job, elb_url, defaults, models=None):
    """Submit benchmark job targeting the HyperPod ELB URL.
    Uses the 'url' parameter in the workload spec + NetworkConfig for VPC access.
    """
    role = defaults["role_arn"]
    s3_output = defaults["s3_output"].format(region=defaults.get("region", "us-east-2"))

    # Hierarchical path: {env}/{model}/{workload}/{instance}/{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    s3_job_path = f"{s3_output}/{job['model_key']}/{job['workload_key']}/p6-b200-c{job['concurrency']}/{timestamp}/"

    ts = datetime.now().strftime("%m%d%H%M")
    job_name = f"{job['id'][:50]}-{ts}".replace("_", "-")
    config_name = f"hp-{job['model_key']}-{job['workload_key']}-c{job['concurrency']}"[:63].replace("_", "-")

    print(f"\n{'─'*60}")
    print(f"[benchmark] {job['id']}")
    print(f"  target: {elb_url}")
    print(f"  concurrency={job['concurrency']} in={job['input_tokens']} out={job['output_tokens']}")
    print(f"{'─'*60}")

    workload_spec = {
        "benchmark": {"type": "aiperf"},
        "parameters": {
            "url": elb_url,
            "model": job["model_name"],
            "tokenizer": job["model_name"],
            "concurrency": job["concurrency"],
            "streaming": job["streaming"],
            "prompt_input_tokens_mean": job["input_tokens"],
            "output_tokens_mean": job["output_tokens"],
            "benchmark_duration": job["duration"],
            "warmup_duration": job["warmup"],
        },
        "tooling": {"api_standard": "openai"},
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
        print(f"  ✗ {e}")
        return {"success": False, "error": str(e)}

    # Submit job with direct-URL sentinel (service routes to url in workload spec)
    try:
        client.delete_ai_benchmark_job(AIBenchmarkJobName=job_name)
    except Exception:
        pass
    try:
        create_kwargs = {
            "AIBenchmarkJobName": job_name,
            "AIWorkloadConfigIdentifier": config_name,
            "BenchmarkTarget": {"Endpoint": {"Identifier": DIRECT_URL_SENTINEL}},
            "OutputConfig": {"S3OutputLocation": s3_job_path},
            "RoleArn": role,
            "Tags": [
                {"Key": "project", "Value": "benchmarking-initiative"},
                {"Key": "environment", "Value": "hyperpod"},
                {"Key": "model", "Value": job["model_key"]},
                {"Key": "workload", "Value": job["workload_key"]},
                {"Key": "concurrency", "Value": str(job["concurrency"])},
                {"Key": "source-region", "Value": defaults.get("region", "")},
            ],
        }
        # Add NetworkConfig if VPC settings provided (for ELB access)
        vpc = defaults.get("vpc_config")
        if vpc:
            create_kwargs["NetworkConfig"] = {"VpcConfig": vpc}
        client.create_ai_benchmark_job(**create_kwargs)
        print(f"  ✓ Created benchmark job: {job_name}")
    except Exception as e:
        print(f"  ✗ {e}")
        return {"success": False, "error": str(e)}

    # Poll
    for _ in range(60):
        try:
            resp = client.describe_ai_benchmark_job(AIBenchmarkJobName=job_name)
            status = resp["AIBenchmarkJobStatus"]
        except Exception:
            time.sleep(30)
            continue
        if status == "Completed":
            print(f"  ✓ Completed")
            s3_loc = resp["OutputConfig"]["S3OutputLocation"]
            from athena_writer import write_athena_record
            model_cfg = (models or {}).get(job["model_key"], {})
            write_athena_record({
                "job_id": job_name,
                "environment": "hyperpod",
                "model_key": job["model_key"],
                "model_name": job["model_name"],
                "workload": job["workload_key"],
                "concurrency": job["concurrency"],
                "input_tokens": job["input_tokens"],
                "output_tokens": job["output_tokens"],
                "streaming": job["streaming"],
                "duration": job["duration"],
                "warmup": job["warmup"],
                "instance_type": model_cfg.get("instance_type", defaults.get("instance_type", "")),
                "num_gpus": job.get("num_gpus", model_cfg.get("num_gpus", 8)),
                "dataset": job.get("dataset"),
                "source_region": defaults.get("region", ""),
                "s3_output": s3_loc,
                "timestamp": timestamp,
            }, model_config=model_cfg, defaults_config=defaults, config_source="benchmarks.yaml")
            return {"success": True, "status": "completed", "s3_output": s3_loc}
        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "unknown")
            print(f"  ✗ {status}: {reason}")
            return {"success": False, "error": reason}
        time.sleep(30)
    return {"success": False, "error": "Timed out"}


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="HyperPod EKS benchmarking — deploys vLLM on EKS, exposes via NLB, benchmarks via direct-URL.",
        epilog="""
examples:
  %(prog)s --validate                    Show expanded job matrix without running
  %(prog)s --check                       Verify kubectl, nodes, and credentials
  %(prog)s                               Deploy + benchmark all models
  %(prog)s --model=gemma                 Run only models matching 'gemma'
  %(prog)s --workload=rag               Run only workloads matching 'rag'
  %(prog)s --deploy-only                 Deploy vLLM pods + NLB, skip benchmarking
  %(prog)s --benchmark-only              Benchmark existing pods (skip deploy)
  %(prog)s --cleanup                     Delete pods and services
  %(prog)s --submit                      Submit as unattended Processing Job

config: reads from benchmarks.yaml (models, workloads, hyperpod_defaults, hyperpod_benchmarks)
flow:   auto-configures kubectl → deploys vLLM pod → waits for readiness → creates NLB → benchmarks
note:   single p6-b200 node — models are deployed sequentially with cleanup between them
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default="../benchmarks.yaml", help="Path to benchmarks.yaml config (default: ../benchmarks.yaml)")
    parser.add_argument("--check", action="store_true", help="Verify kubectl access, node availability, and credentials")
    parser.add_argument("--validate", action="store_true", help="Show expanded job matrix and exit (no AWS calls)")
    parser.add_argument("--model", help="Filter by model key substring (e.g. 'gemma', 'qwen3')")
    parser.add_argument("--workload", help="Filter by workload key substring (e.g. 'rag', 'chat')")
    parser.add_argument("--deploy-only", action="store_true", help="Deploy vLLM pods and NLB only, skip benchmarking")
    parser.add_argument("--benchmark-only", action="store_true", help="Benchmark existing pods (skip deploy, use existing ELB)")
    parser.add_argument("--cleanup", action="store_true", help="Delete all deployed pods and services")
    parser.add_argument("--submit", action="store_true", help="Submit as a remote SageMaker Processing Job")
    args = parser.parse_args()

    if args.submit:
        # HyperPod needs VPC access + kubectl — use a wrapper entrypoint
        from benchmark import submit_processing_job

        # Override the entrypoint to install kubectl first
        config = load_config(args.config)
        defaults = config.get("hyperpod_defaults", {})
        region = defaults.get("region", "us-east-2")
        role = defaults.get("role_arn")
        if not role:
            raise ValueError("'role_arn' must be set in hyperpod_defaults")
        s3_output = defaults.get("s3_output", "").format(region=region)
        bucket = s3_output.replace("s3://", "").split("/")[0] if s3_output else f"sagemaker-benchmark-{region}-{boto3.client('sts', region_name=region).get_caller_identity()['Account']}"
        job_name = f"bench-hp-{datetime.now().strftime('%m%d-%H%M')}"[:63]

        sm = boto3.client("sagemaker", region_name=region)
        s3 = boto3.client("s3", region_name="us-east-2")

        # Upload script + config + deps
        script_dir = os.path.dirname(__file__)
        s3.put_object(Bucket=bucket, Key=f"processing-configs/{job_name}/script.py", Body=open(__file__).read())
        s3.put_object(Bucket=bucket, Key=f"processing-configs/{job_name}/benchmarks.yaml", Body=open(args.config).read())
        req_path = os.path.join(script_dir, "requirements.txt")
        if os.path.exists(req_path):
            s3.put_object(Bucket=bucket, Key=f"processing-configs/{job_name}/requirements.txt", Body=open(req_path).read())
        athena_path = os.path.join(script_dir, "athena_writer.py")
        if os.path.exists(athena_path):
            s3.put_object(Bucket=bucket, Key=f"processing-configs/{job_name}/athena_writer.py", Body=open(athena_path).read())
        # Bootstrap script — installs kubectl + deps, then runs the main script
        bootstrap = (
            "#!/bin/bash\nset -e\n"
            "curl -sLO https://dl.k8s.io/release/v1.31.0/bin/linux/amd64/kubectl\n"
            "chmod +x kubectl && mv kubectl /usr/local/bin/\n"
            "pip install -q -r /opt/ml/processing/input/script/requirements.txt\n"
            "pip install -q --upgrade awscli\n"
            "exec python3 /opt/ml/processing/input/script/script.py "
            "/opt/ml/processing/input/script/benchmarks.yaml \"$@\"\n"
        )
        s3.put_object(Bucket=bucket, Key=f"processing-configs/{job_name}/bootstrap.sh", Body=bootstrap)

        container_args = []
        if args.model:
            container_args += ["--model", args.model]
        if args.workload:
            container_args += ["--workload", args.workload]

        # VPC config for EKS access
        vpc_config = defaults.get("vpc_config", {})

        print(f"\n{'='*60}")
        print(f"Submitting HyperPod Processing Job: {job_name}")
        print(f"  Region: {region}")
        print(f"  VPC: {'configured' if vpc_config else 'NOT SET — may not reach EKS'}")
        print(f"{'='*60}\n")

        create_kwargs = {
            "ProcessingJobName": job_name,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m7i.4xlarge",
                    "VolumeSizeInGB": 50,
                }
            },
            "AppSpecification": {
                "ImageUri": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-cpu-py311",
                "ContainerEntrypoint": ["/bin/bash", "/opt/ml/processing/input/script/bootstrap.sh"],
                **({"ContainerArguments": container_args} if container_args else {}),
            },
            "ProcessingInputs": [
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
            "ProcessingOutputConfig": {
                "Outputs": [{
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/processing-results/{job_name}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }]
            },
            "RoleArn": role,
            "StoppingCondition": {"MaxRuntimeInSeconds": 432000},
            "NetworkConfig": {"EnableNetworkIsolation": False},
        }

        # # Add VPC config if provided (required for EKS access)
        # if vpc_config:
        #     create_kwargs["NetworkConfig"]["VpcConfig"] = vpc_config

        sm.create_processing_job(**create_kwargs)
        print(f"  ✓ Job submitted: {job_name}")
        print(f"  Monitor: aws sagemaker describe-processing-job --processing-job-name {job_name} --region {region}")
        return

    config = load_config(args.config)
    defaults = config.get("hyperpod_defaults", {})
    if not defaults:
        print("ERROR: No 'hyperpod_defaults' section in config", file=sys.stderr)
        sys.exit(1)

    if args.check:
        configure_kubectl(defaults)
        check_dependencies(defaults)
        return

    jobs = expand_hyperpod_jobs(config, args.model, args.workload)

    if args.validate:
        models = sorted(set(j["model_key"] for j in jobs))
        workloads = sorted(set(j["workload_key"] for j in jobs))
        print(f"Expanded {len(jobs)} HyperPod benchmark job(s)\n")
        print(f"Models ({len(models)}):")
        for m in models:
            mc = config["models"][m]
            print(f"  ✓ {m} ({mc['model_name']}, {mc.get('num_gpus',8)} GPUs)")
        print(f"\nWorkloads ({len(workloads)}):")
        for w in workloads:
            wl = config["workloads"][w]
            print(f"  ✓ {w}: in={wl['input_tokens']} out={wl['output_tokens']} c=[{','.join(map(str, wl['concurrency']))}]")
        print(f"\nTotal jobs: {len(jobs)}")
        return

    # Configure kubectl for the cluster
    configure_kubectl(defaults)

    # Ensure S3 output bucket exists
    s3_output = defaults.get("s3_output", "").format(region=defaults.get("region", "us-east-2"))
    if s3_output:
        bucket = s3_output.replace("s3://", "").split("/")[0]
        region = defaults.get("region", "us-east-2")
        s3 = boto3.client("s3", region_name=region)
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            try:
                s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
                print(f"  ✓ Created S3 bucket: {bucket}")
            except Exception:
                pass

    if args.cleanup:
        models = sorted(set(j["model_key"] for j in jobs))
        for m in models:
            cleanup_model_k8s(m, defaults)
        return

    # Group by model
    by_model = {}
    for j in jobs:
        by_model.setdefault(j["model_key"], []).append(j)

    run_dir = RESULTS_DIR / "hyperpod-latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    client = get_sagemaker_client(defaults)

    for model_key, model_jobs in by_model.items():
        model_cfg = config["models"][model_key]

        # Deploy
        if not args.benchmark_only:
            deploy_result = deploy_model_k8s(model_key, model_cfg, defaults)
            if not deploy_result["success"]:
                print(f"  ✗ Deploy failed: {deploy_result['error']}")
                for j in model_jobs:
                    results.append({"id": j["id"], "success": False, "error": deploy_result["error"]})
                continue
            elb_url = deploy_result["url"]
        else:
            # Get existing ELB URL
            namespace = defaults.get("namespace", "default")
            name = model_key[:40]
            r = subprocess.run(
                ["kubectl", "get", "svc", f"vllm-{name}", "-n", namespace,
                 "-o", "jsonpath={.status.loadBalancer.ingress[0].hostname}"],
                capture_output=True, text=True
            )
            elb_url = f"http://{r.stdout.strip()}/v1"
            print(f"  Using existing ELB: {elb_url}")

        if args.deploy_only:
            continue

        # Benchmark
        for job in model_jobs:
            result_file = run_dir / f"{job['id']}.json"
            if result_file.exists():
                existing = json.loads(result_file.read_text())
                if existing.get("success"):
                    print(f"\n  ⏭️  Skipping {job['id']} (already completed)")
                    results.append(existing)
                    continue

            result = run_hyperpod_benchmark(client, job, elb_url, defaults, models=config.get("models", {}))
            result["id"] = job["id"]
            result["model_key"] = job["model_key"]
            results.append(result)
            (run_dir / f"{job['id']}.json").write_text(json.dumps(result, indent=2, default=str))

        # Cleanup between models (single node)
        if defaults.get("cleanup", True) and not args.benchmark_only:
            cleanup_model_k8s(model_key, defaults)
            time.sleep(30)

    # Summary
    summary = {"total": len(results), "completed": sum(1 for r in results if r.get("success")),
               "failed": sum(1 for r in results if not r.get("success"))}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{'═'*60}")
    print(f"HYPERPOD BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    print(f"Total: {summary['total']} | Completed: {summary['completed']} | Failed: {summary['failed']}")
    print(f"Results: {run_dir}/")


if __name__ == "__main__":
    main()
