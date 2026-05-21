"""
Unified deployment engine for SageMaker inference endpoints.

Replaces deploy_matrix.py (standard pattern) and deploy_p5e_models.py (IC pattern).
All configuration comes from YAML recipes via config_loader.

Usage:
    python -m scripts.deployer deploy recipes/qwen3-32b-g7e-eagle3.yaml
    python -m scripts.deployer smoke-test recipes/recipe.yaml --endpoint NAME
    python -m scripts.deployer cleanup recipes/recipe.yaml --endpoint NAME
    python -m scripts.deployer list --region us-west-2
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.config import Config

from scripts.config_loader import (
    BenchmarkConfig,
    build_container_uri,
    build_endpoint_name,
    build_env_vars,
    get_optimization_label,
    load_config,
    print_config_summary,
    validate_config,
)


@dataclass
class DeploymentResult:
    endpoint_name: str
    ic_name: Optional[str]
    success: bool
    region: str
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deploy(config: BenchmarkConfig) -> DeploymentResult:
    """Deploy an inference endpoint from a benchmark config.

    Routes based on platform (sagemaker/hyperpod) and pattern (standard/IC).
    """
    start = time.time()
    region = config.deployment.endpoint.region
    platform = config.deployment.platform

    if platform == "hyperpod":
        result = _deploy_hyperpod(config)
        result.elapsed_sec = time.time() - start
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n[{status}] HyperPod endpoint: {result.endpoint_name} ({result.elapsed_sec:.0f}s)")
        return result

    # SageMaker platform
    warnings = validate_config(config)
    for w in warnings:
        print(f"  WARNING: {w}")

    image_uri = build_container_uri(config)
    env = build_env_vars(config)
    endpoint_name = build_endpoint_name(config)

    _print_deploy_summary(config, image_uri, env, endpoint_name)

    pattern = config.deployment.endpoint.pattern
    if pattern == "inference_component":
        result = _deploy_ic(config, image_uri, env, endpoint_name)
    else:
        result = _deploy_standard(config, image_uri, env, endpoint_name)

    result.elapsed_sec = time.time() - start
    status = "SUCCESS" if result.success else "FAILED"
    print(f"\n[{status}] {endpoint_name} ({result.elapsed_sec:.0f}s)")
    return result


def wait_for_endpoint(endpoint_name: str, region: str, timeout_sec: int = 1200) -> bool:
    """Poll endpoint until InService, Failed, or timeout."""
    sm = boto3.client("sagemaker", region_name=region)
    print(f"Waiting for {endpoint_name} to become InService...")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            resp = sm.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]
            if status == "InService":
                print(f"  {endpoint_name} is InService! ({time.time() - start:.0f}s)")
                return True
            if status == "Failed":
                reason = resp.get("FailureReason", "unknown")
                print(f"  {endpoint_name} FAILED: {reason}")
                return False
            print(f"  {endpoint_name}: {status} ({time.time() - start:.0f}s)")
        except Exception as e:
            print(f"  Error checking status: {e}")
        time.sleep(30)
    print(f"  TIMEOUT waiting for {endpoint_name} ({timeout_sec}s)")
    return False


def smoke_test(config: BenchmarkConfig, endpoint_name: str, ic_name: str = None) -> bool:
    """Send a quick test request to verify the endpoint is working."""
    region = config.deployment.endpoint.region
    client = boto3.client(
        "sagemaker-runtime", region_name=region,
        config=Config(read_timeout=120, retries={"max_attempts": 0}),
    )

    payload = {
        "messages": [{"role": "user", "content": "Hello! Tell me a short joke."}],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    # Apply model-specific params from config
    extra = config.benchmark.inference_params.extra_payload
    if extra:
        payload.update(extra)

    invoke_kwargs = {
        "EndpointName": endpoint_name,
        "ContentType": "application/json",
        "Body": json.dumps(payload),
    }
    if ic_name:
        invoke_kwargs["InferenceComponentName"] = ic_name

    print(f"\nSmoke test: {endpoint_name}" + (f" (IC: {ic_name})" if ic_name else ""))
    start = time.perf_counter()
    try:
        response = client.invoke_endpoint(**invoke_kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        body = json.loads(response["Body"].read())

        usage = body.get("usage", {})
        output = body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        print(f"  OK ({elapsed:.0f}ms)")
        print(f"  Tokens: in={usage.get('prompt_tokens', 0)}, out={usage.get('completion_tokens', 0)}")
        print(f"  Output: {output[:200]}")
        return True
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  FAILED ({elapsed:.0f}ms): {e}")
        return False


def cleanup(endpoint_name: str, region: str, pattern: str = "standard",
            platform: str = "sagemaker", hyperpod_config=None):
    """Delete an endpoint and all associated resources.

    For HyperPod: kubectl delete inferenceendpointconfig.
    For IC pattern: delete ICs first, wait, then endpoint/config/model.
    For standard pattern: delete endpoint, config, model.
    """
    if platform == "hyperpod":
        _cleanup_hyperpod(endpoint_name, hyperpod_config)
        return
    sm = boto3.client("sagemaker", region_name=region)
    print(f"Cleaning up: {endpoint_name} ({region})")

    # IC pattern: delete inference components first
    if pattern == "inference_component":
        try:
            ics = sm.list_inference_components(
                EndpointNameEquals=endpoint_name, MaxResults=10,
            )
            for ic in ics.get("InferenceComponents", []):
                ic_name = ic["InferenceComponentName"]
                print(f"  Deleting IC: {ic_name}")
                sm.delete_inference_component(InferenceComponentName=ic_name)
            if ics.get("InferenceComponents"):
                print("  Waiting for ICs to delete...")
                time.sleep(30)
        except Exception as e:
            print(f"  IC cleanup error: {e}")

    # Delete endpoint
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"  Deleted endpoint: {endpoint_name}")
    except Exception as e:
        print(f"  Error deleting endpoint: {e}")

    # Delete endpoint config
    for epc_candidate in [endpoint_name, f"epc-{endpoint_name}"]:
        try:
            sm.delete_endpoint_config(EndpointConfigName=epc_candidate)
            print(f"  Deleted endpoint config: {epc_candidate}")
            break
        except Exception:
            pass

    # Delete model
    for mdl_candidate in [endpoint_name, f"mdl-{endpoint_name}"]:
        try:
            sm.delete_model(ModelName=mdl_candidate)
            print(f"  Deleted model: {mdl_candidate}")
            break
        except Exception:
            pass

    print("  Cleanup complete.")


def find_endpoints_by_prefix(region: str, prefix: str) -> list[dict]:
    """Find endpoints matching a prefix. Returns list of {name, status} dicts."""
    sm = boto3.client("sagemaker", region_name=region)
    kwargs = {"MaxResults": 100}
    if prefix:
        kwargs["NameContains"] = prefix
    endpoints = sm.list_endpoints(**kwargs)
    return [
        {"name": ep["EndpointName"], "status": ep["EndpointStatus"]}
        for ep in endpoints.get("Endpoints", [])
    ]


def list_endpoints(region: str, name_prefix: str = None):
    """List SageMaker endpoints matching an optional prefix."""
    sm = boto3.client("sagemaker", region_name=region)
    kwargs = {"MaxResults": 100}
    if name_prefix:
        kwargs["NameContains"] = name_prefix

    endpoints = sm.list_endpoints(**kwargs)
    if not endpoints["Endpoints"]:
        print(f"[{region}] No endpoints found" + (f" matching '{name_prefix}'" if name_prefix else "") + ".")
        return

    print(f"\n[{region}]")
    for ep in endpoints["Endpoints"]:
        name = ep["EndpointName"]
        status = ep["EndpointStatus"]
        # Get instance type
        try:
            cfg = sm.describe_endpoint_config(EndpointConfigName=name)
            inst = cfg["ProductionVariants"][0].get("InstanceType", "unknown")
        except Exception:
            inst = "unknown"
        print(f"  {name} | {inst} | {status}")

        # List ICs if any
        try:
            ics = sm.list_inference_components(EndpointNameEquals=name, MaxResults=10)
            for ic in ics.get("InferenceComponents", []):
                ic_status = ic.get("InferenceComponentStatus", "unknown")
                print(f"    IC: {ic['InferenceComponentName']} | {ic_status}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _deploy_standard(config: BenchmarkConfig, image_uri: str,
                     env: dict, endpoint_name: str) -> DeploymentResult:
    """Deploy using SageMaker SDK Model.deploy() — standard pattern."""
    import sagemaker
    from sagemaker.model import Model
    from sagemaker.predictor import Predictor
    from sagemaker.serializers import JSONSerializer

    d = config.deployment
    region = d.endpoint.region

    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    model = Model(
        image_uri=image_uri,
        role=d.endpoint.role_arn,
        predictor_cls=Predictor,
        env=env,
        sagemaker_session=sm_session,
    )

    try:
        model.deploy(
            instance_type=d.instance.type,
            initial_instance_count=d.instance.count,
            endpoint_name=endpoint_name,
            container_startup_health_check_timeout=d.endpoint.health_check_timeout,
            serializer=JSONSerializer(),
        )
        print(f"\nEndpoint deployed: {endpoint_name}")
        return DeploymentResult(
            endpoint_name=endpoint_name, ic_name=None,
            success=True, region=region, elapsed_sec=0,
        )
    except Exception as e:
        print(f"\nDeployment failed: {e}")
        return DeploymentResult(
            endpoint_name=endpoint_name, ic_name=None,
            success=False, region=region, elapsed_sec=0,
        )


def _deploy_ic(config: BenchmarkConfig, image_uri: str,
               env: dict, endpoint_name: str) -> DeploymentResult:
    """Deploy using Inference Component pattern — for p5e/LMI."""
    d = config.deployment
    region = d.endpoint.region
    role_arn = d.endpoint.role_arn
    ic_cfg = d.endpoint.ic

    model_name = f"mdl-{endpoint_name}"[:63]
    epc_name = f"epc-{endpoint_name}"[:63]
    ic_name = f"ic-{endpoint_name}"[:63]

    sm = boto3.client("sagemaker", region_name=region)

    # 1. Create Model
    print("Creating SageMaker Model...")
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": image_uri, "Environment": env},
            ExecutionRoleArn=role_arn,
        )
        print(f"  Model created: {model_name}")
    except Exception as e:
        print(f"  Failed to create model: {e}")
        return DeploymentResult(endpoint_name, None, False, region, 0)

    # 2. Create Endpoint Config (IC-enabled)
    print("Creating IC-enabled Endpoint Config...")
    try:
        variant = {
            "VariantName": "AllTraffic",
            "InstanceType": d.instance.type,
            "InitialInstanceCount": d.instance.count,
            "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
        }
        if ic_cfg.managed_scaling:
            variant["ManagedInstanceScaling"] = {
                "Status": "ENABLED",
                "MinInstanceCount": 1,
                "MaxInstanceCount": 1,
            }
        sm.create_endpoint_config(
            EndpointConfigName=epc_name,
            ExecutionRoleArn=role_arn,
            ProductionVariants=[variant],
        )
        print(f"  Endpoint config created: {epc_name}")
    except Exception as e:
        print(f"  Failed to create endpoint config: {e}")
        return DeploymentResult(endpoint_name, None, False, region, 0)

    # 3. Create Endpoint
    print("Creating Endpoint...")
    try:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=epc_name)
        print(f"  Endpoint creating: {endpoint_name}")
    except Exception as e:
        print(f"  Failed to create endpoint: {e}")
        return DeploymentResult(endpoint_name, None, False, region, 0)

    # 4. Wait for endpoint InService
    if not wait_for_endpoint(endpoint_name, region, timeout_sec=1200):
        return DeploymentResult(endpoint_name, None, False, region, 0)

    # 5. Create Inference Component
    print("Creating Inference Component...")
    try:
        sm.create_inference_component(
            InferenceComponentName=ic_name,
            EndpointName=endpoint_name,
            VariantName="AllTraffic",
            Specification={
                "ModelName": model_name,
                "ComputeResourceRequirements": {
                    "NumberOfAcceleratorDevicesRequired": ic_cfg.num_accelerators,
                    "MinMemoryRequiredInMb": ic_cfg.min_memory_mb,
                },
            },
            RuntimeConfig={"CopyCount": 1},
        )
        print(f"  IC creating: {ic_name}")
    except Exception as e:
        print(f"  Failed to create IC: {e}")
        return DeploymentResult(endpoint_name, ic_name, False, region, 0)

    # 6. Wait for IC InService (model loading — can take 30+ min)
    print("Waiting for IC to become InService (model loading)...")
    ic_start = time.time()
    timeout = d.endpoint.health_check_timeout
    while time.time() - ic_start < timeout:
        try:
            resp = sm.describe_inference_component(InferenceComponentName=ic_name)
            status = resp["InferenceComponentStatus"]
            if status == "InService":
                print(f"  IC InService! ({time.time() - ic_start:.0f}s)")
                return DeploymentResult(endpoint_name, ic_name, True, region, 0)
            if status == "Failed":
                reason = resp.get("FailureReason", "unknown")
                print(f"  IC FAILED: {reason}")
                return DeploymentResult(endpoint_name, ic_name, False, region, 0)
            print(f"  IC Status: {status} ({time.time() - ic_start:.0f}s)")
        except Exception as e:
            print(f"  Error checking IC: {e}")
        time.sleep(30)

    print(f"  TIMEOUT waiting for IC ({timeout}s)")
    return DeploymentResult(endpoint_name, ic_name, False, region, 0)


def _print_deploy_summary(config: BenchmarkConfig, image_uri: str,
                          env: dict, endpoint_name: str):
    d = config.deployment
    pattern = d.endpoint.pattern
    opt = get_optimization_label(config)
    container_tag = image_uri.split("/")[-1]

    print(f"\n{'=' * 70}")
    print(f"Deploying: {endpoint_name}")
    print(f"  Model: {d.model.id}")
    print(f"  Instance: {d.instance.type} (x{d.instance.count})")
    print(f"  Container: {d.container.type} ({container_tag})")
    print(f"  Optimization: {opt}")
    print(f"  Pattern: {pattern}")
    print(f"  Region: {d.endpoint.region}")
    print(f"  Health check timeout: {d.endpoint.health_check_timeout}s")
    if d.speculative_decoding.enabled:
        print(f"  Speculator: {d.speculative_decoding.model}")
    print(f"  Env vars:")
    for k, v in sorted(env.items()):
        display_v = v if len(str(v)) < 80 else str(v)[:77] + "..."
        print(f"    {k}={display_v}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# HyperPod deployment (InferenceEndpointConfig CRD via kubectl)
# ---------------------------------------------------------------------------

def _deploy_hyperpod(config: BenchmarkConfig) -> DeploymentResult:
    """Deploy model to HyperPod EKS cluster via InferenceEndpointConfig CRD."""
    import subprocess
    import tempfile

    d = config.deployment
    hp = d.hyperpod
    region = d.endpoint.region
    endpoint_name = hp.cluster_name or build_endpoint_name(config)

    # Derive worker image from container config if not set
    worker_image = hp.worker.image
    if not worker_image:
        worker_image = build_container_uri(config).replace("-sagemaker", "-ec2-v1.0")

    # Build vLLM args from vllm config
    vllm_args = [
        "--model", "/opt/ml/model",
        "--max-model-len", str(d.vllm.max_model_len),
        "--tensor-parallel-size", str(d.vllm.tensor_parallel_size),
        "--gpu-memory-utilization", str(d.vllm.gpu_memory_utilization),
        "--dtype", d.vllm.dtype,
    ]
    if d.vllm.max_num_seqs:
        vllm_args.extend(["--max-num-seqs", str(d.vllm.max_num_seqs)])

    # Build model source spec
    if hp.model_source.type == "s3":
        model_source_yaml = f"""    modelSourceType: s3
    s3Storage:
      bucketName: {hp.model_source.s3_bucket}
      region: {hp.model_source.s3_region or region}
    modelLocation: {hp.model_source.model_location}"""
    else:  # fsx
        model_source_yaml = f"""    modelSourceType: fsx
    fsxStorage:
      fileSystemId: {hp.model_source.fsx_file_system_id}
    modelLocation: {hp.model_source.model_location}"""

    # Build InferenceEndpointConfig YAML
    manifest = f"""apiVersion: inference.sagemaker.aws.amazon.com/v1
kind: InferenceEndpointConfig
metadata:
  name: {endpoint_name}
  namespace: {hp.namespace}
spec:
  modelName: {d.model.short_name or d.model.id.split('/')[-1].lower()}
  instanceType: {d.instance.type}
  replicas: {hp.replicas}
  invocationEndpoint: {hp.invocation_endpoint}
  kvCacheSpec:
    enableL1Cache: {str(hp.kv_cache.l1_cache).lower()}
    enableL2Cache: {str(hp.kv_cache.l2_cache).lower()}
    l2CacheSpec:
      l2CacheBackend: "{hp.kv_cache.l2_backend}"
  intelligentRoutingSpec:
    enabled: {str(hp.routing.enabled).lower()}
    routingStrategy: {hp.routing.strategy}
  modelSourceConfig:
{model_source_yaml}
  worker:
    image: {worker_image}
    args:
{chr(10).join(f'      - "{a}"' for a in vllm_args)}
    resources:
      limits:
        nvidia.com/gpu: "{hp.worker.gpu_count}"
      requests:
        cpu: "{hp.worker.cpu_request}"
        memory: {hp.worker.memory_request}
        nvidia.com/gpu: "{hp.worker.gpu_count}"
    modelInvocationPort:
      containerPort: 8080
      name: http
    modelVolumeMount:
      name: model-weights
      mountPath: /opt/ml/model
"""

    print(f"\n{'=' * 70}")
    print(f"Deploying to HyperPod: {endpoint_name}")
    print(f"  Model: {d.model.id}")
    print(f"  Instance: {d.instance.type}")
    print(f"  Cluster: {hp.cluster_name}")
    print(f"  Namespace: {hp.namespace}")
    print(f"  KV Cache: L1={hp.kv_cache.l1_cache}, L2={hp.kv_cache.l2_cache} ({hp.kv_cache.l2_backend})")
    print(f"  Routing: {hp.routing.strategy}")
    print(f"  Image: {worker_image}")
    print(f"{'=' * 70}\n")

    # Write manifest and apply
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(manifest)
            manifest_path = f.name

        print(f"Applying manifest: {manifest_path}")
        result = subprocess.run(
            ["kubectl", "apply", "-f", manifest_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  kubectl apply failed: {result.stderr}")
            return DeploymentResult(endpoint_name, None, False, region, 0)
        print(f"  {result.stdout.strip()}")

        # Wait for pods to be ready
        print(f"Waiting for pods (namespace={hp.namespace})...")
        ready = _wait_for_hyperpod_pods(endpoint_name, hp.namespace,
                                         timeout_sec=config.deployment.endpoint.health_check_timeout)
        return DeploymentResult(endpoint_name, None, ready, region, 0)

    except FileNotFoundError:
        print("  ERROR: kubectl not found. Install kubectl and configure kubeconfig.")
        return DeploymentResult(endpoint_name, None, False, region, 0)
    except Exception as e:
        print(f"  ERROR: {e}")
        return DeploymentResult(endpoint_name, None, False, region, 0)


def _wait_for_hyperpod_pods(name: str, namespace: str, timeout_sec: int = 1200) -> bool:
    """Wait for HyperPod inference pods to be Running."""
    import subprocess

    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace, "-l", f"app={name}",
                 "-o", "jsonpath={.items[*].status.phase}"],
                capture_output=True, text=True, timeout=10,
            )
            phases = result.stdout.strip().split()
            if phases and all(p == "Running" for p in phases):
                print(f"  All {len(phases)} pod(s) Running ({time.time() - start:.0f}s)")
                return True
            print(f"  Pods: {' '.join(phases) or 'pending'} ({time.time() - start:.0f}s)")
        except Exception as e:
            print(f"  Error checking pods: {e}")
        time.sleep(30)

    print(f"  TIMEOUT waiting for pods ({timeout_sec}s)")
    return False


def _cleanup_hyperpod(endpoint_name: str, hyperpod_config=None):
    """Delete HyperPod InferenceEndpointConfig."""
    import subprocess

    namespace = hyperpod_config.namespace if hyperpod_config else "default"
    print(f"Cleaning up HyperPod: {endpoint_name} (namespace={namespace})")

    try:
        result = subprocess.run(
            ["kubectl", "delete", "inferenceendpointconfig", endpoint_name,
             "-n", namespace, "--ignore-not-found"],
            capture_output=True, text=True, timeout=60,
        )
        print(f"  {result.stdout.strip() or result.stderr.strip()}")
    except FileNotFoundError:
        print("  ERROR: kubectl not found.")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("  Cleanup complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SageMaker Endpoint Deployer")
    sub = parser.add_subparsers(dest="command")

    # deploy
    p_deploy = sub.add_parser("deploy", help="Deploy endpoint from config")
    p_deploy.add_argument("config", help="YAML config path")

    # smoke-test
    p_test = sub.add_parser("smoke-test", help="Smoke test an endpoint")
    p_test.add_argument("config", help="YAML config path")
    p_test.add_argument("--endpoint", required=True)
    p_test.add_argument("--ic", default=None, help="Inference Component name")

    # cleanup
    p_clean = sub.add_parser("cleanup", help="Delete endpoint and resources")
    p_clean.add_argument("config", help="YAML config path")
    p_clean.add_argument("--endpoint", required=True)

    # list
    p_list = sub.add_parser("list", help="List endpoints")
    p_list.add_argument("--region", default="us-west-2")
    p_list.add_argument("--prefix", default=None)

    args = parser.parse_args()

    if args.command == "deploy":
        cfg = load_config(args.config)
        deploy(cfg)
    elif args.command == "smoke-test":
        cfg = load_config(args.config)
        smoke_test(cfg, args.endpoint, args.ic)
    elif args.command == "cleanup":
        cfg = load_config(args.config)
        pattern = cfg.deployment.endpoint.pattern
        cleanup(args.endpoint, cfg.deployment.endpoint.region, pattern)
    elif args.command == "list":
        list_endpoints(args.region, args.prefix)
    else:
        parser.print_help()
