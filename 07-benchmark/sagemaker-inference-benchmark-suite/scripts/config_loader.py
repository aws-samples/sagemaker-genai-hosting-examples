"""
Config loader for SageMaker Inference Benchmark Suite.

Loads YAML recipe files and provides:
- Config validation (container/GPU compatibility, required fields)
- Container image URI construction (vllm-dlc, djl-lmi, byoc)
- Environment variable construction (SM_VLLM_* vs OPTION_*)
- Endpoint name generation
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import yaml


# --- Container image URI patterns ---

DLC_ACCOUNT = "763104351884"

CONTAINER_URI_TEMPLATES = {
    # Legacy private ECR format (pre-0.18)
    "vllm-dlc": "{account}.dkr.ecr.{region}.amazonaws.com/vllm:{version}-gpu-py312-{cuda}-ubuntu22.04-sagemaker",
    # Public ECR format (0.18+)
    "vllm-dlc-public": "public.ecr.aws/deep-learning-containers/vllm:{version}-gpu-py312",
    "djl-lmi": "{account}.dkr.ecr.{region}.amazonaws.com/djl-inference:{version}",
}

# Known DJL LMI version mappings (short name → full tag)
# Source: 763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference
DJL_VERSION_MAP = {
    "v10": "0.28.0-lmi10.0.0-cu124",
    "v11": "0.29.0-lmi11.0.0-cu124",
    "v12": "0.30.0-lmi12.0.0-cu124",
    "v13": "0.31.0-lmi13.0.0-cu124",
    "v14": "0.32.0-lmi14.0.0-cu126",
    "v15": "0.33.0-lmi15.0.0-cu128",
    "v16": "0.34.0-lmi16.0.0-cu128",
    "v17": "0.35.0-lmi17.0.0-cu128",
    "v18": "0.36.0-lmi18.0.0-cu128",
    "v19": "0.36.0-lmi19.0.0-cu128",
    "v20": "0.36.0-lmi20.0.0-cu128-v1.0",
    "v21": "0.36.0-lmi21.0.0-cu129-v1.0",
    "v22": "0.36.0-lmi22.0.0-cu129-v1.0",
    "v23": "0.36.0-lmi23.0.0-cu129-v1.0",
}

# Container/GPU compatibility rules
COMPATIBILITY_RULES = [
    # (container_type, cuda, instance_pattern, allowed, reason)
    ("vllm-dlc", "cu129", r"ml\.g5\.", False, "CUDA 12.9 fails on Ampere (g5). Use cu128."),
    ("vllm-dlc", "cu128", r"ml\.g7e\.", False, "Blackwell (g7e) requires CUDA 12.9+. Use cu129."),
    ("vllm-dlc", "cu128", r"ml\.g6e\.", False, "Ada Lovelace (g6e) requires CUDA 12.9+. Use cu129."),
    ("djl-lmi", None, r"ml\.g7e\.", False, "DJL LMI ships CUDA 12.8, incompatible with Blackwell (g7e)."),
]


# --- Dataclasses ---

@dataclass
class ContainerConfig:
    type: str = "vllm-dlc"  # vllm-dlc | djl-lmi | byoc
    version: str = "0.15.1"
    cuda: str = "cu129"
    image_uri: Optional[str] = None  # Override URI for any container type
    public_ecr: bool = False  # Use public.ecr.aws format (vLLM 0.18+)

@dataclass
class VllmConfig:
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: Optional[int] = None
    enable_chunked_prefill: bool = True
    swap_space: Optional[int] = None
    trust_remote_code: bool = False
    extra_env: dict = field(default_factory=dict)

@dataclass
class SpeculativeDecodingConfig:
    enabled: bool = False
    method: str = "eagle3"
    model: Optional[str] = None
    num_speculative_tokens: int = 4

@dataclass
class PrefixCachingConfig:
    enabled: bool = False

@dataclass
class ICConfig:
    num_accelerators: int = 8
    min_memory_mb: int = 10240
    managed_scaling: bool = True

@dataclass
class EndpointConfig:
    pattern: str = "standard"  # standard | inference_component
    name_prefix: str = "bench"
    region: str = "us-west-2"
    role_arn: str = ""
    health_check_timeout: int = 1200
    ic: ICConfig = field(default_factory=ICConfig)

@dataclass
class ModelConfig:
    id: str = ""
    short_name: str = ""
    trust_remote_code: bool = False

@dataclass
class InstanceConfig:
    type: str = "ml.g7e.2xlarge"
    count: int = 1

@dataclass
class HyperPodKVCacheConfig:
    l1_cache: bool = True
    l2_cache: bool = True
    l2_backend: str = "tieredstorage"  # tieredstorage | redis

@dataclass
class HyperPodRoutingConfig:
    enabled: bool = True
    strategy: str = "prefixaware"  # prefixaware | kvaware | roundrobin

@dataclass
class HyperPodModelSourceConfig:
    type: str = "s3"  # s3 | fsx
    s3_bucket: str = ""
    s3_region: str = ""
    model_location: str = ""
    fsx_file_system_id: str = ""

@dataclass
class HyperPodWorkerConfig:
    gpu_count: int = 1
    cpu_request: str = "6"
    memory_request: str = "30Gi"
    image: str = ""  # container image (auto-derived from container config if empty)

@dataclass
class HyperPodConfig:
    cluster_name: str = ""  # EKS cluster name (for kubeconfig)
    namespace: str = "default"
    replicas: int = 1
    invocation_endpoint: str = "v1/chat/completions"
    model_source: HyperPodModelSourceConfig = field(default_factory=HyperPodModelSourceConfig)
    kv_cache: HyperPodKVCacheConfig = field(default_factory=HyperPodKVCacheConfig)
    routing: HyperPodRoutingConfig = field(default_factory=HyperPodRoutingConfig)
    worker: HyperPodWorkerConfig = field(default_factory=HyperPodWorkerConfig)

@dataclass
class DeploymentConfig:
    platform: str = "sagemaker"  # sagemaker | hyperpod
    model: ModelConfig = field(default_factory=ModelConfig)
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    speculative_decoding: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)
    prefix_caching: PrefixCachingConfig = field(default_factory=PrefixCachingConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    hyperpod: HyperPodConfig = field(default_factory=HyperPodConfig)

@dataclass
class InferenceParams:
    max_tokens: int = 600
    temperature: float = 0.7
    extra_payload: dict = field(default_factory=dict)

@dataclass
class BenchmarkParams:
    concurrency_levels: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    requests_per_level: int = 30
    warmup_requests: int = 3
    pause_between_levels_sec: int = 3
    streaming: bool = True
    inference_params: InferenceParams = field(default_factory=InferenceParams)
    use_cases: list = field(default_factory=lambda: ["multiturn_chat", "tool_calling", "long_context"])

@dataclass
class CostConfig:
    instance_cost_per_hour: float = 0.0

@dataclass
class BenchmarkConfig:
    name: str = ""
    description: str = ""
    version: str = "1.0"
    pipeline: list = field(default_factory=lambda: ["deploy", "benchmark", "cleanup"])
    endpoint: str = ""  # existing endpoint name (for benchmark-only recipes)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    benchmark: BenchmarkParams = field(default_factory=BenchmarkParams)
    cost: CostConfig = field(default_factory=CostConfig)


# --- Loading ---

def _merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a nested dataclass."""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()} if hasattr(cls, '__dataclass_fields__') else {}
    kwargs = {}
    for k, v in data.items():
        k_snake = k.replace("-", "_")
        if k_snake in field_types:
            ft = field_types[k_snake]
            # Resolve Optional[X] → X
            if hasattr(ft, '__origin__') and ft.__origin__ is type(None):
                kwargs[k_snake] = v
            elif isinstance(ft, str):
                # String annotation — try to resolve
                resolved = globals().get(ft.replace("Optional[", "").rstrip("]"), None)
                if resolved and hasattr(resolved, '__dataclass_fields__') and isinstance(v, dict):
                    kwargs[k_snake] = _dict_to_dataclass(resolved, v)
                else:
                    kwargs[k_snake] = v
            elif hasattr(ft, '__dataclass_fields__') and isinstance(v, dict):
                kwargs[k_snake] = _dict_to_dataclass(ft, v)
            else:
                kwargs[k_snake] = v
        else:
            kwargs[k_snake] = v
    # Only pass known fields
    known = set(cls.__dataclass_fields__.keys()) if hasattr(cls, '__dataclass_fields__') else set()
    filtered = {k: v for k, v in kwargs.items() if k in known}
    return cls(**filtered)


def load_config(yaml_path: str) -> BenchmarkConfig:
    """Load a YAML recipe file and return a BenchmarkConfig."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Empty config file: {yaml_path}")

    # Build nested dataclass
    config = BenchmarkConfig()

    if "name" in raw:
        config.name = raw["name"]
    if "description" in raw:
        config.description = raw["description"]
    if "version" in raw:
        config.version = str(raw["version"])
    if "pipeline" in raw:
        config.pipeline = raw["pipeline"]
    if "endpoint" in raw:
        config.endpoint = raw["endpoint"]

    if "deployment" in raw:
        d = raw["deployment"]
        config.deployment = DeploymentConfig()
        if "platform" in d:
            config.deployment.platform = d["platform"]
        if "model" in d:
            config.deployment.model = _dict_to_dataclass(ModelConfig, d["model"])
        if "instance" in d:
            config.deployment.instance = _dict_to_dataclass(InstanceConfig, d["instance"])
        if "container" in d:
            config.deployment.container = _dict_to_dataclass(ContainerConfig, d["container"])
        if "vllm" in d:
            config.deployment.vllm = _dict_to_dataclass(VllmConfig, d["vllm"])
        if "speculative_decoding" in d:
            config.deployment.speculative_decoding = _dict_to_dataclass(SpeculativeDecodingConfig, d["speculative_decoding"])
        if "prefix_caching" in d:
            config.deployment.prefix_caching = _dict_to_dataclass(PrefixCachingConfig, d["prefix_caching"])
        if "endpoint" in d:
            ep = d["endpoint"]
            config.deployment.endpoint = _dict_to_dataclass(EndpointConfig, ep)
            if "ic" in ep:
                config.deployment.endpoint.ic = _dict_to_dataclass(ICConfig, ep["ic"])
        if "hyperpod" in d:
            hp = d["hyperpod"]
            config.deployment.hyperpod = _dict_to_dataclass(HyperPodConfig, hp)
            if "model_source" in hp:
                config.deployment.hyperpod.model_source = _dict_to_dataclass(HyperPodModelSourceConfig, hp["model_source"])
            if "kv_cache" in hp:
                config.deployment.hyperpod.kv_cache = _dict_to_dataclass(HyperPodKVCacheConfig, hp["kv_cache"])
            if "routing" in hp:
                config.deployment.hyperpod.routing = _dict_to_dataclass(HyperPodRoutingConfig, hp["routing"])
            if "worker" in hp:
                config.deployment.hyperpod.worker = _dict_to_dataclass(HyperPodWorkerConfig, hp["worker"])

    if "benchmark" in raw:
        b = raw["benchmark"]
        config.benchmark = _dict_to_dataclass(BenchmarkParams, b)
        if "inference_params" in b:
            config.benchmark.inference_params = _dict_to_dataclass(InferenceParams, b["inference_params"])

    if "cost" in raw:
        config.cost = _dict_to_dataclass(CostConfig, raw["cost"])

    return config


# --- Validation ---

def validate_config(config: BenchmarkConfig) -> list[str]:
    """Validate config and return list of warnings. Raises ValueError for fatal issues."""
    warnings = []
    d = config.deployment

    # Required fields
    if not d.model.id:
        raise ValueError("deployment.model.id is required")

    # Platform-specific validation
    if d.platform == "hyperpod":
        if not d.hyperpod.cluster_name:
            raise ValueError("deployment.hyperpod.cluster_name is required for HyperPod")
        if d.hyperpod.model_source.type == "s3" and not d.hyperpod.model_source.s3_bucket:
            raise ValueError("deployment.hyperpod.model_source.s3_bucket is required for S3 source")
    else:
        if not d.endpoint.role_arn:
            raise ValueError("deployment.endpoint.role_arn is required")

    # Auto-derive short_name if not set
    if not d.model.short_name:
        d.model.short_name = d.model.id.split("/")[-1].lower().replace("-", "").replace("_", "")[:20]

    # Container/GPU compatibility (SageMaker DLC-specific, skip for HyperPod)
    inst = d.instance.type
    ct = d.container
    if d.platform != "hyperpod":
        for rule_ct, rule_cuda, inst_pattern, allowed, reason in COMPATIBILITY_RULES:
            if ct.type == rule_ct and re.search(inst_pattern, inst):
                if rule_cuda is None or ct.cuda == rule_cuda:
                    if not allowed:
                        raise ValueError(f"Incompatible config: {reason}")

    # Speculative decoding checks
    if d.speculative_decoding.enabled and not d.speculative_decoding.model:
        raise ValueError("speculative_decoding.enabled=true but no speculator model specified")

    # IC pattern checks
    if d.endpoint.pattern == "inference_component" and "g7e" in inst:
        warnings.append("g7e does not support Inference Component pattern well. Consider using standard pattern.")

    # BYOC requires image_uri (other types can optionally use it as override)
    if ct.type == "byoc" and not ct.image_uri:
        raise ValueError("container.type=byoc requires container.image_uri")

    # Trust remote code propagation
    if d.model.trust_remote_code and ct.type == "vllm-dlc":
        # SM_VLLM_TRUST_REMOTE_CODE may not work on all DLC versions
        warnings.append("trust_remote_code with vLLM DLC: verify SM_VLLM_TRUST_REMOTE_CODE is supported in your DLC version.")

    return warnings


# --- Container Image URI ---

def build_container_uri(config: BenchmarkConfig) -> str:
    """Build the container image URI based on config.

    Priority: image_uri (explicit override) > template construction.
    image_uri works for ANY container type, not just byoc.
    """
    ct = config.deployment.container
    region = config.deployment.endpoint.region

    # Explicit URI override — works for all container types
    if ct.image_uri:
        uri = ct.image_uri
        if "{region}" in uri:
            uri = uri.replace("{region}", region)
        return uri

    if ct.type == "byoc":
        raise ValueError("container.type=byoc requires container.image_uri")

    if ct.type == "vllm-dlc":
        if ct.public_ecr:
            return CONTAINER_URI_TEMPLATES["vllm-dlc-public"].format(version=ct.version)
        return CONTAINER_URI_TEMPLATES["vllm-dlc"].format(
            account=DLC_ACCOUNT, region=region,
            version=ct.version, cuda=ct.cuda,
        )

    if ct.type == "djl-lmi":
        version = DJL_VERSION_MAP.get(ct.version, ct.version)
        return CONTAINER_URI_TEMPLATES["djl-lmi"].format(
            account=DLC_ACCOUNT, region=region, version=version,
        )

    raise ValueError(f"Unknown container type: {ct.type}")


# --- Environment Variables ---

def build_env_vars(config: BenchmarkConfig) -> dict[str, str]:
    """Build environment variables for the SageMaker model container."""
    d = config.deployment
    ct = d.container
    v = d.vllm

    if ct.type == "djl-lmi":
        return _build_env_djl_lmi(config)
    elif ct.type == "byoc":
        return _build_env_byoc(config)
    else:
        return _build_env_vllm_dlc(config)


def _build_env_vllm_dlc(config: BenchmarkConfig) -> dict[str, str]:
    """Build SM_VLLM_* env vars for standard vLLM DLC containers."""
    d = config.deployment
    v = d.vllm

    env = {
        "SM_VLLM_MODEL": d.model.id,
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(v.tensor_parallel_size),
        "SM_VLLM_DTYPE": v.dtype,
        "SM_VLLM_MAX_MODEL_LEN": str(v.max_model_len),
        "SM_VLLM_MAX_NUM_SEQS": str(v.max_num_seqs),
        "SM_VLLM_GPU_MEMORY_UTILIZATION": str(v.gpu_memory_utilization),
    }

    if v.max_num_batched_tokens:
        env["SM_VLLM_MAX_NUM_BATCHED_TOKENS"] = str(v.max_num_batched_tokens)

    # Boolean flags (empty string = flag enabled)
    if v.enable_chunked_prefill:
        env["SM_VLLM_ENABLE_CHUNKED_PREFILL"] = ""

    if d.prefix_caching.enabled:
        env["SM_VLLM_ENABLE_PREFIX_CACHING"] = ""

    if d.model.trust_remote_code or v.trust_remote_code:
        env["SM_VLLM_TRUST_REMOTE_CODE"] = ""

    # KV cache offloading (LMCache)
    if v.swap_space:
        env["SM_VLLM_KV_CACHE_DTYPE"] = "auto"
        env["SM_VLLM_SWAP_SPACE"] = str(v.swap_space)

    # Speculative decoding
    if d.speculative_decoding.enabled:
        sd = d.speculative_decoding
        env["SM_VLLM_SPECULATIVE_CONFIG"] = json.dumps({
            "method": sd.method,
            "model": sd.model,
            "num_speculative_tokens": sd.num_speculative_tokens,
        })

    # Extra env vars
    env.update(v.extra_env)

    return env


def _build_env_djl_lmi(config: BenchmarkConfig) -> dict[str, str]:
    """Build env vars for DJL LMI containers (OPTION_* prefix)."""
    d = config.deployment
    v = d.vllm

    env = {
        "HF_MODEL_ID": d.model.id,
        "SERVING_FAIL_FAST": "true",
        "OPTION_ASYNC_MODE": "true",
        "OPTION_ROLLING_BATCH": "disable",
        "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
        "OPTION_TENSOR_PARALLEL_DEGREE": str(v.tensor_parallel_size),
        "OPTION_MAX_MODEL_LEN": str(v.max_model_len),
        # Use MAX_ROLLING_BATCH_SIZE, NOT MAX_NUM_SEQS (conflicts with DJL defaults)
        "OPTION_MAX_ROLLING_BATCH_SIZE": str(v.max_num_seqs),
        "OPTION_GPU_MEMORY_UTILIZATION": str(v.gpu_memory_utilization),
    }

    if d.model.trust_remote_code or v.trust_remote_code:
        env["OPTION_TRUST_REMOTE_CODE"] = "true"

    if d.prefix_caching.enabled:
        env["OPTION_ENABLE_PREFIX_CACHING"] = "true"

    # Speculative decoding
    if d.speculative_decoding.enabled:
        sd = d.speculative_decoding
        env["OPTION_SPECULATIVE_CONFIG"] = json.dumps({
            "method": sd.method,
            "model": sd.model,
            "num_speculative_tokens": sd.num_speculative_tokens,
        })

    # Tool calling parsers (model-specific)
    if "kimi" in d.model.id.lower():
        env["OPTION_ENABLE_AUTO_TOOL_CHOICE"] = "true"
        env["OPTION_TOOL_CALL_PARSER"] = "kimi_k2"

    env.update(v.extra_env)

    return env


def _build_env_byoc(config: BenchmarkConfig) -> dict[str, str]:
    """Build env vars for BYOC containers (OPTION_* prefix, custom serve script)."""
    d = config.deployment
    v = d.vllm

    env = {
        "OPTION_MODEL": d.model.id,
        "OPTION_SERVED_MODEL_NAME": "model",
        "OPTION_TENSOR_PARALLEL_SIZE": str(v.tensor_parallel_size),
        "OPTION_MAX_MODEL_LEN": str(v.max_model_len),
        "OPTION_MAX_NUM_SEQS": str(v.max_num_seqs),
        "OPTION_GPU_MEMORY_UTILIZATION": str(v.gpu_memory_utilization),
        "OPTION_ASYNC_SCHEDULING": "true",
    }

    if d.prefix_caching.enabled:
        env["OPTION_ENABLE_PREFIX_CACHING"] = ""

    # Speculative decoding — must remove ASYNC_SCHEDULING (incompatible in vLLM 0.10.2)
    if d.speculative_decoding.enabled:
        env.pop("OPTION_ASYNC_SCHEDULING", None)
        sd = d.speculative_decoding
        env["OPTION_SPECULATIVE_CONFIG"] = json.dumps({
            "method": sd.method,
            "model": sd.model,
            "num_speculative_tokens": sd.num_speculative_tokens,
        })

    env.update(v.extra_env)

    return env


# --- Endpoint Naming ---

def build_endpoint_name(config: BenchmarkConfig) -> str:
    """Generate a deterministic endpoint name from config values."""
    d = config.deployment
    prefix = d.endpoint.name_prefix

    model_short = d.model.short_name or d.model.id.split("/")[-1][:15]
    model_short = re.sub(r"[^a-zA-Z0-9]", "-", model_short).strip("-").lower()

    inst_short = d.instance.type.replace("ml.", "").replace(".", "").replace("xlarge", "xl")

    opt_parts = []
    if d.speculative_decoding.enabled:
        opt_parts.append("eagle3")
    if d.prefix_caching.enabled and not d.speculative_decoding.enabled:
        opt_parts.append("prefcache")
    if d.vllm.swap_space:
        opt_parts.append("lmcache")
    if not opt_parts:
        opt_parts.append("vanilla")
    opt = "-".join(opt_parts)

    ts = datetime.now().strftime("%m%d-%H%M")

    name = f"{prefix}-{model_short}-{inst_short}-{opt}-{ts}"
    # SageMaker max 63 chars
    return name[:63]


# --- Utility ---

def get_optimization_label(config: BenchmarkConfig) -> str:
    """Get a human-readable optimization label."""
    d = config.deployment
    if d.speculative_decoding.enabled:
        return "eagle3"
    if d.vllm.swap_space:
        return "lmcache"
    if d.prefix_caching.enabled:
        return "prefix_cache"
    return "vanilla"


def print_config_summary(config: BenchmarkConfig):
    """Print a summary of the config for logging."""
    d = config.deployment
    print(f"  Model: {d.model.id}")
    print(f"  Instance: {d.instance.type} (x{d.instance.count})")
    print(f"  Container: {d.container.type} {d.container.version}")
    print(f"  TP: {d.vllm.tensor_parallel_size}")
    print(f"  Optimization: {get_optimization_label(config)}")
    if d.speculative_decoding.enabled:
        print(f"  Speculator: {d.speculative_decoding.model}")
    print(f"  Region: {d.endpoint.region}")
    print(f"  Pattern: {d.endpoint.pattern}")
