"""Shared utility: write flattened benchmark result with metrics to central Athena prefix.

v2 — Enriched with self-describing serving context for tranche-independent comparisons.
Each result row now includes top-level scalars for common QuickSight filter axes
and a serving_config JSON blob for full audit trail.
"""
import hashlib
import io
import json
import os
import re
import tarfile
import boto3
import yaml

# Load athena config from benchmarks.yaml if available
_athena_config = {}
for cfg_path in ["../benchmarks.yaml", "/opt/ml/processing/input/script/benchmarks.yaml"]:
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            _athena_config = yaml.safe_load(f).get("athena", {})
        break

CENTRAL_BUCKET = _athena_config.get("bucket") or os.environ.get("ATHENA_BUCKET", "")
CENTRAL_REGION = _athena_config.get("region") or os.environ.get("ATHENA_REGION", "us-east-2")
ATHENA_PREFIX = _athena_config.get("prefix", "athena/results")


def _get_bucket():
    if not CENTRAL_BUCKET:
        raise ValueError("No Athena bucket configured. Set 'athena.bucket' in benchmarks.yaml or ATHENA_BUCKET env var.")
    return CENTRAL_BUCKET


# ─── Serving Context Enrichment ─────────────────────────────────────────────


def _parse_vllm_version(image_uri):
    """Extract vLLM version from DLC image URI.

    Example input:  "763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.20.2-gpu-py312-cu130-ubuntu22.04-sagemaker-v1.2"
    Example output: "0.20.2"

    Returns None if the version cannot be parsed.
    """
    if not image_uri:
        return None
    # Match pattern: /vllm:<version>-  (version is digits + dots)
    m = re.search(r"/vllm:(\d+\.\d+(?:\.\d+)?)", image_uri)
    return m.group(1) if m else None


def _safe_float(value):
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enrich_with_serving_context(record, model_config=None, defaults_config=None, config_source=None):
    """Enrich a benchmark record with self-describing serving context fields.

    Adds top-level scalar columns for common QuickSight filter axes and a
    serving_config JSON blob for full audit trail.

    Args:
        record: The benchmark result dict (modified in-place).
        model_config: The model's config dict from benchmarks.yaml (e.g. models.gemma-4-31b-vllm).
                      If None, enrichment is skipped (backward-compatible).
        defaults_config: The defaults dict (sagemaker_defaults, hyperpod_defaults, or byom_defaults).
                         If None, enrichment is skipped (backward-compatible).
        config_source: Optional filename of the config that produced this record
                       (e.g. "benchmarks.yaml", "benchmarks-hyperpod.yaml"). For audit trail.

    Returns:
        The record dict (same object, modified in-place).
    """
    if not model_config and not defaults_config:
        return record

    model_config = model_config or {}
    defaults_config = defaults_config or {}
    env = model_config.get("env", {})
    vllm_config = defaults_config.get("vllm_config", {})

    # ─── Top-Level Scalars (QuickSight-native filtering) ────────────────────

    # vllm_version: parsed from the DLC image tag
    image_uri = defaults_config.get("sagemaker_image", "")
    # For HyperPod, the image might be in a different field
    if not image_uri:
        image_uri = defaults_config.get("vllm_image", "")
    record["vllm_version"] = _parse_vllm_version(image_uri)

    # kv_cache_dtype: from model env var, default "auto"
    record["kv_cache_dtype"] = env.get("SM_VLLM_KV_CACHE_DTYPE", "auto")

    # quantization: from model env var, default "none"
    record["quantization"] = env.get("SM_VLLM_QUANTIZATION", "none")

    # benchmark_tokenizer: proxy tokenizer if used (null = model's own)
    record["benchmark_tokenizer"] = model_config.get("benchmark_tokenizer", None)

    # ─── JSON Blob (full audit trail) ───────────────────────────────────────

    serving_config = {
        "max_model_len": vllm_config.get("max_model_len"),
        "gpu_memory_utilization": vllm_config.get("gpu_memory_utilization"),
        "instance_type": record.get("instance_type"),
        "moe_backend": env.get("SM_VLLM_MOE_BACKEND"),
        "speculative_config": env.get("SM_VLLM_SPECULATIVE_MODEL"),
        "dlc_image": image_uri if image_uri else None,
        "env": env if env else None,
        "inference_ami_version": defaults_config.get("inference_ami_version"),
        "training_plan_arn": defaults_config.get("ml_reservation_arn"),
        "model_name": model_config.get("model_name"),
        "num_gpus": model_config.get("num_gpus"),
        "s3_model_uri": model_config.get("s3_model_uri"),
        "config_source": config_source,
    }

    # Compute config_hash: deterministic fingerprint of everything that affects
    # benchmark results. If ANY performance-impacting parameter changes, the hash
    # changes. Non-performance fields (config_source, training_plan_arn) are excluded.
    #
    # Rule: if it can change the numbers, it's in the hash.
    #       if it can't (billing, filenames), it's excluded.
    _NON_PERF_FIELDS = {"config_source", "training_plan_arn", "config_hash"}
    hash_input = json.dumps(
        {k: v for k, v in sorted(serving_config.items()) if k not in _NON_PERF_FIELDS},
        sort_keys=True
    )
    serving_config["config_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # Remove None values to keep the blob compact
    serving_config = {k: v for k, v in serving_config.items() if v is not None}
    record["serving_config"] = json.dumps(serving_config, sort_keys=True)

    return record


def _compute_error_rate(aiperf):
    """Compute error rate from AIPerf metrics JSON.

    Uses osl_mismatch_count and request_count if available.
    Returns float (0.0 to 1.0) or None if not computable.
    """
    if not aiperf:
        return None

    request_count = aiperf.get("request_count", {}).get("avg")
    osl_mismatch = aiperf.get("osl_mismatch_count", {}).get("avg")

    if request_count and request_count > 0 and osl_mismatch is not None:
        return round(osl_mismatch / request_count, 6)

    # Fallback: check for error_count field
    error_count = aiperf.get("error_count", {}).get("avg")
    if request_count and request_count > 0 and error_count is not None:
        return round(error_count / request_count, 6)

    return None


def _compute_job_id(record):
    """Compute a deterministic job_id that uniquely identifies a benchmark run.

    Same model + workload + concurrency + config = same job_id = same S3 key.
    Re-runs overwrite the previous record. No duplicates.
    """
    unique_factors = json.dumps({
        "concurrency": record.get("concurrency"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "dataset": record.get("dataset"),
        "serving_config": record.get("serving_config", ""),
    }, sort_keys=True)
    run_hash = hashlib.sha256(unique_factors.encode()).hexdigest()[:8]

    model_key = record.get("model_key", "unknown")
    workload = record.get("workload", "unknown")
    concurrency = record.get("concurrency", 0)

    return f"{model_key}--{workload}--c{concurrency}--{run_hash}"


# ─── Metrics Extraction ─────────────────────────────────────────────────────


def extract_metrics_from_s3(s3_output, source_region):
    """Download the results tarball from S3 and extract key metrics."""
    s3 = boto3.client("s3", region_name=source_region)
    # Find the tarball in the output location
    bucket = s3_output.replace("s3://", "").split("/")[0]
    prefix = "/".join(s3_output.replace("s3://", "").split("/")[1:]).rstrip("/") + "/"

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        tar_key = None
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".tar.gz"):
                tar_key = obj["Key"]
                break
        if not tar_key:
            return {}

        # Download and extract
        tar_obj = s3.get_object(Bucket=bucket, Key=tar_key)
        tar_bytes = io.BytesIO(tar_obj["Body"].read())
        with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith("profile_export_aiperf.json"):
                    f = tf.extractfile(member)
                    if f:
                        return json.loads(f.read().decode())
    except Exception:
        pass
    return {}


def flatten_metrics(aiperf):
    """Extract the key metrics into a flat dict for Athena."""
    if not aiperf:
        return {}
    m = {}

    # Throughput
    m["request_throughput_rps"] = aiperf.get("request_throughput", {}).get("avg")
    m["total_token_throughput_tps"] = aiperf.get("total_token_throughput", {}).get("avg")
    m["output_token_throughput_tps"] = aiperf.get("output_token_throughput", {}).get("avg")
    m["request_count"] = aiperf.get("request_count", {}).get("avg")

    # TTFT
    ttft = aiperf.get("time_to_first_token", {})
    m["ttft_avg_ms"] = ttft.get("avg")
    m["ttft_p50_ms"] = ttft.get("p50")
    m["ttft_p90_ms"] = ttft.get("p90")
    m["ttft_p99_ms"] = ttft.get("p99")

    # ITL
    itl = aiperf.get("inter_token_latency", {})
    m["itl_avg_ms"] = itl.get("avg")
    m["itl_p50_ms"] = itl.get("p50")
    m["itl_p90_ms"] = itl.get("p90")
    m["itl_p99_ms"] = itl.get("p99")

    # E2E latency
    e2e = aiperf.get("request_latency", {})
    m["e2e_latency_avg_ms"] = e2e.get("avg")
    m["e2e_latency_p50_ms"] = e2e.get("p50")
    m["e2e_latency_p90_ms"] = e2e.get("p90")
    m["e2e_latency_p99_ms"] = e2e.get("p99")

    # Prefill throughput
    prefill = aiperf.get("prefill_throughput_per_user", {})
    m["prefill_tps_avg"] = prefill.get("avg")
    m["prefill_tps_p50"] = prefill.get("p50")

    # E2E output token throughput (user-perceived, includes TTFT)
    e2e_otp = aiperf.get("e2e_output_token_throughput", {})
    m["e2e_output_token_tps_avg"] = e2e_otp.get("avg")
    m["e2e_output_token_tps_p50"] = e2e_otp.get("p50")
    m["e2e_output_token_tps_p90"] = e2e_otp.get("p90")

    # Time to second token (decode startup)
    ttst = aiperf.get("time_to_second_token", {})
    m["ttst_p50_ms"] = ttst.get("p50")
    m["ttst_p90_ms"] = ttst.get("p90")

    # Sequence lengths (actual vs requested)
    m["output_sequence_length_avg"] = aiperf.get("output_sequence_length", {}).get("avg")
    m["input_sequence_length_avg"] = aiperf.get("input_sequence_length", {}).get("avg")

    # Error/quality metrics
    m["error_request_count"] = aiperf.get("error_request_count", {}).get("avg")
    m["osl_mismatch_count"] = aiperf.get("osl_mismatch_count", {}).get("avg")
    m["benchmark_duration_sec"] = aiperf.get("benchmark_duration", {}).get("avg")

    # Metadata
    m["aiperf_version"] = aiperf.get("aiperf_version")

    # Dataset — extracted from input_config
    input_config = aiperf.get("input_config", {})
    aiperf_dataset = input_config.get("input", {}).get("public_dataset")
    if aiperf_dataset:
        m["dataset"] = aiperf_dataset  # AIPerf's actual dataset takes precedence (runtime truth)

    # Raw JSON blob — full AIPerf output for ad-hoc Athena queries
    # Exclude large objects (input_config contains resolved prompts metadata,
    # telemetry_data and error_summary are useful to keep)
    raw_blob = {k: v for k, v in aiperf.items() if k != "input_config"}
    m["raw_aiperf_json"] = json.dumps(raw_blob)

    return m


# ─── Write to Athena ────────────────────────────────────────────────────────


def write_athena_record(record, model_config=None, defaults_config=None, config_source=None):
    """Write a flattened benchmark record with metrics to the central Athena prefix.

    Args:
        record: Dict with benchmark metadata (job_id, environment, model_key, etc.).
        model_config: Optional model config dict from benchmarks.yaml for serving context enrichment.
        defaults_config: Optional defaults dict (sagemaker_defaults, etc.) for serving context enrichment.

    If model_config/defaults_config are not provided, the record is written without
    serving context enrichment (backward-compatible with existing call sites).
    """
    s3 = boto3.client("s3", region_name=CENTRAL_REGION)

    # Extract metrics from the results tarball
    s3_output = record.get("s3_output", "")
    source_region = record.get("source_region", CENTRAL_REGION)
    aiperf = {}
    if s3_output:
        aiperf = extract_metrics_from_s3(s3_output, source_region)
        record.update(flatten_metrics(aiperf))

    # Enrich with serving context (if config provided)
    enrich_with_serving_context(record, model_config, defaults_config, config_source=config_source)

    # Compute error rate from AIPerf data
    error_rate = _compute_error_rate(aiperf)
    if error_rate is not None:
        record["error_rate"] = error_rate

    # Compute deterministic job_id (overwrites any caller-provided job_id)
    record["job_id"] = _compute_job_id(record)

    # Strip carrier keys that shouldn't be persisted
    record.pop("model_config", None)
    record.pop("defaults_config", None)

    # Partition key: environment/model/workload
    env = record.get("environment", "unknown")
    model = record.get("model_key", "unknown")
    workload = record.get("workload", "unknown")
    key = f"{ATHENA_PREFIX}/environment={env}/model={model}/workload={workload}/{record['job_id']}.json"

    try:
        s3.put_object(
            Bucket=_get_bucket(),
            Key=key,
            Body=json.dumps(record) + "\n",
            ContentType="application/json",
        )
    except Exception:
        pass  # non-fatal — don't break the benchmark run


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract metrics from benchmark results and write to Athena.")
    parser.add_argument("s3_output", help="S3 location of benchmark results (e.g. s3://bucket/path/)")
    parser.add_argument("--environment", default="managed-inference")
    parser.add_argument("--model", required=True, help="Model key")
    parser.add_argument("--workload", required=True, help="Workload key")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--region", default=CENTRAL_REGION, help="Source region of the results")
    parser.add_argument("--config", default="../benchmarks.yaml", help="Path to benchmarks.yaml for serving context")
    args = parser.parse_args()

    # Load config for enrichment
    model_config = None
    defaults_config = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_config = cfg.get("models", {}).get(args.model)
        env_defaults_key = {"managed-inference": "sagemaker_defaults", "hyperpod": "hyperpod_defaults", "byom": "byom_defaults"}
        defaults_config = cfg.get(env_defaults_key.get(args.environment, "sagemaker_defaults"), {})

    record = {
        "job_id": f"{args.model}--{args.workload}--c{args.concurrency}",
        "environment": args.environment,
        "model_key": args.model,
        "workload": args.workload,
        "concurrency": args.concurrency,
        "s3_output": args.s3_output,
        "source_region": args.region,
    }
    write_athena_record(record, model_config=model_config, defaults_config=defaults_config)
    aiperf = extract_metrics_from_s3(args.s3_output, args.region)
    metrics = flatten_metrics(aiperf)
    if metrics:
        print(f"✓ Written to s3://{_get_bucket()}/{ATHENA_PREFIX}/")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print("✗ No metrics found in tarball")
