#!/usr/bin/env python3
"""Backfill Athena records from existing S3 tarballs with enriched serving context.

v2 — Loads the benchmarks.yaml used for each run (from processing-configs/ in S3)
to populate serving context fields (vllm_version, quantization, kv_cache_dtype, etc.).

Falls back to local benchmarks.yaml if the run-specific config isn't available in S3
(e.g., for locally-executed runs that didn't use --submit).
"""
import argparse
import io
import re
import boto3
import yaml
from athena_writer import write_athena_record


def _load_config_from_s3(s3_client, bucket, job_name):
    """Try to load benchmarks.yaml from processing-configs/{job_name}/ in S3.

    Returns parsed config dict, or None if not found.
    """
    key = f"processing-configs/{job_name}/benchmarks.yaml"
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        content = resp["Body"].read().decode("utf-8")
        return yaml.safe_load(content)
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception:
        return None


def _infer_job_name_from_path(model_key, workload, concurrency, timestamp):
    """Infer the Processing Job name from the S3 path components.

    Processing jobs are typically named like: bench-{model_key}-{workload}-c{N}
    or: bmk-prod-{model_key}-{workload}-c{N}-{timestamp}

    This is a best-effort heuristic — if it doesn't match, we'll fall back to local config.
    """
    # Common patterns used by benchmark.py --submit
    candidates = [
        f"bench-{model_key}-{workload}-c{concurrency}",
        f"bmk-prod-{model_key}-{workload}-c{concurrency}",
        f"bmk-{model_key}-{workload}-c{concurrency}",
    ]
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Athena metrics from S3 tarballs with enriched serving context."
    )
    parser.add_argument("--model", help="Model key (e.g. gemma-4-31b-vllm). If omitted, backfills all models in config.")
    parser.add_argument(
        "--environment", required=True,
        choices=["managed-inference", "hyperpod", "byom"],
        help="Benchmark environment"
    )
    parser.add_argument("--config", default="../benchmarks.yaml",
                        help="Path to benchmarks.yaml")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without writing")
    parser.add_argument("--since", default=None,
                        help="Only process tarballs with timestamp >= this value (format: 20260527 or 20260527T100000)")
    parser.add_argument("--latest-only", action="store_true",
                        help="For each (workload, concurrency) combo, only process the most recent tarball (skips older/failed runs)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        local_config = yaml.safe_load(f)

    # Determine which models to backfill
    if args.model:
        model_keys = [args.model]
    else:
        benchmarks_key = {
            "managed-inference": "sagemaker_benchmarks",
            "hyperpod": "hyperpod_benchmarks",
            "byom": "byom_benchmarks",
        }[args.environment]
        entries = local_config.get(benchmarks_key, [])
        model_keys = list(dict.fromkeys(e["model"] for e in entries))
        print(f"Backfilling {len(model_keys)} models from {benchmarks_key}\n")

    for model_key in model_keys:
        print(f"\n{'─'*60}")
        print(f"[backfill] {model_key}")
        print(f"{'─'*60}")
        _backfill_model(model_key, args.environment, args.region, args.dry_run, args.config, local_config, since=args.since, latest_only=args.latest_only)

def _backfill_model(model_key, environment, region, dry_run, config_path, local_config, since=None, latest_only=False):
    """Backfill a single model's results."""
    models_key = "byom_models" if environment == "byom" else "models"
    model_cfg = local_config.get(models_key, {}).get(model_key)
    if not model_cfg:
        print(f"  ⚠️  Model '{model_key}' not in config, skipping")
        return

    defaults_key = {
        "managed-inference": "sagemaker_defaults",
        "hyperpod": "hyperpod_defaults",
        "byom": "byom_defaults"
    }[environment]
    local_defaults = local_config.get(defaults_key, {})
    s3_output_template = local_defaults.get("s3_output", "").format(region=region)
    bucket = s3_output_template.replace("s3://", "").split("/")[0]
    prefix_base = "/".join(s3_output_template.replace("s3://", "").split("/")[1:])
    prefix = f"{prefix_base}/{model_key}/"

    # Get workload configs for token counts
    workloads = local_config.get("workloads", {})

    s3 = boto3.client("s3", region_name=region)

    # List all tarballs for this model
    paginator = s3.get_paginator("list_objects_v2")
    tarballs = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar.gz") and obj["Size"] >= 1000:
                tarballs.append(obj["Key"])

    if not tarballs:
        print(f"No tarballs found under s3://{bucket}/{prefix}")
        return

    print(f"Found {len(tarballs)} tarballs for {model_key}")

    # --since filter: drop tarballs older than the specified timestamp
    if since:
        filtered = []
        for key in tarballs:
            parts = key.split("/")
            try:
                model_idx = parts.index(model_key)
                ts = parts[model_idx + 3] if len(parts) > model_idx + 3 else ""
            except (ValueError, IndexError):
                ts = ""
            if ts >= since:
                filtered.append(key)
        print(f"  --since {since}: {len(tarballs)} → {len(filtered)} tarballs")
        tarballs = filtered

    # --latest-only filter: for each (workload, concurrency) combo, keep only the newest tarball
    if latest_only:
        from collections import defaultdict
        groups = defaultdict(list)
        for key in tarballs:
            parts = key.split("/")
            try:
                model_idx = parts.index(model_key)
                wl = parts[model_idx + 1]
                conc_part = parts[model_idx + 2]
                ts = parts[model_idx + 3] if len(parts) > model_idx + 3 else ""
                groups[(wl, conc_part)].append((ts, key))
            except (ValueError, IndexError):
                groups[("unknown", "unknown")].append(("", key))
        # Keep only the latest (max timestamp) per group
        tarballs = [max(entries, key=lambda x: x[0])[1] for entries in groups.values()]
        print(f"  --latest-only: keeping {len(tarballs)} latest tarballs (1 per workload×concurrency)")

    # Track config loading (cache across jobs with same config)
    s3_config_cache = {}
    count = 0
    skipped = 0

    for key in tarballs:
        parts = key.split("/")

        # Parse: {prefix_base}/{model}/{workload}/p6-b200-c{N}/{timestamp}/...
        try:
            model_idx = parts.index(model_key)
        except ValueError:
            skipped += 1
            continue

        workload = parts[model_idx + 1]
        m = re.search(r"-c(\d+)", parts[model_idx + 2])
        concurrency = int(m.group(1)) if m else 1
        timestamp = parts[model_idx + 3] if len(parts) > model_idx + 3 else "unknown"

        wl_cfg = workloads.get(workload, {})
        s3_path = f"s3://{bucket}/{'/'.join(parts[:model_idx + 4])}/"

        # Try to load run-specific config from S3
        job_candidates = _infer_job_name_from_path(model_key, workload, concurrency, timestamp)
        run_config = None
        for candidate in job_candidates:
            if candidate in s3_config_cache:
                run_config = s3_config_cache[candidate]
                break
            loaded = _load_config_from_s3(s3, bucket, candidate)
            s3_config_cache[candidate] = loaded
            if loaded:
                run_config = loaded
                break

        # Resolve model_config and defaults_config
        if run_config:
            effective_model_cfg = run_config.get("models", {}).get(model_key, model_cfg)
            effective_defaults = run_config.get(defaults_key, local_defaults)
            config_source = "s3"
        else:
            effective_model_cfg = model_cfg
            effective_defaults = local_defaults
            config_source = "local (fallback)"

        record = {
            "job_id": f"{model_key}--{workload}--c{concurrency}-{timestamp}",
            "environment": environment,
            "model_key": model_key,
            "model_name": effective_model_cfg.get("model_name", model_cfg["model_name"]),
            "workload": workload,
            "concurrency": concurrency,
            "input_tokens": wl_cfg.get("input_tokens", 1000),
            "output_tokens": wl_cfg.get("output_tokens", 500),
            "streaming": wl_cfg.get("streaming", True),
            "duration": wl_cfg.get("duration", 300),
            "warmup": wl_cfg.get("warmup", 30),
            "dataset": wl_cfg.get("dataset"),
            "instance_type": effective_model_cfg.get("instance_type", model_cfg.get("instance_type", "")),
            "num_gpus": effective_model_cfg.get("num_gpus", model_cfg.get("num_gpus", 8)),
            "source_region": region,
            "s3_output": s3_path,
            "timestamp": timestamp,
        }

        if dry_run:
            from athena_writer import enrich_with_serving_context, _parse_vllm_version
            enrich_with_serving_context(record, effective_model_cfg, effective_defaults)
            vllm_v = record.get("vllm_version", "?")
            print(f"  [DRY-RUN] {workload} c{concurrency} | vllm={vllm_v} | config={config_source}")
        else:
            write_athena_record(record, model_config=effective_model_cfg, defaults_config=effective_defaults)
            count += 1
            vllm_v = "?"  # Will be set by write_athena_record internally
            print(f"  ✓ {workload} c{concurrency} ({timestamp}) [config: {config_source}]")

    print(f"\n  Backfilled {count} records for {model_key} ({skipped} skipped)")
    if dry_run:
        print("  (Dry run — nothing was written)")


if __name__ == "__main__":
    main()
