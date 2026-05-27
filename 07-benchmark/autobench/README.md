# SageMaker GenAI Inference Benchmarking

Configuration-driven LLM inference benchmarking across SageMaker Managed Inference, HyperPod EKS, and Bedrock BYOM. One config file drives all execution paths.

## Architecture

```
benchmarks.yaml (single source of truth)
       │
       ├── sdk/download_models.py         → Stage weights from HuggingFace to S3
       ├── sdk/benchmark.py               → SageMaker Managed Inference (FTP/reserved capacity)
       ├── sdk/benchmark_hyperpod.py      → HyperPod EKS (kubectl + direct-URL)
       └── sdk/benchmark_bedrock_byom.py  → Bedrock BYOM (Mantle API + RU reservations)
```

All paths use **SageMaker AI Benchmarking (NVIDIA AIPerf)** as the load generator.

## Quick Start

```bash
cd sdk
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 1. Copy and configure
cp benchmarks.yaml.example benchmarks.yaml
# Edit benchmarks.yaml with your account ID, region, role ARN, etc.

# 2. Stage model weights to S3
python download_models.py --submit --region us-east-2 --model=gemma

# 3. Update s3_model_uri in benchmarks.yaml with the printed URI

# 4. Run benchmarks
python benchmark.py --model=gemma              # SageMaker MI
python benchmark_hyperpod.py --model=gemma     # HyperPod EKS
python benchmark_bedrock_byom.py --model=gemma # Bedrock BYOM
```

## Execution Paths

| Path | Script | Deploy Method | Compute |
|------|--------|---------------|---------|
| **SageMaker MI** | `benchmark.py` | `create_model` → endpoint | FTP reserved capacity |
| **HyperPod EKS** | `benchmark_hyperpod.py` | `kubectl apply` → NLB | FTP node group |
| **Bedrock BYOM** | `benchmark_bedrock_byom.py` | Mantle API import + RU reservation | AWS-managed |

## Configuration

All configuration lives in `benchmarks.yaml`. See [`benchmarks.yaml.example`](benchmarks.yaml.example) for a complete template with all features documented.

### Key Sections

| Section | Purpose |
|---------|---------|
| `athena` | Central results bucket + region |
| `workloads` | Business use cases → benchmark parameters |
| `models` | Model definitions shared across all paths |
| `sagemaker_defaults` + `sagemaker_benchmarks` | SMAI-specific config |
| `hyperpod_defaults` + `hyperpod_benchmarks` | HyperPod-specific config |
| `byom_defaults` + `byom_benchmarks` | Bedrock BYOM-specific config |

### Model Config Structure

Models are defined once and shared across all three paths:

```yaml
models:
  my-model-vllm:
    model_name: org/Model-Name        # HuggingFace ID
    instance_type: ml.p6-b200.48xlarge
    num_gpus: 8
    s3_model_uri: "s3://..."          # Pre-staged weights
    env:                               # SMAI: SM_VLLM_* env vars
      SM_VLLM_KV_CACHE_DTYPE: "fp8"
    hyperpod_args:                     # HyperPod: vLLM CLI args
      - "--kv-cache-dtype"
      - "fp8"
    hyperpod_env:                      # HyperPod: pod env vars
      VLLM_USE_FLASHINFER_MOE_FP4: "1"
    byom:                              # Bedrock BYOM: import config
      base_model_id: "provider.model"
      model_id: ""                     # filled after import
```

## Results Pipeline

Results flow to a central Athena table for cross-model, cross-platform analysis:

```
S3 (AIPerf tar.gz) → athena_writer.py (extract + enrich) → S3 (Hive-partitioned JSON) → Athena → QuickSight
```

Each result row is **self-describing** — includes serving config, vLLM version, dataset, error rate, and a full raw AIPerf JSON blob for ad-hoc queries.

### Athena Setup

```bash
# Create the table (run in Athena console)
# See sdk/athena_ddl.sql for the full DDL

# After benchmarks complete, discover new partitions:
MSCK REPAIR TABLE benchmarking.benchmark_metrics;
```

### Backfill

Re-process existing results with the latest schema:

```bash
python backfill.py --environment=managed-inference
python backfill.py --environment=hyperpod --model=gemma-4-31b-vllm
```

## CLI Reference

All scripts support `--help` for full usage. Common flags:

```bash
--validate          # Show expanded job matrix (no AWS calls)
--model=<substr>    # Filter by model key (substring match)
--workload=<substr> # Filter by workload key (substring match)
--submit            # Submit as unattended Processing Job (5-day timeout)
--deploy-only       # Deploy endpoints/pods only (no benchmark)
--benchmark-only    # Benchmark existing endpoints/pods
--cleanup           # Delete deployed resources
```

## Infrastructure

For automated HyperPod cluster setup with pre-configured networking (security groups, NLB subnet tags, IRSA), see [`infra/hyperpod-cluster-bootstrap.yaml`](infra/hyperpod-cluster-bootstrap.yaml).

## Key Features

- **Resume-safe** — re-run safely; completed jobs are skipped automatically
- **Gap capture** — failures classified into actionable categories
- **Set-and-forget** — `--submit` runs as Processing Job (5-day timeout, no credential expiry)
- **Single config** — `benchmarks.yaml` drives all paths
- **S3-first model loading** — pre-stage weights to avoid deploy timeouts
- **Centralized results** — athena_writer pushes all results to one region
- **Self-describing records** — every Athena row includes full serving context + config hash
- **Deterministic deduplication** — re-runs overwrite previous results (no duplicates)

## Extending

### Add a New Model

1. Add to `models` in `benchmarks.yaml`
2. Add to the appropriate `*_benchmarks` section
3. Stage weights: `python download_models.py --submit --region <region> --model=<key>`
4. Run: `python benchmark.py --model=<key>`

### Add a New Workload

Add to `workloads` in `benchmarks.yaml`, then reference in benchmark entries.

### Change vLLM Version

Update `sagemaker_image` in `sagemaker_defaults` (SMAI) and/or `vllm_image` in `hyperpod_defaults` (HyperPod).
