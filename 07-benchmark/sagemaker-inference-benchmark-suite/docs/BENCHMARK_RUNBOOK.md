# SageMaker Inference Benchmark Runbook

Step-by-step guide for running LLM inference benchmarks on Amazon SageMaker.

---

## 1. Prerequisites

### AWS Setup
- AWS account with SageMaker access
- IAM role: `arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole`
- GPU instance quota (request via Service Quotas console):
  - `ml.g7e.2xlarge` — Blackwell (best cost/token)
  - `ml.g6e.12xlarge` — Ada Lovelace
  - `ml.g5.12xlarge` — Ampere
  - `ml.p5e.48xlarge` — Large models (120B+)

### Local Setup
```bash
pip install -r requirements.txt
```

### Repository Structure
```
sagemaker-lmcache-benchmark/
├── run.py                  # CLI entry point
├── requirements.txt        # Python dependencies
├── recipes/                # 31 YAML recipes + template
├── scripts/                # Core modules
├── docs/                   # This runbook + methodology
└── results/                # CSV results + reports (gitignored)
```

---

## 2. Quick Start

```bash
# Validate recipe (no AWS calls)
python run.py -f recipes/qwen3-32b-g7e-eagle3.yaml --dry-run

# Full pipeline: deploy → benchmark → cleanup
python run.py -f recipes/qwen3-32b-g7e-eagle3.yaml

# Benchmark existing endpoint
python run.py -f recipes/recipe.yaml --only benchmark --endpoint NAME

# Delete when done
python run.py --cleanup --endpoint NAME
```

---

## 3. Step-by-Step Guide

### Step 1: Choose a Recipe

Browse [`recipes/`](../recipes/) for 31 pre-tested recipes, or create your own:

```bash
cp recipes/template.yaml recipes/my-model.yaml
# Edit model ID, instance type, container, pipeline
python run.py -f recipes/my-model.yaml --dry-run
```

**Instance selection by model size:**

| Model Size | Recommended Instance | $/hr |
|------------|---------------------|------|
| < 40B (dense/MoE) | ml.g7e.2xlarge | $4.20 |
| 40-100B (MoE) | ml.g6e.12xlarge | $13.12 |
| 100B+ (MoE) | ml.p5e.48xlarge | ~$80 |

**Container selection by GPU:**

| GPU Architecture | Container | CUDA |
|-----------------|-----------|------|
| Blackwell (g7e) | vLLM 0.15.1+ | cu129 |
| Ada (g6e) | vLLM 0.15.1+ | cu129 |
| Ampere (g5) | vLLM 0.11.0 | cu128 |
| Hopper (p5e) | vLLM 0.17+ or DJL LMI v20 | cu129/cu128 |

### Step 2: Run

```bash
# Full pipeline (deploy → benchmark → cleanup, defined in YAML)
python run.py -f recipes/qwen3-32b-g7e-eagle3.yaml

# Deploy only
python run.py -f recipes/recipe.yaml --only deploy

# Benchmark + cleanup (existing endpoint)
python run.py -f recipes/recipe.yaml --only benchmark,cleanup --endpoint NAME

# Quick test (override params)
python run.py -f recipes/recipe.yaml --only benchmark --endpoint NAME \
  --use-case multiturn_chat --concurrency 1,4 --requests 5

# Streaming TTFT measurement
python run.py -f recipes/recipe.yaml --only benchmark --endpoint NAME --streaming
```

**Expected times:**
- Deploy (g7e/g6e/g5): 10-20 min
- Deploy (p5e): 30-60 min
- Benchmark (3 use cases × 6 levels × 30 req): 20-40 min

### Step 3: Report & Cleanup

```bash
python run.py --report --cost ml.g7e.2xlarge=4.20 ml.p5e.48xlarge=80.00
python run.py --status --region us-west-2
python run.py --cleanup --endpoint NAME
```

---

## 4. Adding a New Model

1. Copy template: `cp recipes/template.yaml recipes/<model>-<instance>-<opt>.yaml`
2. Fill in: model ID, instance type, container, vLLM params
3. Check EAGLE3 speculator: search HuggingFace for `RedHatAI/<model>-speculator.eagle3`
4. Set pipeline: `[deploy, benchmark, cleanup]`
5. Validate: `python run.py -f recipes/your-recipe.yaml --dry-run`
6. Run: `python run.py -f recipes/your-recipe.yaml`

---

## 5. Understanding the Results

### CSV Output

Each benchmark run produces CSV files in `results/matrix/`:

| Column | Description |
|--------|-------------|
| `concurrency` | Simultaneous requests (1-32) |
| `ttft_p50/p90/avg` | Time To First Token percentiles (ms) — streaming |
| `latency_p50/p90/p99` | End-to-end latency percentiles (ms) |
| `tok_per_sec_avg` | Average per-request output token speed |
| `rps` | Requests per second = `concurrency / avg_latency_sec` |
| `aggregate_output_tok_sec` | Total output tokens/sec = `RPS × avg_output_tokens` |
| `avg_input_tokens` | Average prompt tokens (from vLLM `usage`) |
| `avg_output_tokens` | Average completion tokens (from vLLM `usage`) |
| `output_validation_rate` | Fraction of responses with >10 chars |

### Key Metrics

- **TTFT**: Time To First Token — critical for interactive UX. Measured via SSE streaming.
- **Aggregate tok/s**: True throughput at a given concurrency. The capacity metric.
- **RPS**: Requests per second. For capacity planning: `concurrency / avg_latency_sec`.
- **$/M output tokens**: Self-hosted cost efficiency. `($/hr) / (agg_tok_sec × 3600) × 1M`.
  Measured at peak concurrency. No hardcoded assumptions — cost comes from recipe YAML.

### What's "Good"?

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|-----------|
| C=1 tok/s (32B dense) | >40 | 20-40 | 10-20 |
| C=32 RPS | >3 | 1-3 | 0.5-1 |
| Latency degradation C=1→C=32 | <50% | 50-100% | 100-200% |
| $/M tokens (g7e) | <$1 | $1-3 | $3-10 |

---

## 6. Troubleshooting

### Deployment Failures

| Error | Fix |
|-------|-----|
| `InsufficientInstanceCapacity` | Retry later or try different region |
| `does not recognize this architecture` | Use newer vLLM version (0.17+) |
| `trust_remote_code` | Set `trust_remote_code: true` in recipe |
| `ping health check failed` | Increase `health_check_timeout` to 3600+ |
| `EAGLE3 not supported for model_type` | Model not in vLLM EAGLE3 whitelist |
| `CUDA error` | g5 needs cu128, g7e/g6e needs cu129 |

### Benchmark Issues

| Issue | Fix |
|-------|-----|
| High failure rate | Check `extra_payload` — Qwen3 needs `enable_thinking: false` |
| All requests fail | Verify endpoint is InService: `run.py --status` |
| Low output validation | Thinking mode consuming output budget |

### CloudWatch Logs

```bash
aws logs describe-log-groups \
  --log-group-name-prefix /aws/sagemaker/Endpoints/<NAME> \
  --region us-west-2
```

---

## 7. Cost Management

| Instance | $/hr | $/day |
|----------|------|-------|
| ml.g7e.2xlarge | $4.20 | $100 |
| ml.g6e.12xlarge | $13.12 | $315 |
| ml.g5.12xlarge | $7.09 | $170 |
| ml.p5e.48xlarge | ~$80 | $1,920 |

**Tips:**
1. Always include `cleanup` in pipeline — endpoints cost money when idle
2. Run one p5e at a time ($80/hr)
3. Check for orphans: `python run.py --status --region us-west-2`
4. Use `--only deploy` to deploy, validate, then `--only benchmark,cleanup`

| Scenario | Time | Cost |
|----------|------|------|
| 1 model, vanilla, g7e | ~30 min | ~$2 |
| 1 model, vanilla + EAGLE3, g7e | ~1 hr | ~$4 |
| Full matrix (5 models), g7e | ~1 day | ~$500-1000 |

---

*Last updated: March 30, 2026*
