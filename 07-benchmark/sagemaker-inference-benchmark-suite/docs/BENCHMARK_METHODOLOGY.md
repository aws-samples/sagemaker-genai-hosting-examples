# SageMaker Inference Benchmark Methodology

## Overview

This document describes the benchmark methodology for evaluating SageMaker inference optimizations across multiple models, instance types, and use cases. Results are directly comparable across configurations, with particular attention to prefix caching behavior.

## Models Under Test

| Model | Architecture | Total Params | Active Params | Quantization |
|-------|-------------|-------------|--------------|-------------|
| Qwen3-32B | Dense Transformer | 32B | 32B | BF16 |
| GPT-OSS-20B | Mixture-of-Experts | 21B | 3.6B | MXFP4 |
| Kimi K2.5 | Mixture-of-Experts | 1T | 32B | INT4 (native) |
| Qwen3-235B-A22B | Mixture-of-Experts | 235B | 22B | BF16 |
| Qwen3.5-122B-A10B | Mixture-of-Experts | 122B | 10B | BF16 |

## Instance Types

| Instance | GPUs | VRAM | Architecture | CUDA | $/hr |
|----------|------|------|-------------|------|------|
| ml.g7e.2xlarge | 1x RTX PRO 6000 | 96 GB | Blackwell | 12.9 | $4.20 |
| ml.g6e.12xlarge | 4x L40S | 192 GB | Ada Lovelace | 12.9 | $13.12 |
| ml.g5.12xlarge | 4x A10G | 96 GB | Ampere | 12.8 | $7.09 |
| ml.p5e.48xlarge | 8x H200 | 640 GB | Hopper | 12.8+ | ~$80 |

Note: Blackwell (g7e) requires CUDA 12.9+. Ampere (g5) requires CUDA 12.8 (12.9 containers fail). This constrains container selection per instance type.

## Optimization Configurations

| Optimization | Description | Key Setting |
|-------------|-------------|-------------|
| vanilla | No optimizations, baseline | Default vLLM config |
| prefix_cache | Automatic prefix caching | `enable_prefix_caching` |
| eagle3 | EAGLE3 speculative decoding + prefix cache | `speculative_config` with draft model |
| lmcache | KV cache CPU offloading + prefix cache | `swap_space` |
| mtp | Multi-Token Prediction (native) | `speculative_config` with method=mtp |

## Use Cases and Prompt Design

Each use case has a **shared system prompt** identical across all requests, enabling meaningful prefix caching measurement. Prompts are defined in `scripts/prompts.py`.

### Use Case 1: Multi-turn Chat

Progressive conversation depth with a shared software engineering assistant persona.

| Prompt | Cacheable Prefix | New Content | Total Input |
|--------|-----------------|-------------|-------------|
| Depth 1 | ~300 tok (system) | ~70 tok | ~370 tok |
| Depth 2 | ~500 tok (system + turn 1) | ~80 tok | ~580 tok |
| Depth 3 | ~820 tok (system + turns 1-2) | ~70 tok | ~890 tok |

### Use Case 2: Tool Calling

AI assistant with 10 tool definitions. System prompt (~500 tokens) is 80%+ of input.

| Prompt | System (cached) | User Query | Total Input |
|--------|----------------|------------|-------------|
| Business trip | ~500 tok | ~80 tok | ~580 tok |
| Revenue analysis | ~500 tok | ~95 tok | ~595 tok |
| Incident response | ~500 tok | ~115 tok | ~615 tok |

### Use Case 3: Long Context Document QA

Short shared system prompt (~100 tokens) + varying long documents (1000-1600 tokens). Prefix caching benefit is minimal here — serves as a decode-dominated baseline.

**Output**: ~600 tokens (max_tokens=600) for all use cases.

## Benchmark Execution

### Parameters
- **Concurrency levels**: 1, 2, 4, 8, 16, 32
- **Requests per level**: 30
- **Warmup**: 3 requests before measurement
- **Pause between levels**: 3 seconds
- **Prompt rotation**: Round-robin through 3 prompts per use case

### Metrics Collection

Per-request metrics (streaming by default):

| Metric | Source | Description |
|--------|--------|-------------|
| `ttft_ms` | SSE first chunk | Time To First Token — first content delta arrival |
| `latency_ms` | Wall clock | Full request-to-completion time |
| `input_tokens` | `usage.prompt_tokens` | Prompt token count from vLLM |
| `output_tokens` | `usage.completion_tokens` | Completion token count from vLLM |
| `tok_per_sec` | `output_tokens / (latency_ms / 1000)` | Per-request output speed |

Aggregate metrics per concurrency level:

| Metric | Formula | Description |
|--------|---------|-------------|
| `ttft_p50/p90/avg` | percentiles of `ttft_ms` | TTFT distribution |
| `latency_p50/p90/p99` | percentiles of `latency_ms` | Latency distribution |
| `rps` | `concurrency / avg_latency_seconds` | Throughput in requests/sec |
| `aggregate_output_tok_sec` | `rps × avg_output_tokens` | Total output tokens/sec |
| `$/M output tokens` | `($/hr) / (agg_tok_sec × 3600) × 1M` | Cost efficiency |

**Cost calculation**: Self-hosted pricing — instance $/hr is fixed. Output throughput at peak concurrency determines cost per token. Input tokens are not separately priced but affect throughput via prefill time. No model-specific or customer-specific hardcoding.

### Output Validation
- Output must be at least 10 characters
- Token counts from vLLM internal metrics, not external tokenization

## Interpreting Results

### Prefix Caching Impact
Most visible when shared prefix is large relative to total input (tool calling: ~80% cached). Modest benefit for decode-dominated workloads (long outputs).

### EAGLE3 Speculative Decoding
- Best at C=1 (1.6-2.1x speedup), tapering at C=32 as GPU saturates
- TP=1 retains more speedup than TP=4 at high concurrency
- Structured outputs (tool calling) have higher acceptance rates

### Instance Selection
- **TP=1 vs TP=4**: Single-GPU eliminates inter-GPU sync overhead
- **Cost per token**: Cheapest instance/hr is not always cheapest per token
- **Latency degradation**: How much p50 increases from C=1 to C=32

## Reproducibility

All configuration is in YAML recipes under `recipes/`. Run any recipe with:

```bash
python run.py -f recipes/<recipe>.yaml --dry-run   # validate
python run.py -f recipes/<recipe>.yaml              # execute
```

Results are saved as CSV in `results/matrix/`:
```
{model}_{optimization}_{instance}_{use_case}_{region}_{timestamp}.csv
```
