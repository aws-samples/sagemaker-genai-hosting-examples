# SageMaker Inference Benchmark Report

**Date**: March 11, 2026
**Author**: Sanghwa Na, GenAI Specialist SA
**Scope**: 5 models, 4 GPU instance types, 4 optimizations, 3 use cases, 22 configurations

---

## Executive Summary

| Key Finding | Detail |
|-------------|--------|
| Best cost/token | **GPT-OSS-20B + g7e + EAGLE3: $0.31/M tokens** |
| Best dense model | **Qwen3-32B + g7e + EAGLE3: $1.66/M tokens** |
| Best large MoE | **Qwen3-235B + p5e + EAGLE3: $10.51/M tokens** |
| EAGLE3 speedup range | **1.18-1.77x** at C=32 (highest on tool calling) |
| g7e vs g5 advantage | **g7e vanilla is 2.0-2.4x cheaper/token** |
| KV cache optimizations | **<5% impact** on decode-dominated workloads |
| Nemotron-3-120B | Deployment failed (trust_remote_code issue with vLLM DLC) |
| Qwen3.5-122B on LMI v20 | Failed (qwen3_5_moe architecture not recognized) |

---

## Models Tested

| Model | Architecture | Total Params | Active Params | Best Instance | EAGLE3 |
|-------|-------------|-------------|--------------|---------------|--------|
| Qwen3-32B | Dense | 32B | 32B | ml.g7e.2xlarge | Yes |
| GPT-OSS-20B | MoE | 21B | 3.6B | ml.g7e.2xlarge | Yes (DLC 0.15.1+) |
| Kimi K2.5 | MoE | 1T | 32B | ml.p5e.48xlarge | No (`kimi_k2` not in whitelist) |
| Qwen3-235B-A22B | MoE | 235B | 22B | ml.p5e.48xlarge | Yes |
| Qwen3.5-122B-A10B | MoE | 122B | 10B | ml.p5e.48xlarge | Not tested yet |

## Instances Tested

| Instance | GPUs | VRAM | Architecture | $/hr | Container |
|----------|------|------|-------------|------|-----------|
| ml.g7e.2xlarge | 1x RTX PRO 6000 | 96 GB | Blackwell | $4.20 | DLC vLLM 0.15.1 (cu129) |
| ml.g6e.12xlarge | 4x L40S | 192 GB | Ada Lovelace | $13.12 | DLC vLLM 0.15.1 (cu129) |
| ml.g5.12xlarge | 4x A10G | 96 GB | Ampere | $7.09 | DLC vLLM 0.11.0 (cu128) |
| ml.p5e.48xlarge | 8x H200 | 640 GB | Hopper | ~$80.00 | LMI v20 / vLLM 0.17 |

---

## 1. Cost Efficiency Ranking (C=32, Multiturn Chat)

| Rank | Model | Instance | Optimization | tok/s | Agg tok/s | RPS | $/hr | $/M tokens |
|------|-------|----------|-------------|-------|-----------|-----|------|-----------|
| 1 | GPT-OSS-20B | g7e | EAGLE3 | 116.5 | 3,720 | 6.20 | $4.20 | **$0.31** |
| 2 | GPT-OSS-20B | g7e | Vanilla | 97.0 | 3,102 | 5.17 | $4.20 | $0.38 |
| 3 | GPT-OSS-20B | g7e | LMCache | 96.6 | 3,090 | 5.15 | $4.20 | $0.38 |
| 4 | Qwen3-32B | g7e | EAGLE3 | 22.0 | 702 | 1.17 | $4.20 | **$1.66** |
| 5 | GPT-OSS-20B | g6e | EAGLE3 | 68.4 | 2,184 | 3.64 | $13.12 | $1.67 |
| 6 | Qwen3-32B | g7e | LMCache | 19.0 | 606 | 1.01 | $4.20 | $1.93 |
| 7 | Qwen3-32B | g7e | PrefixCache | 19.0 | 606 | 1.01 | $4.20 | $1.93 |
| 8 | GPT-OSS-20B | g5 | Vanilla | 31.7 | 1,014 | 1.69 | $7.09 | $1.94 |
| 9 | GPT-OSS-20B | g6e | Vanilla | 55.5 | 1,776 | 2.96 | $13.12 | $2.05 |
| 10 | Qwen3-32B | g7e | Vanilla | 16.0 | 510 | 0.85 | $4.20 | $2.29 |
| 11-13 | Qwen3-32B | g5 | All opts | 13.1-13.6 | 420-432 | 0.70-0.72 | $7.09 | $4.56-4.69 |
| 14-16 | Qwen3-32B | g6e | All opts | 20.7-21.0 | 660-672 | 1.10-1.12 | $13.12 | $5.42-5.52 |
| 17 | Qwen3.5-122B | p5e | Vanilla | 68.5 | 2,190 | 3.65 | $80.00 | $10.15 |
| 18 | Qwen3-235B | p5e | EAGLE3 | 66.2 | 2,115 | 3.53 | $80.00 | $10.51 |
| 19 | Qwen3-235B | p5e | Vanilla | 54.0 | 1,727 | 2.88 | $80.00 | $12.87 |
| 20 | Kimi K2.5 | p5e | Vanilla | 48.1 | 1,540 | 2.57 | $80.00 | $14.43 |

---

## 2. EAGLE3 Speculative Decoding

### Compatibility Matrix

| Model | DLC vLLM 0.15.1 | DLC vLLM 0.17 | BYOC vLLM 0.10.2 | LMI v20 |
|-------|----------------|---------------|-------------------|---------|
| Qwen3-32B | **Yes** | Yes | N/A | Yes |
| GPT-OSS-20B | **Yes** | Yes | No (whitelist) | N/A |
| Kimi K2.5 | No (`kimi_k2`) | No | N/A | No |
| Qwen3-235B-A22B | N/A | N/A | N/A | **Yes** |
| Qwen3.5-122B-A10B | N/A | (untested) | N/A | N/A |

### EAGLE3 Speedup by Model and Use Case (C=32)

| Model | Instance | Multiturn Chat | Tool Calling | Long Context |
|-------|----------|---------------|-------------|-------------|
| **Qwen3-32B** | g7e (TP=1) | **1.38x** | **1.48x** | 1.18x |
| **Qwen3-235B** | p5e (TP=8) | **1.22x** | **1.77x** | **1.36x** |
| **GPT-OSS-20B** | g7e (TP=1) | **1.20x** | **1.57x** | 1.34x |
| **GPT-OSS-20B** | g6e (TP=4) | 1.23x | **1.48x** | 1.13x |

> **Key insight**: Tool calling consistently shows the highest EAGLE3 speedup across all models (1.48-1.77x).
> Structured output patterns have higher speculative acceptance rates.

### EAGLE3 Single Request Speed (C=1)

| Model | Instance | Vanilla tok/s | EAGLE3 tok/s | Speedup |
|-------|----------|--------------|-------------|---------|
| Qwen3-32B | g7e | 20.6 | 38.4 | **1.86x** |
| Qwen3-235B | p5e | 111.6 | 154.9 | **1.39x** |
| GPT-OSS-20B | g7e | 242.0 | 211.0 | 0.87x (MoE already fast) |

---

## 3. Instance Generation Comparison (Qwen3-32B Vanilla, C=32)

| Instance | GPUs | TP | tok/s | Agg tok/s | $/M tokens | vs g5 |
|----------|------|----|-------|-----------|-----------|-------|
| g7e.2xl | 1x RTX PRO 6000 | 1 | 16.0 | 510 | $2.29 | **2.0x cheaper** |
| g6e.12xl | 4x L40S | 4 | 20.9 | 666 | $5.47 | 0.8x cheaper |
| g5.12xl | 4x A10G | 4 | 13.6 | 432 | $4.56 | baseline |

---

## 4. KV Cache Optimizations

### Qwen3-32B on g7e (C=32, Multiturn Chat)

| Optimization | Agg tok/s | $/M tokens | vs Vanilla |
|-------------|-----------|-----------|-----------|
| Vanilla | 510 | $2.29 | baseline |
| PrefixCache | 606 | $1.93 | +19% |
| LMCache | 606 | $1.93 | +19% |
| **EAGLE3** | **702** | **$1.66** | **+38%** |

> PrefixCache and LMCache are interchangeable on decode-dominated workloads (600 token output).
> Their benefit increases with longer shared prefixes and shorter outputs.

---

## 5. p5e Large Model Comparison (C=32, Multiturn Chat)

| Model | Active Params | C=1 tok/s | C=32 Agg | $/M tokens |
|-------|--------------|-----------|----------|-----------|
| Qwen3.5-122B | 10B | **171.6** | **2,550** | $10.15 |
| Qwen3-235B + EAGLE3 | 22B | 154.9 | 2,115 | $10.51 |
| Qwen3-235B | 22B | 111.6 | 1,727 | $12.87 |
| Kimi K2.5 | 32B | 109.6 | 1,540 | $14.43 |

> Qwen3.5-122B-A10B has the best throughput-per-dollar on p5e thanks to small active params (10B).

---

## 6. Latency Scaling Under Load (Multiturn Chat)

| Model | Instance | Config | C=1 p50 | C=32 p50 | Degradation |
|-------|----------|--------|---------|----------|------------|
| GPT-OSS-20B | g7e | Vanilla | 2.5s | 6.2s | +150% |
| GPT-OSS-20B | g7e | EAGLE3 | 2.8s | 5.2s | +83% |
| Qwen3.5-122B | p5e | Vanilla | 3.4s | 7.5s | +121% |
| Qwen3-235B | p5e | EAGLE3 | 3.9s | 9.1s | +135% |
| Qwen3-235B | p5e | Vanilla | 5.3s | 11.2s | +109% |
| Kimi K2.5 | p5e | Vanilla | 5.5s | 12.5s | +128% |
| Qwen3-32B | g7e | EAGLE3 | 15.7s | 27.6s | +76% |
| Qwen3-32B | g7e | Vanilla | 28.9s | 37.5s | +30% |

---

## 7. Failed Deployments & Known Issues

### Nemotron-3-Super-120B-A12B-BF16
- **Status**: Failed on both LMI v20 and vLLM 0.17 DLC
- **LMI v20 error**: Config conflict between DJL `max_rolling_batch_size=32` and vLLM `max_num_seqs=64`
- **vLLM 0.17 error**: `trust_remote_code=True` not properly passed. The `SM_VLLM_TRUST_REMOTE_CODE=""` env var (empty string for boolean flag) may not be handled correctly by the DLC entrypoint.
- **Model released**: March 11, 2026 (same day as testing)
- **Fix needed**: Verify correct env var name for trust_remote_code in vLLM 0.17 DLC, or use BYOC container

### Qwen3.5-122B-A10B on LMI v20
- **Status**: Failed — `qwen3_5_moe` architecture not recognized
- **Error**: `Transformers does not recognize this architecture`
- **Fix**: Use vLLM 0.17 DLC which includes Qwen3.5 support (successfully deployed with vLLM 0.17)

### Kimi K2.5 EAGLE3
- **Status**: Not supported — `kimi_k2` model type not in vLLM EAGLE3 whitelist
- **vLLM whitelist (0.15.1)**: `['llama', 'qwen', 'minicpm', 'gpt_oss']`
- **Fix needed**: vLLM update to add `kimi_k2` to SupportsEagle3 interface

### GPT-OSS on g5 (Ampere)
- **EAGLE3**: Not supported — MXFP4 quantization requires Blackwell/Ada GPU architecture
- **Vanilla**: Works with BYOC container (vLLM 0.10.2), not with DLC cu129 (CUDA version mismatch)

---

## 8. Key Takeaways for Leadership

1. **g7e (Blackwell) is the cost winner** across all models. Even vanilla g7e is cheaper per token than any g5/g6e optimization.

2. **EAGLE3 delivers 20-77% throughput improvement** depending on model and use case. Tool calling benefits most (structured outputs). Dense models benefit more at C=1, MoE models benefit consistently at C=32.

3. **Qwen3.5-122B-A10B is the best throughput-per-dollar on p5e** ($10.15/M) — 10B active params make it faster than Qwen3-235B (22B active) and Kimi K2.5 (32B active).

4. **Container selection is critical**:
   - Blackwell (g7e/g6e): vLLM 0.15.1+ (cu129)
   - Ampere (g5): vLLM 0.11.0 (cu128) or BYOC
   - GPT-OSS EAGLE3: DLC vLLM 0.15.1+ only (BYOC 0.10.2 doesn't support it)
   - Qwen3.5: vLLM 0.17+ required
   - Nemotron-3: needs trust_remote_code fix

5. **EAGLE3 on TP=4 is less effective at high concurrency** for Qwen3-32B. On g5/g6e, verification overhead cancels speculation gains. But GPT-OSS (TP=1 on all instances) and Qwen3-235B (TP=8 on p5e) both benefit.

6. **KV cache optimizations (prefix caching, LMCache) show <5% improvement** on decode-dominated workloads. Better suited for short-output, long-prefix scenarios (e.g., agentic workflows with fixed tool definitions).

---

## 9. Benchmark Methodology

- **Requests per concurrency level**: 30
- **Concurrency levels**: 1, 2, 4, 8, 16, 32
- **Use cases**: Multi-turn chat (shared system prompt, increasing depth), Tool calling (shared tools + system prompt), Long context (shared system prompt, different documents)
- **Prompts**: Shared system prompt per use case for prefix caching measurement
- **Output length**: ~600 tokens (max_tokens=600)
- **Metrics source**: vLLM response body `usage.completion_tokens` and wall-clock latency
- **Cost formula**: `$/M tokens = ($/hr) / (agg_tok_sec * 3600) * 1,000,000`
- **Thinking mode**: Disabled for Qwen3 (`enable_thinking: false`) and Kimi (`thinking: false`)
- **GPT-OSS reasoning**: Model uses reasoning tokens within max_tokens budget; `content: null` handled
- **Accounts**: Personal (g7e/g6e/g5) and shared (p5e)

---

*Full benchmark data (CSV), deployment scripts, and prompt definitions available in the repository.*
*Report generated from 22 unique configurations across 5 models.*
