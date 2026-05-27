# 07 — Benchmark

Benchmarking utilities for SageMaker LLM inference.

| Suite | Description |
|-------|-------------|
| [`sagemaker-inference-benchmark-suite/`](sagemaker-inference-benchmark-suite/) | Config-driven (YAML recipe) framework for deploying and benchmarking LLMs on SageMaker — supports Vanilla, Prefix Caching, EAGLE3 / MTP speculative decoding, LMCache, and KV-offload across g5 / g6e / g7e / p5e instances. Ships with 40+ pre-tested recipes (Qwen3, GPT-OSS, Llama4, GLM-5, Kimi K2.5, etc.). |
| [`autobench/`](autobench/) | Cross-engine benchmarking capabiltiies for LLMs on SageMaker and Bedrock |

## Overview

| | **AutoBench** (`autobench/`) | **SageMaker Inference Benchmark Suite** (`sagemaker-inference-benchmark-suite/`) |
|---|---|---|
| **Focus** | Multi-path cross-platform comparison (SMAI vs HyperPod vs BYOM) at scale | Single-path (SMAI only) with deep optimization exploration |
| **Load Generator** | SageMaker AI Benchmarking (NVIDIA AIPerf) — AWS-managed service | Custom Python client (direct HTTP to endpoint with streaming SSE) |
| **Config Model** | Single monolithic YAML; matrix expansion (models × workloads × concurrency) | Per-recipe YAML files (one file per model × instance × optimization combo) |
| **Execution** | Unattended 24/7 via Processing Jobs (`--submit`) | Interactive (TUI) or recipe-driven, typically attended |
| **Platforms** | SMAI, HyperPod EKS, Bedrock BYOM | SMAI only |
| **Optimizations Tested** | Baseline + planned optimized (future) | EAGLE3, MTP, Prefix Cache, LMCache, KV Offload |
| **Hardware** | p6-b200 (B200) focused | g5, g6e, g7e, p5e — wider range |
| **Results Pipeline** | Athena → QuickSight (centralized, queryable, self-describing rows) | Local CSV → Markdown report (per-run) |
| **Dedup/Versioning** | Deterministic job_id with config_hash | N/A (one recipe = one result file) |
| **Instance Types** | Single focus (p6-b200.48xlarge) | Multiple (14B on g7e through 744B on p5e) |
| **Cost Analysis** | Not yet | Built-in ($/M output tokens calculation) |
| **UX** | CLI-only (headless, automation-first) | TUI dashboard + CLI (interactive-first) |
| **Metrics** | 31 fields from AIPerf (P1-P99 distributions, HTTP trace, raw JSON) | 10 fields from custom client (TTFT, latency, throughput, tokens) |

## Different Questions Answered

**AutoBench answers:**
- How does the same model perform across SMAI vs HyperPod vs BYOM?
- How do results change across vLLM versions and configs over time?
- What's the throughput/latency profile at varying concurrency for a fleet of models?
- Can I track regressions across tranches automatically?

**Benchmark Suite answers:**
- What's the best optimization strategy for this specific model on this instance type?
- How does EAGLE3 speculative decoding compare to vanilla for Qwen3-32B on g7e?
- What's the cost per million output tokens with prefix caching enabled?
- Which instance type gives the best price-performance for my model?

## Complementary Strengths

| Opportunity | How |
|---|---|
| Suite's recipes → AutoBench configs | Translate working recipe optimizations (EAGLE3 params, prefix caching) into AutoBench's `hyperpod_args` or `model-configs/` |
| AutoBench's Athena pipeline for Suite | Suite results could write to the same Athena table for unified QuickSight dashboards |
| Suite's cost analysis → AutoBench | Add `$/M tokens` calculation to AutoBench's `flatten_metrics` |
| Suite's wider instance coverage | Port Suite's g7e/g6e/g5 recipes as AutoBench tranches |
| AutoBench's HyperPod/BYOM paths | Suite is SMAI-only; AutoBench extends comparison to EKS and Bedrock |

## When to Use Which

| Scenario | Use |
|----------|-----|
| "Run overnight benchmarks for 7 models across all platforms" | AutoBench |
| "Try EAGLE3 vs vanilla for Qwen3-32B on g7e and see cost impact" | Benchmark Suite |
| "Compare SMAI vs HyperPod for the same model" | AutoBench |
| "Find the cheapest instance type for my model" | Benchmark Suite |
| "Track performance across vLLM version upgrades" | AutoBench |
| "Interactive exploration of optimization strategies" | Benchmark Suite (TUI) |
| "Generate a QuickSight dashboard for leadership" | AutoBench (Athena pipeline) |
| "Quick one-off benchmark of a new model" | Benchmark Suite (ad-hoc mode) |
