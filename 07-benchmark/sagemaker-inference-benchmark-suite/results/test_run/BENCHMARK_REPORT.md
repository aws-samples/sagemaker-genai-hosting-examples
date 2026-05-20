# SageMaker Inference Benchmark Report
**Generated**: 2026-03-30 13:31:13
**Data points**: 5

## Peak Throughput (Highest Concurrency)

| Model | Instance | Optimization | Use Case | tok/s | RPS | Agg tok/s | p50 (ms) | $/M tokens |
|-------|----------|-------------|----------|-------|-----|-----------|----------|-----------|
| qwen3-32b | g7e.2xl | eagle3 | multiturn_chat | 34.2 | 0.91 | 546.0 | 17652.7 | $2.14 |

## Single Request Performance (C=1)

| Model | Instance | Optimization | Use Case | tok/s | Latency p50 (ms) | Avg Input | Avg Output |
|-------|----------|-------------|----------|-------|-----------------|-----------|------------|
| qwen3-32b | g7e.2xl | eagle3 | multiturn_chat | 39.9 | 14444.7 | 464.4 | 600 |
| qwen3-32b | g7e.2xl | eagle3 | multiturn_chat | 44.6 | 13108.3 | 495.5 | 600 |

## Optimization Speedup (vs Vanilla)

No vanilla baseline data for comparison.

## Latency Scaling Under Load

| Model | Instance | Optimization | Use Case | C=1 p50 | Peak C p50 | Degradation |
|-------|----------|-------------|----------|---------|-----------|------------|
| qwen3-32b | g7e.2xl | eagle3 | multiturn_chat | 13108ms | 17653ms (C=16) | +35% |
