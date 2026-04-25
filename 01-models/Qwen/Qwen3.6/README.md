# Deploy Qwen3.6-27B on Amazon SageMaker

This example demonstrates how to deploy [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on a SageMaker real-time endpoint using boto3.

## Overview

Qwen3.6 is a multimodal model from the Qwen team that delivers substantial upgrades in agentic coding and thinking preservation. It handles text, image, and video input and generates text output with optional chain-of-thought reasoning.

Learn more in the official [Qwen3.6 blog post](https://qwen.ai/blog?id=qwen3.6-27b).

The notebook covers:

- Deploying Qwen3.6-27B on `ml.g6e.12xlarge` (4 GPUs) with three container options: vLLM, SGLang, and LMI
- Text generation
- Video understanding
- Text generation with reasoning output (thinking mode)
- Streaming inference with token-per-second metrics
- Image text extraction (OCR)
- Speculative decoding performance benchmarks
- Endpoint cleanup

## Container Options

The notebook provides three deployment options — pick one:

| Option | Container | Version |
|---|---|---|
| vLLM | SageMaker vLLM DLC | 0.19.0 |
| SGLang | SageMaker SGLang DLC | 0.5.9 |
| LMI | DJL LMI | v22 (0.36.0) |

All three options include speculative decoding configuration using Qwen3.6's native multi-token prediction (MTP), which improves throughput with no degradation in output quality.

## Key Configuration

| Setting | Value |
|---|---|
| Instance | ml.g6e.12xlarge (4x NVIDIA L40S) |
| Tensor Parallel | 4 |
| Max Context Length | 32,768 tokens |
| Model | `Qwen/Qwen3.6-27B` |
| Tool Calling | Enabled (`qwen3_coder` parser) |
| Reasoning | Enabled (`qwen3` parser) |

## Speculative Decoding Performance

The notebook includes benchmarks comparing speculative decoding (SD) vs standard decoding:

| Metric | Concurrency | SD | no-SD | SD Improvement |
|---|---|---|---|---|
| Latency | 1 | 2,894 ms | 4,885 ms | −41% |
| Latency | 10 | 6,427 ms | 8,775 ms | −27% |
| Throughput | 1 | 137.2 tok/s | 81.2 tok/s | +69% |
| Throughput | 10 | 528.0 tok/s | 383.6 tok/s | +38% |

## Sampling Parameters (Best Practices)

| Mode | Temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking (general) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking (precise coding) | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

## Getting Started

1. Open the notebook in SageMaker Studio or a Jupyter environment with appropriate AWS permissions.
2. Ensure your execution role has access to create SageMaker endpoints and pull ECR images.
3. Choose one of the three container options (vLLM, SGLang, or LMI) and run only that cell.
4. Run the remaining cells sequentially.
5. Remember to run the cleanup cell when done to avoid ongoing charges.
