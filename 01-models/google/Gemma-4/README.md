# Deploy Gemma 4 31B on Amazon SageMaker

This example demonstrates how to deploy [Google's Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it) model on a SageMaker real-time endpoint using the vLLM container and boto3.

## Overview

Gemma 4 is a multimodal model from Google DeepMind that handles text and image input and generates text output. It supports a 256K context window and 140+ languages.

The notebook covers:

- Deploying Gemma 4 31B-it on `ml.g6e.12xlarge` (4 GPUs) using the SageMaker vLLM container
- Text generation (with and without reasoning/thinking mode)
- Image understanding via OpenAI-compatible vision API
- Streaming inference with token-per-second metrics
- Tool calling support (`gemma4` parser)
- Endpoint cleanup

## Key Configuration

| Setting | Value |
|---|---|
| Container | vLLM 0.19.1 (SageMaker DLC) |
| Instance | ml.g6e.12xlarge (4x NVIDIA L40S) |
| Tensor Parallel | 4 |
| Max Context Length | 32,768 tokens |
| Model | `google/gemma-4-31B-it` |

> The notebook also references `google/gemma-4-26B-A4B-it` (MoE variant) as an alternative.

## Getting Started

1. Open the notebook in SageMaker Studio or a Jupyter environment with appropriate AWS permissions.
2. Ensure your execution role has access to create SageMaker endpoints and pull ECR images.
3. Run the cells sequentially.
4. Remember to run the cleanup cell when done to avoid ongoing charges.
