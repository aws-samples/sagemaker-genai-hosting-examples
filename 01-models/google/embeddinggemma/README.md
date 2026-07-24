# Deploy EmbeddingGemma on Amazon SageMaker with vLLM

Deploy [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) — Google's 300M parameter multilingual text embedding model — to a **SageMaker AI real-time endpoint** using the **AWS Deep Learning Container for vLLM**.

## What's covered

- Deploying EmbeddingGemma using the vLLM SageMaker DLC (`server-sagemaker-cuda` variant)
- Configuring vLLM as a pooling (embedding) model via `SM_VLLM_*` environment variables
- Calling the OpenAI-compatible `/v1/embeddings` endpoint via `invoke_endpoint`
- Using EmbeddingGemma's task-specific prompt prefixes for retrieval, QA, classification, and similarity
- Client-side embedding truncation with Matryoshka Representation Learning (MRL)

## Architecture

```
SageMaker Studio Notebook
        │
        ▼
SageMaker Real-Time Endpoint
        │
        ▼
vLLM DLC (server-sagemaker-cuda)
        │
        ▼
google/embeddinggemma-300m (downloaded from Hugging Face Hub at startup)
```

## Prerequisites

- An AWS account with SageMaker access
- A Hugging Face account with:
  - Access granted to [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) (gated — requires accepting Google's Gemma license)
  - A [Hugging Face access token](https://huggingface.co/settings/tokens) (read scope)

## Quick Start

1. Open `deploy_embeddinggemma_vllm_sagemaker.ipynb` in SageMaker Studio
2. Run all cells top to bottom
3. When prompted, paste your Hugging Face access token (input is hidden, never stored)

## Notebook walkthrough

| Section | Description |
|---------|-------------|
| Setup | Imports, region/role detection, helper functions |
| Token input | Securely capture HF token via `getpass` |
| Container config | vLLM DLC image URI, instance type, env variables |
| Deploy | Create SageMaker Model → EndpointConfig → Endpoint |
| Inference | Embed queries and documents, compute cosine similarity |
| MRL truncation | Client-side embedding truncation to 512/256/128 dims |
| Cleanup | Delete endpoint, config, and model |

## Instance & Container

| Parameter | Value |
|-----------|-------|
| Instance type | `ml.g5.xlarge` |
| vLLM DLC version | `0.25.1-gpu-py312-cu130-ubuntu22.04-sagemaker` |
| Inference AMI | `al2-ami-sagemaker-inference-gpu-3-1` |
| Model context length | 2048 tokens |
| Output dimensions | 768 (client-side MRL truncation to 512/256/128 supported) |

## Task-Specific Prompt Prefixes

EmbeddingGemma requires task-specific prefixes for best results. vLLM does **not** add these automatically — you must prepend them in your code:

| Use case | Prefix |
|----------|--------|
| Query (retrieval) | `task: search result \| query: {text}` |
| Document (retrieval) | `title: none \| text: {text}` |
| Question answering | `task: question answering \| query: {text}` |
| Classification | `task: classification \| query: {text}` |
| Semantic similarity | `task: sentence similarity \| query: {text}` |

See the [model card](https://huggingface.co/google/embeddinggemma-300m#prompt-instructions) for the full list.

## Cost Estimate

| Resource | Estimated cost |
|----------|---------------|
| `ml.g5.xlarge` endpoint | ~$1.21/hour while active |
| Model download at startup | One-time (~600MB from HF Hub) |

**Always run the cleanup cell** at the end of the notebook to delete the endpoint and avoid ongoing charges.

## Key Implementation Notes

**Role resolution:** The notebook resolves the SageMaker execution role via `iam.get_role()` rather than regex reconstruction from the STS ARN. This correctly handles roles under the `service-role/` path prefix.

**MRL truncation:** The `dimensions` parameter for server-side truncation is not supported by vLLM `0.25.1` for this model. The notebook implements equivalent client-side truncation: slice the full 768-dim vector and re-normalize.

**Token security:** The Hugging Face token is captured via `getpass.getpass()` — it is never hardcoded, written to disk, or stored in notebook outputs.

## References

- [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m)
- [vLLM DLC — SageMaker AI deployment guide](https://aws.github.io/deep-learning-containers/vllm/deployment/sagemaker/)
- [vLLM DLC — configuration reference](https://aws.github.io/deep-learning-containers/vllm/configuration/)
- [Available Deep Learning Container images](https://aws.github.io/deep-learning-containers/reference/available_images/#vllm-ubuntu)
- [SageMaker Python SDK docs](https://sagemaker.readthedocs.io/)
