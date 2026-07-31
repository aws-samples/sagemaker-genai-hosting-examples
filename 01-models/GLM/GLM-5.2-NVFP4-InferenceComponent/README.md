# Deploy GLM-5.2-NVFP4 on a SageMaker AI Inference Component (ml.g7e.48xlarge)

This example demonstrates how to deploy [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) — the NVFP4 (4-bit) quantization of GLM-5.2 — on an Amazon SageMaker AI **Inference Component** backed by a single `ml.g7e.48xlarge`, using `boto3` (no SageMaker Python SDK dependency).

## Model

| Property | Value |
| :--- | :--- |
| Model | [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| Base model | [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) |
| Architecture | Mixture-of-Experts (`GlmMoeDsaForCausalLM`), 753B total / 40B active |
| Quantization | NVFP4 (Model Optimizer v0.46.0), ~464 GB on disk |
| License | MIT |

## Why this configuration

- **NVFP4 on g7e.48xlarge**: the bf16 (~1.5 TB) and FP8 (~753 GB) releases exceed the instance's 768 GB total GPU memory (8x96 GB Blackwell). NVFP4 weights land at ~58 GB per GPU at TP=8, leaving headroom for KV cache.
- **Inference Component**: the endpoint is created without a model; the model attaches as a component with an explicit accelerator/CPU/memory claim. This makes resource allocation first-class and allows additional components to share the instance later.
- **vLLM DLC 0.26.0 (minimum)**: on the 0.25.1 DLC this model fails at runtime with a flashinfer MLA kernel signature mismatch (`trtllm_batch_decode_with_kv_cache_mla() got an unexpected keyword argument 'kv_scale_format'`). vLLM 0.26.0 selects the Blackwell sparse-MLA backend (`FLASHINFER_MLA_SPARSE_SM120`) and serves the architecture out of the box.

## Instance Requirements

- **Instance type:** `ml.g7e.48xlarge` (8 GPUs, 768 GB total GPU memory)
- Generous startup timeouts: the ~464 GB checkpoint downloads from the Hugging Face Hub at container startup (typically 15-40 minutes depending on throughput).

## Notebook

[GLM-5.2-NVFP4-IC.ipynb](GLM-5.2-NVFP4-IC.ipynb) walks through:

1. Creating the IC-hosting endpoint (no model on the variant; managed instance scaling + LOR routing)
2. Creating the model (vLLM DLC 0.26.0, `SM_VLLM_*` environment)
3. Attaching the inference component (all 8 accelerators) and waiting for `InService`
4. Invoking with `InferenceComponentName` (reasoning and non-reasoning examples)
5. Cleanup (component first, then endpoint — the instance bills until the endpoint is deleted)
