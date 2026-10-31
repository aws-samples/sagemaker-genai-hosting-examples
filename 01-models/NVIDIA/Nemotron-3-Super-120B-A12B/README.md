# NVIDIA Nemotron-3-Super-120B-A12B on Amazon SageMaker AI

Two deployment examples for NVIDIA's 120B-parameter (12B active) hybrid Mamba-2 / Transformer / Mixture-of-Experts model.

| Notebook | Checkpoint | Container | Endpoint style | Instance |
| :--- | :--- | :--- | :--- | :--- |
| [Nemotron-3-Super-120B-A12B-NVFP4.ipynb](Nemotron-3-Super-120B-A12B-NVFP4.ipynb) | [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) (4-bit) | LMI (DJL) | classic endpoint | `ml.g7e.2xlarge` |
| [Nemotron-3-Super-120B-A12B-FP8-IC.ipynb](Nemotron-3-Super-120B-A12B-FP8-IC.ipynb) | [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) (~128 GB) | vLLM DLC 0.32.0 **or a custom vLLM image** | **Inference Component** | `ml.p6-b300.48xlarge` |

## The FP8 Inference Component example

| Property | Value |
| :--- | :--- |
| Model | [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) |
| Architecture | `NemotronHForCausalLM`: hybrid Mamba-2 / attention / MoE (512 routed + 1 shared experts, 22 active), one MTP layer |
| Quantization | FP8 weights and activations, FP8 KV-cache scheme (NVIDIA Model Optimizer) |
| License | NVIDIA Open Model License |

What it shows:

- **Inference Components**: the endpoint is created without a model; the model attaches as a component claiming `tp` accelerators per copy. At `tp=1` the checkpoint fits on one B300, so `CopyCount` can go up to 8 on one instance — eight independent vLLM engines behind one endpoint, with least-outstanding-requests routing.
- **Speculative decoding** with the checkpoint's built-in MTP layer (`{"method": "mtp", "num_speculative_tokens": 3}`) and an **FP8 KV cache**.
- **DLC or custom image**: the SageMaker vLLM DLC is the default; a switch deploys a custom vLLM image. The notebook lists what SageMaker requires of such an image (same-region ECR, Docker schema-2 manifest, a `serve` executable) and ships [`custom-image/usr/local/bin/serve`](custom-image/usr/local/bin/serve), a wrapper that reproduces the DLC's `SM_VLLM_*` flag translation, with a `crane` recipe to add it to an upstream `vllm/vllm-openai` image without Docker.
- **Tuned MoE kernels without a custom image**: tuned fused-MoE configurations can be shipped inside the model's S3 prefix and enabled with `VLLM_TUNED_CONFIG_FOLDER`.
- **Model from S3**: set `model_s3_uri` to an uncompressed prefix to deploy a fine-tuned checkpoint of the same architecture.
- **Training-plan capacity**: an optional `ml_reservation_arn` places the endpoint on a SageMaker training plan bought for inference.
- Non-streaming and **streaming** invocation (`invoke_endpoint_with_response_stream`, with time-to-first-token), scaling `CopyCount`, and cleanup.

## Instance requirements

- `ml.p6-b300.48xlarge` (8 x 288 GB) at `tp=1`; the notebook's table lists `tp` for `ml.p6-b200.48xlarge`, `ml.p5en.48xlarge` and `ml.p5.48xlarge`.
- Startup: ~128 GB of weights download per copy, then kernel warm-up; allow 15-25 minutes before the component is `InService`.
