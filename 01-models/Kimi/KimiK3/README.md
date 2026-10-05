# Deploy Kimi K3 on Amazon SageMaker AI

This example deploys [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) on an Amazon SageMaker AI real-time endpoint using `boto3` (no SageMaker Python SDK dependency).

## Model

Kimi K3 is Moonshot AI's open-weight, native multimodal agentic model — the first open model in the 3-trillion-parameter class. It is built on Kimi Delta Attention (KDA) and Attention Residuals (AttnRes), with a Stable LatentMoE framework that activates 16 of 896 experts per token.

| Property | Value |
| :--- | :--- |
| Model | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) |
| Architecture | Mixture-of-Experts, 2.8T total / 104B activated |
| Layers | 93 (69 KDA + 24 Gated MLA, 1 dense) |
| Experts | 896 total, 16 selected per token, 2 shared |
| Context window | 1,048,576 tokens (notebook configures 131,072) |
| Vision encoder | MoonViT-V2 (401M) |
| Quantization | MXFP4 weights / MXFP8 activations (quantization-aware training from SFT onward) |
| Modality | Text, image |
| License | [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE) |

### Key architecture highlights

- **Kimi Delta Attention (KDA)** — a delta-rule take on linear attention replacing quadratic attention in most layers. KDA breaks conventional prefix caching assumptions; both serving engines ship a KDA-aware cache implementation.
- **Attention Residuals + Stable LatentMoE** — together with KDA these yield roughly 2.5x better scaling efficiency than Kimi K2.
- **Always-on thinking** — depth is set by the top-level `reasoning_effort` field (`low` / `high` / `max`, default `max`). There is no off switch.
- **Preserved thinking history** — multi-turn and tool-calling requests must echo the complete assistant message back, `reasoning_content` and `tool_calls` included.

Details are quoted from the published Hugging Face model card. See also the [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3).

## Serving frameworks

The notebook provides two deployment options using AWS Deep Learning Containers — pick **one**:

| Option | Container | Notes |
| :--- | :--- | :--- |
| vLLM | SageMaker [vLLM DLC](https://aws.github.io/deep-learning-containers/vllm/) | FlashInfer MXFP4 MoE runner on Blackwell, DeepEP, `runai-model-streamer` for S3 weights |
| SGLang | SageMaker [SGLang DLC](https://aws.github.io/deep-learning-containers/sglang/) | Exposes KDA-specific knobs (SSM state dtype, KDA radix cache strategy) |

Both run an OpenAI-compatible API server on port 8080.

> **Pin a container tag that supports K3.** KDA, Attention Residuals, and the MXFP4 MoE path are new model code that landed in vLLM and SGLang alongside the weight release. The tags in the notebook are placeholders — check the DLC release pages and pin the first tag listing Kimi K3 support before running.

## Instance requirements

| Instance | GPUs | Total HBM | Fits K3? |
| :--- | :--- | :--- | :--- |
| `ml.p5e.48xlarge` | 8x H200 (141 GB) | 1,128 GB | No — the ~1.4 TB of MXFP4 weights alone exceed HBM |
| `ml.p5en.48xlarge` | 8x H200 (141 GB) | 1,128 GB | No |
| `ml.p6-b200.48xlarge` | 8x B200 (180 GB) | 1,440 GB | Yes, with reduced context length |

This is the main practical difference from the [Kimi K2.5 example](../LMI/kimi-k2.5.ipynb) in this repo, which fits comfortably on a single `ml.p5e.48xlarge`.

**On multi-node.** vLLM's published K3 recipe recommends at least 8x GB300, with multi-node for real production traffic. SageMaker real-time endpoints do not shard one model across instances — `InitialInstanceCount > 1` creates independent replicas that each hold the full model. If a single instance cannot hold K3 at your target context length and concurrency, the deployment target is [SageMaker HyperPod](../../../SageMakerHyperpod/), not a real-time endpoint.

**Weight loading.** The checkpoint is roughly 1.5 TB. Pulling it from the Hugging Face Hub at container start is the most common cause of a failed health check on this model. The notebook sets `ContainerStartupHealthCheckTimeoutInSeconds` to the 3600s maximum; for repeat deployments, stage the weights in S3 once and let `runai-model-streamer` stream them.

## Key configuration

| Setting | Value |
| :--- | :--- |
| Instance | `ml.p6-b200.48xlarge` (8 GPUs) |
| Tensor parallel | 8 |
| Max context length | 131,072 (raise only after confirming free HBM) |
| KV cache | FP8 |
| KDA state dtype (SGLang) | BF16 |
| Tool calling | Enabled, `kimi_k3` parser |
| Reasoning | Enabled, `kimi_k3` parser |
| Vision encoder TP mode | `data` |
| Inference AMI | `al2023-ami-sagemaker-inference-gpu-4-1` |
| Startup health check timeout | 3600s |

## Notebook contents

| Section | Description |
| :--- | :--- |
| Prerequisites | Instance sizing, quotas, weight-loading strategy |
| Configuration | Instance, model ID, endpoint names, context length |
| Deployment options | Choose and configure the vLLM or SGLang container |
| Deployment | Create the Model, Endpoint Configuration, and Endpoint |
| Text generation | Basic inference and `reasoning_effort` levels |
| Preserved thinking history | Demonstrates the multi-turn contract K3 requires |
| Vision input | Image transcription via `image_url` content parts |
| Agentic tool calling | Multi-turn tool loop with correct message accumulation |
| OpenAI-compatible invocation | Using the endpoint's `/openai/v1` path |
| Cleanup | Delete endpoint, endpoint config, and model |

## Gotchas

1. **Do not reuse the `kimi_k2` parsers.** K3 uses `kimi_k3` for both the reasoning and tool-call parsers. The K2 parsers will not fail loudly — you will get malformed tool calls and reasoning text leaking into `content`.
2. **Append the whole assistant message.** Appending only `content` on follow-up turns silently degrades multi-turn and agentic quality.
3. **`reasoning_effort` is a top-level field**, not a `chat_template_kwargs` entry.
4. **Blackwell capacity is scarce.** The notebook includes a commented-out `CapacityReservationConfig` block; use it if you hold a reservation.

## References

- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3)
- [vLLM K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- [SGLang K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- [AWS Deep Learning Containers](https://aws.github.io/deep-learning-containers/)
- [OpenAI-compatible API support for SageMaker AI endpoints](https://aws.amazon.com/blogs/machine-learning/announcing-openai-compatible-api-support-for-amazon-sagemaker-ai-endpoints/)
