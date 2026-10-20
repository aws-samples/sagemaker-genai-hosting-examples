# Deploy Qwen3.8-27B on Amazon SageMaker

This example demonstrates how to deploy [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a SageMaker real-time endpoint using boto3.

## Overview

Qwen3.8-27B is the dense 27B member of the Qwen3.8 family. It keeps the Qwen3.5 hybrid architecture (48 Gated DeltaNet linear-attention layers + 16 Gated Attention layers, built-in multi-token prediction head) and is a native vision-language model: it handles text, image, and video input and generates text output with optional chain-of-thought reasoning. New in Qwen3.8 is per-request control of reasoning depth (`reasoning_effort`) and preserved thinking across turns (`preserve_thinking`). Native context is 262,144 tokens.

Learn more in the official [model card](https://huggingface.co/Qwen/Qwen3.8-27B).

The notebook covers:

- Deploying Qwen3.8-27B on `ml.g6e.24xlarge` (4 GPUs) with three container options: vLLM, SGLang, and LMI
- Text generation
- Video understanding
- Text generation with reasoning output (thinking mode)
- Controlling reasoning depth with `reasoning_effort`
- Instruct (non-thinking) mode
- Tool calling
- Streaming inference with token-per-second metrics
- Image understanding
- Speculative decoding performance benchmarks
- Endpoint cleanup

## Container Options

The notebook provides three deployment options — pick one:

| Option | Container | Version | Image tag | transformers |
|---|---|---|---|---|
| vLLM | SageMaker vLLM DLC | 0.32.0 | `vllm:0.32.0-gpu-py312-cu130-ubuntu24.04-sagemaker` | 5.19.0 |
| SGLang | SageMaker SGLang DLC | 0.5.22 | `sglang:0.5.22-gpu-py312-cu130-ubuntu24.04-sagemaker` | 5.19.0 |
| LMI | DJL LMI | v30 (0.38.0, vLLM 0.31.0 backend) | `djl-inference:0.38.0-lmi30.0.0-cu130` | 5.17.0 |

All three options include speculative decoding configuration using Qwen3.8's native multi-token prediction (MTP) head (`mtp` method in vLLM and LMI, `EAGLE` with 3/1/4 steps/top-k/draft tokens in SGLang), which improves throughput with no degradation in output quality.

Container notes:

- **vLLM / LMI** return the reasoning trace in `message.reasoning` (`delta.reasoning` when streaming); **SGLang** returns it in `message.reasoning_content`. The notebook handles both.
- **vLLM 0.32** rejects per-request `mm_processor_kwargs` unless the server runs with `--trust-request-mm-kwargs`, so the video example relies on the default frame sampling (`fps=2`).
- **SGLang**: `SM_SGLANG_ENABLE_FLASHINFER_ALLREDUCE_FUSION` (used in the Qwen3.6 sample) must not be set on L40S: SGLang 0.5.22 fails at startup with `FlashInfer allreduce fusion requires SM90 or SM10X NVIDIA GPUs`. With the settings in the notebook, SGLang caps `max_running_requests` at 29 because of the GDN (Mamba) state cache; raise `--mamba-full-memory-ratio` / `--max-mamba-cache-size` or use `--mamba-ssm-dtype bfloat16` if you need more concurrent requests.
- The SageMaker `invoke_endpoint` call times out after 60 seconds. Long thinking traces (for example the video example, about 2,000 reasoning tokens) can approach that limit; use streaming, a lower `reasoning_effort`, or non-thinking mode for long generations.

## Key Configuration

| Setting | Value |
|---|---|
| Instance | ml.g6e.24xlarge (4x NVIDIA L40S, 48 GB each) |
| Tensor Parallel | 4 |
| Max Context Length | 262,144 tokens (full native context) |
| Model | `Qwen/Qwen3.8-27B` (BF16, 55.6 GB) |
| Tool Calling | Enabled (`qwen3_coder` parser) |
| Reasoning | Enabled (`qwen3` parser) |

The full 262,144-token context fits with BF16 weights and BF16 KV cache on 4x L40S: at startup vLLM reported a KV cache of 1,413,131 tokens with MTP enabled (5.4x concurrency at full context) and LMI reported 1,458,988 tokens.

`ml.g6e.12xlarge` has the same four L40S GPUs and the settings should carry over, but it was **not tested**: repeated endpoint creations in us-east-1 and us-west-2 failed with `InsufficientInstanceCapacity` while this sample was being validated.

## Tested Configurations

All three options were deployed with the notebook on `ml.g6e.24xlarge` in us-east-1 and every inference cell was run:

| Cell | vLLM | SGLang | LMI |
|---|---|---|---|
| Text | ✅ | ✅ | ✅ |
| Video | ✅ | ✅ | ✅ (notebook run 42.9 s; 2 further runs at default effort hit the 60 s invoke timeout, see notes) |
| Reasoning | ✅ | ✅ | ✅ |
| `reasoning_effort` | ✅ | ✅ | ✅ |
| Non-thinking | ✅ | ✅ | ✅ |
| Tool calling (2 turns) | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ |
| Image understanding | ✅ | ✅ | ✅ |

With MTP enabled and the same benchmark workload as below, the three containers performed similarly:

| Container | Latency (c=1) | Throughput (c=1) | Latency (c=10) | Throughput (c=10) |
|---|---|---|---|---|
| vLLM 0.32.0 | 5,514 ms | 92.8 tok/s | 10,300 ms | 491.7 tok/s |
| SGLang 0.5.22 | 5,504 ms | 93.0 tok/s | 11,248 ms | 451.9 tok/s |
| LMI v30 | 5,622 ms | 91.1 tok/s | 10,189 ms | 500.4 tok/s |

## Speculative Decoding Performance

Measured with the vLLM option on `ml.g6e.24xlarge`: MTP with 3 speculative tokens vs. the same configuration without `SM_VLLM_SPECULATIVE_CONFIG`:

| Metric | Concurrency | SD | no-SD | SD Improvement |
|---|---|---|---|---|
| Latency | 1 | 5,514 ms | 11,632 ms | −53% |
| Latency | 10 | 10,300 ms | 18,225 ms | −43% |
| Throughput | 1 | 92.8 tok/s | 44.0 tok/s | +111% |
| Throughput | 10 | 491.7 tok/s | 280.3 tok/s | +75% |

Methodology: a single fixed prompt in non-thinking mode, `temperature=0`, `max_tokens=512` (every request generated 512 tokens), 2 warm-up requests, then 10 requests at concurrency 1 and 50 requests at concurrency 10. Latency is the mean end-to-end request latency measured on the client through `invoke_endpoint`; throughput is total completion tokens divided by wall time. The vLLM log reported a mean MTP acceptance length of ~2.98 tokens per step during this run. Results depend on the prompt, sampling settings, and output length; treat them as indicative of this setup only.

## Sampling Parameters (Best Practices)

| Mode | Temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

`reasoning_effort` accepts `xhigh` (default), `medium`, and `low`; pass it in `chat_template_kwargs`. Disable thinking with `"chat_template_kwargs": {"enable_thinking": false}`.

## Getting Started

1. Open the notebook in SageMaker Studio or a Jupyter environment with appropriate AWS permissions.
2. Ensure your execution role has access to create SageMaker endpoints and pull ECR images.
3. Choose one of the three container options (vLLM, SGLang, or LMI) and run only that cell.
4. Run the remaining cells sequentially.
5. Remember to run the cleanup cell when done to avoid ongoing charges.
