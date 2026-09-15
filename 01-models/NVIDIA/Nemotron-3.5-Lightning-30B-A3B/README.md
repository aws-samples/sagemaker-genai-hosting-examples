# Deploy NVIDIA Nemotron 3.5 Lightning 30B A3B on Amazon SageMaker AI

This directory contains examples for deploying
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
with speculative decoding on Amazon SageMaker AI.

The model is a 4-bit NVFP4-quantized, 30-billion-parameter mixture-of-experts
model with approximately 3 billion active parameters.

## Contents

| File | Description |
| --- | --- |
| [`Nemotron-3.5-Lightning-30B-A3B-NVFP4.ipynb`](Nemotron-3.5-Lightning-30B-A3B-NVFP4.ipynb) | Notebook example that deploys the model to a SageMaker real-time endpoint, invokes it, benchmarks speculative decoding, and deletes the created resources. |
| [`hp_deploy.yaml`](hp_deploy.yaml) | SageMaker HyperPod deployment example using an `InferenceEndpointConfig` resource and the SageMaker AI Inference Operator. |

## SageMaker Endpoint Notebook

The notebook uses `boto3` and the SageMaker vLLM 0.27.1 inference container to:

1. Deploy the model to one `ml.g7.2xlarge` instance.
2. Enable prefix caching and configure SageMaker's new `PREFIX_AWARE` routing
   strategy with a prefix length of 1,024 and a concurrency threshold of 10.
3. Enable speculative decoding with the
   `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` draft model and
   three speculative tokens.
4. Run text-generation requests and a SageMaker AI benchmark job.
5. Clean up the benchmark job, endpoint, endpoint configuration, and model.

Prefix-aware routing directs requests with similar prefixes to the same
instance to improve prefix-cache reuse. The concurrency threshold provides
overload protection by allowing requests to be routed to a less busy instance.

Before running the notebook, replace the placeholders for the Hugging Face
token, benchmark IAM role ARN, and benchmark S3 output location. The notebook
also requires an AWS identity with permissions to create and invoke SageMaker
endpoints and benchmark jobs.

## Speculative Decoding Results

The notebook includes results for 300 requests with approximately 1,024 input
tokens and 256 output tokens per request.

| Metric | Baseline | Speculative decoding | Change |
| --- | ---: | ---: | ---: |
| Average request latency | 4,637 ms | 3,250 ms | -29.9% |
| Median request latency | 4,618 ms | 3,111 ms | -32.6% |
| Median time to first token | 219 ms | 153 ms | -30.0% |
| Total token throughput | 2,757 tokens/s | 3,912 tokens/s | +41.9% |
| Output token throughput | 552 tokens/s | 783 tokens/s | +41.8% |
| Average inter-token latency | 17.00 ms | 11.64 ms | -31.5% |

These results show higher throughput and lower typical latency. The notebook
also records regressions in p99 request latency and p99 time to first token, so
evaluate the configuration against workload-specific tail-latency requirements.

## HyperPod Deployment

`hp_deploy.yaml` provides a SageMaker HyperPod deployment example configured
with:

- One `ml.g6e.2xlarge` replica and one GPU
- vLLM 0.27.1 with an OpenAI-compatible `v1/chat/completions` endpoint
- Hugging Face model prefetching
- Prefix caching, Nemotron reasoning, and tool-call parsing
- The same DSpark speculative-decoding model used by the notebook

Before applying the manifest, create the referenced `hf-token-secret` in the
target Kubernetes namespace and review the instance type, image, replica count,
and model settings. The manifest declares prefix-aware intelligent routing but
currently sets `intelligentRoutingSpec.enabled` to `false`.

Apply it to a HyperPod cluster configured with the SageMaker AI Inference
Operator:

```bash
kubectl apply -f hp_deploy.yaml
```
