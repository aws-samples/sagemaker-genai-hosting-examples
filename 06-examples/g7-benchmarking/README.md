# Benchmarking Small LLM Inference on SageMaker AI: G7 vs G5 and G6

Companion notebooks for the AWS blog post *Benchmarking Small LLM Inference on SageMaker AI: G7 vs G5 and G6*.

Choosing the right GPU instance for LLM inference is one of the highest-leverage decisions when deploying
generative AI at scale. These notebooks benchmark two representative 30B **Mixture-of-Experts (MoE)** models
across GPU instance families on **Amazon SageMaker AI** real-time inference, and show how the new **G7**
instances (NVIDIA Blackwell) compare on throughput, latency, and cost-per-token.

They demonstrate the two ways to evaluate inference with **SageMaker AI Generative AI Inference Recommendations**:

| Notebook | Use case | Workflow | Model | Instances |
|---|---|---|---|---|
| [`uc1-benchmark-qwen.ipynb`](./uc1-benchmark-qwen.ipynb) | **AI Coding Assistant** | **Benchmarking** — you already have endpoints and want to measure them | Qwen3-Coder-30B-A3B-Instruct-FP8 | `ml.g5.12xlarge`, `ml.g6.12xlarge`, `ml.g7.12xlarge` |
| [`uc2-benchmark-nemotron.ipynb`](./uc2-benchmark-nemotron.ipynb) | **Enterprise AI Assistant** | **Recommendation** — SageMaker AI explores candidates and ranks them | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | `ml.g6`, `ml.g6e`, `ml.g7` (2xlarge → 48xlarge) |

## Use Case 1 — Benchmark Qwen3-Coder-30B

`uc1-benchmark-qwen.ipynb` deploys **Qwen3-Coder-30B-A3B-Instruct-FP8** with the DJL **Large Model Inference
(LMI) 28.0** container and benchmarks it across G5 (A10G), G6 (L4), and G7 (RTX PRO 4500 Blackwell) using
SageMaker AI generative AI benchmarking. The same model, serving container, and workload
(128 input / 128 output tokens, concurrency 4, 100 requests) are used on every instance, so the GPU generation
is the only variable — an apples-to-apples comparison.

For each instance the notebook deploys **one** endpoint and runs **two** benchmark passes against it, in a single
top-to-bottom loop:

- **Non-streaming** — output-token throughput, request throughput, and request latency (avg / P50 / P90 / P99).
- **Streaming** — time to first token (TTFT) and inter-token latency (ITL) for interactive coding applications.

Endpoints are always torn down after each instance. Run it top to bottom to reproduce the results; the
notebook's own markdown carries the step-by-step walkthrough.

> **Container note:** the notebook pins **LMI 28** because `image_uris.retrieve('djl-lmi', 'latest')` resolves to
> LMI 27, which cannot run on G7 (Blackwell / `sm_120`). LMI 28 adds `sm_120` support *and* runs on G5/G6, keeping
> the comparison on one container.

## Use Case 2 — Recommend a deployment for Nemotron-3-Nano-30B

`uc2-benchmark-nemotron.ipynb` uses the SageMaker AI Inference Recommender
(`ModelBuilder` → `generate_deployment_recommendations` → `recommendations`) to explore candidate G6/G6e/G7
configurations for **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4**, optimizing for throughput and deriving cost per
output token across conversational and long-context RAG workloads.

## Prerequisites

- An AWS account with Amazon SageMaker AI access (run from **SageMaker Studio**, or set `SAGEMAKER_ROLE_ARN`).
- An IAM execution role with SageMaker and S3 permissions.
- Instance quota for the target instance types in your Region.
- Python 3.10+ with the SageMaker Python SDK (`sagemaker>=3.16.0`).

> Results in the blog are specific to the models, configurations, workloads, and Region (us-east-2) tested.
> Benchmark with workload characteristics representative of your own application before making production decisions.
