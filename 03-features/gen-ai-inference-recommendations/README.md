# Generative AI Inference Recommendations

This folder contains notebooks for benchmarking and optimizing the deployment of generative AI models on Amazon SageMaker using the **SageMaker AI Benchmarking** and **AI Recommendation** services.

---

## Notebooks

### [`inference_optimization_benchmark_inference_component_boto3.ipynb`](inference_optimization_benchmark_inference_component_boto3.ipynb)
Deploys `openai/gpt-oss-20b` to a SageMaker endpoint using an Inference Component and runs a benchmark job against it. Measures request latency, output token throughput, TTFT, and inter-token latency under realistic load.

> **Already have a deployed endpoint?** Skip to Step 7 — the deployment steps are optional and only needed if you don't have an endpoint to benchmark.

---

### [`inference_optimization_recommendation_boto3.ipynb`](inference_optimization_recommendation_boto3.ipynb)
Runs a SageMaker AI Recommendation Job to automatically find the optimal serving configuration for a model. Uses the ShareGPT public dataset as the workload profile and targets throughput optimization on `ml.g6.24xlarge`.

---

### [`inference_optimization_recommendation_with_dataset_boto3.ipynb`](inference_optimization_recommendation_with_dataset_boto3.ipynb)
Runs a recommendation job with a **custom dataset** from S3 and model optimization enabled (`OptimizeModel=True`). Applies deep optimizations including speculative decoding (EAGLE 3), quantization, and kernel tuning targeting `ml.p5en.48xlarge`.

---

### [`inference_optimization_deploy_recommendation_boto3.ipynb`](inference_optimization_deploy_recommendation_boto3.ipynb)
Deploys a recommendation produced by the AI Recommendation Service. Fetches the `ModelPackage` from a completed recommendation job and deploys it as a SageMaker real-time endpoint.

---

## Prerequisites

- Model weights in S3 in HuggingFace SafeTensor format
- An IAM execution role with SageMaker, S3, and IAM PassRole permissions
- Sufficient GPU quota for the target instance type
- An S3 output bucket for benchmark and recommendation artifacts

## Key Concepts

| Concept | Description |
|---|---|
| **Benchmark** | Stress-tests a live endpoint by sending synthetic or real-world traffic and measuring latency, throughput, TTFT, and ITL |
| **Recommendation** | Analyzes your model and workload to find the best instance type and serving config, returning `ExpectedPerformance` metrics |
| **Inference Component** | Allows multiple model copies to share the same endpoint with fine-grained GPU allocation |
| **TTFT** | Time to First Token — how quickly the model starts responding |
| **ITL** | Inter-Token Latency — time between successive generated tokens |
