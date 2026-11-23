# Optimized SageMaker AI endpoints workshop

This workshop shows how to deploy, benchmark, tune, and harden an open-weight LLM on Amazon SageMaker AI endpoints. It uses only the AWS Python SDK (boto3), with no dependency on the SageMaker Python SDK. All labs serve `Qwen/Qwen3.5-4B` with the AWS vLLM Deep Learning Container (DLC), deployed as [inference components](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-components.html).

## Labs

| Lab | Notebook | What you learn |
|---|---|---|
| 1 | [lab1/lab1_deploy_benchmark.ipynb](./lab1/lab1_deploy_benchmark.ipynb) | Deploy a model, enable speculative decoding, and benchmark both configurations |
| 2 | [lab2/lab2_optimizations.ipynb](./lab2/lab2_optimizations.ipynb) | Let SageMaker GenAI Inference Recommender find the best serving configuration |
| 3 | [lab3/lab3_pools_routing.ipynb](./lab3/lab3_pools_routing.ipynb) | Make a deployment resilient with instance pools and cross-Region inference routing |

### Lab 1: Deploy and benchmark with speculative decoding

- Deploys two inference components on one `ml.g7.12xlarge` endpoint: the base model, and the same model with multi-token prediction (MTP) speculative decoding (`SM_VLLM_SPECULATIVE_CONFIG`).
- Shows how the vLLM DLC turns `SM_VLLM_*` environment variables into vLLM server arguments.
- Uses the benchmarking feature of SageMaker GenAI Inference Recommender (`create_ai_workload_config`, `create_ai_benchmark_job`) to run the same AIPerf workload against both components. The workload sweeps concurrency over 1, 2, 4, and 8.
- Downloads the results from Amazon S3 and plots mean request latency and output token throughput for each configuration.

### Lab 2: Optimize with GenAI Inference Recommender

- Downloads the model weights from the Hugging Face Hub and uploads them to Amazon S3.
- Defines a fixed workload: concurrency 8, 256 input tokens, and 128 output tokens per request.
- Starts a recommendation job (`create_ai_recommendation_job`) that optimizes for throughput on `ml.g7.12xlarge`. The job deploys and benchmarks candidate vLLM configurations and ranks them.
- Reads the top recommendation (container image, instance type, number of copies, and vLLM settings), deploys it with weights loaded from S3, and sends a test request.

### Lab 3: Instance pools and cross-Region Inference Router

- Deploys the model in two Regions (`us-west-2` and `us-east-1` by default).
- Creates each endpoint with `InstancePools`, a priority-ordered list of GPU instance types. If the preferred type has no capacity, SageMaker AI provisions the next type in the list.
- Creates an Inference Router in front of both inference components, then simulates a Regional failure (failover) and the recovery (failback). The router reports which Region served each request and why (`local-preferred` or `only-healthy`).

> **Note:** Workshop Studio does not support the Inference Router API yet, so the router cells (steps 4–6) are commented out. Uncomment them if you run the lab in your own account.

## Prerequisites

- A SageMaker AI Studio environment. The notebooks derive the execution role from the Studio session. Outside Studio, set `role` to your execution role ARN.
- An S3 bucket named `sagemaker-<region>-<account-id>` for benchmark results, model weights, and recommendation output. You can also change the `bucket` variable.
- Service quotas for the GPU instance types used: `ml.g7.12xlarge` in labs 1 and 2, and the instance pool types in lab 3.
- Lab 3 needs access to two Regions.

## Cost and cleanup

The labs create GPU endpoints, and the benchmark and recommendation jobs take 10–20 minutes each. Each notebook ends with a cleanup section that deletes the endpoints, inference components, models, and jobs it created. Benchmark results and model weights stay in S3 until you delete them.
