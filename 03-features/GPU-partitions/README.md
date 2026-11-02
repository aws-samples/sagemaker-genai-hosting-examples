# GPU Partitions (NVIDIA MIG) on SageMaker Endpoints

This example uses NVIDIA Multi-Instance GPU (MIG) to split the GPUs behind a SageMaker real-time endpoint into partitions. It then places [inference components](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html) on specific partition types, so several inference workloads can share one instance. Each MIG partition gets its own GPU memory, cache, and compute.

## What the notebook covers

- Setting a mixed MIG layout on an `ml.p4d.24xlarge` endpoint with `AcceleratorPartitionLayouts`
- Deploying one model (`Qwen/Qwen3-0.6B` on the vLLM DLC) to three inference components, each on a different MIG profile
- Sending requests to each component with `InferenceComponentName`
- Measuring latency and throughput for each component
- Testing isolation: measuring one partition while another is under load
- Scaling a component until it uses every partition of its type
- Requesting more copies than the layout can hold to show the `FailureReason` SageMaker returns

## Architecture

```text
ml.p4d.24xlarge: 8 × A100-40GB
Each GPU: [ mig-4g.20gb ][ mig-2g.10gb ][ mig-1g.10gb ]
```

| MIG profile   | Endpoint capacity | Initial copies |
|---------------|-------------------|----------------|
| `mig-4g.20gb` | 8                 | 1 (scaled to 8) |
| `mig-2g.10gb` | 8                 | 8              |
| `mig-1g.10gb` | 8                 | 4              |

The partition layout is set in the endpoint configuration and does not change. Each inference component requests a partition type in `ComputeResourceRequirements.AcceleratorPartitionType`. Each copy uses one partition, so scaling can only use free partitions of the matching type.

```python
# Endpoint config (production variant)
"AcceleratorPartitionLayouts": [
    {"AcceleratorPartitionTypes": ["mig-4g.20gb", "mig-2g.10gb", "mig-1g.10gb"]}
]

# Inference component
"ComputeResourceRequirements": {
    "AcceleratorPartitionType": "mig-2g.10gb",
    "NumberOfAcceleratorDevicesRequired": 1,
    "MinMemoryRequiredInMb": 1024,
}
```

> MIG profile names depend on the GPU model and its memory size. For example, the profiles for A100-40GB are not the same as the profiles for A100-80GB.

## Prerequisites

- Accelerator partitioning is available in your AWS Region and account
- Enough quota and capacity for an `ml.p4d.24xlarge` endpoint
- An execution role that can manage SageMaker models, endpoint configurations, endpoints, and inference components, and can pull the vLLM image from ECR
- Internet access from the endpoint, because the model downloads from Hugging Face at startup
- If you run outside SageMaker Studio, set `role` to your execution-role ARN in the notebook

## Usage

Open [gpu_partitions_sagemaker.ipynb](gpu_partitions_sagemaker.ipynb) and run the cells in order. These flags turn the optional steps on or off:

| Flag                  | Default | Purpose                                      |
|-----------------------|---------|----------------------------------------------|
| `RUN_BENCHMARKS`      | `True`  | Latency/throughput benchmarks and scaling comparison |
| `RUN_ISOLATION_TEST`  | `RUN_BENCHMARKS` | Isolation experiment              |
| `RUN_OVERCOMMIT_DEMO` | `True`  | Over-capacity request and recovery demo      |

## Cleanup

> **Cost warning:** An `ml.p4d.24xlarge` endpoint incurs cost while up and running. When you finish, run the **Cleanup** section. It deletes the inference components, the endpoint, the endpoint configuration, and the model.
