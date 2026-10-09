# Deploy Decision 2.0 Sol 2B on Amazon SageMaker AI

Deploy [Decision 2.0 Sol 2B](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B)
to a real-time endpoint on one `ml.g6.xlarge`, falling back to `ml.g5.xlarge` when g6
capacity is unavailable, using its published Transformers runtime and native `system_one`
API. It runs on the maintained AWS PyTorch 2.14 SageMaker DLC through the shared
[decision-model serving image](../../../03-features/decision-model-serving-image/), on
the SageMaker inference AMI with NVIDIA driver 580 that the image's CUDA 13 needs.

The model returns Choice, Yes/No (`noul`), and Score judgments over supplied
state without generating text. Its checkpoint, head, and bundled inference code
are loaded together from a pinned Hugging Face revision.

## Prerequisites

- Python 3.10+ in Jupyter, such as SageMaker Studio.
- AWS credentials with SageMaker hosting, S3 artifact and object-version listing,
  `iam:PassRole`, and
  execution-role inspection permissions.
- An existing execution role that trusts `sagemaker.amazonaws.com`, can read the
  artifact bucket, pull the serving image from your Amazon ECR, and write endpoint logs.
  Outside Studio, set `SAGEMAKER_EXECUTION_ROLE_ARN`.
- An AWS CodeBuild service role for the image build; set `CODEBUILD_ROLE_ARN`. See the
  [serving image README](../../../03-features/decision-model-serving-image/README.md#build)
  for its permissions. The notebook's caller also needs permission to create the ECR
  repository and the CodeBuild project. Set `SAGEMAKER_INFERENCE_IMAGE_URI` instead to
  reuse an image you already built.
- Quota for one `ml.g6.xlarge` or `ml.g5.xlarge` real-time endpoint.
- Outbound access from the container to Hugging Face. No Hugging Face token is needed for
  these public artifacts.
- CloudWatch log-group deletion permission for log cleanup.

The example was first validated in `us-west-2`, and on the serving image in `eu-west-1`
(`ml.g6.xlarge`) and `us-east-1` (`ml.g5.xlarge`). The notebook uses your configured AWS
Region, falling back to `us-west-2` when none is configured. In another commercial AWS
Region, verify instance availability. The notebook accepts `SAGEMAKER_INFERENCE_IMAGE_URI`,
`SAGEMAKER_EXECUTION_ROLE_ARN`, `SAGEMAKER_ARTIFACT_BUCKET`, `CODEBUILD_ROLE_ARN`, and
`SAGEMAKER_INSTANCE_TYPES` (priority order, default `ml.g6.xlarge,ml.g5.xlarge`; with more
than one, the endpoint uses a
[capacity-aware instance pool](../../../03-features/capacity-aware-instance-pool/),
available in 16 commercial Regions) overrides. `SAGEMAKER_INFERENCE_AMI_VERSION` sets the
inference AMI (default `al2023-ami-sagemaker-inference-gpu-4-1`). Do not use the default
AMIs of `ml.g5` and `ml.g4dn` (NVIDIA driver 470): they cannot start the CUDA 13 image.

## Run the example

1. Open [deploy_decision_sol_2b_sagemaker.ipynb](deploy_decision_sol_2b_sagemaker.ipynb)
   from this directory.
2. Run the setup, packaging, deployment, and inference cells in order.
3. Inspect typed answers, runtime diagnostics, and bounded scaling measurements.
4. Run cleanup, including after a failed deployment once SageMaker permits deletion.

The notebook resolves account and Region at runtime and uses unique resource
names. It preserves the existing IAM role and shared artifact bucket. It uses
`boto3` directly and does not require SageMaker Python SDK v2.

## Files

| File | Purpose |
|---|---|
| `deploy_decision_sol_2b_sagemaker.ipynb` | Setup, package, deploy, invoke, validate, and clean up |
| `inference.py` | Native `AutoModel` / `system_one` adapter for SageMaker |
| `requirements.txt` | Extra Python packages for the sample; empty by default because the serving image provides the libraries |
| `example-request.json` | Grounding, next-action, and readiness questions |

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one instance: `ml.g6.xlarge`, then `ml.g5.xlarge` |
| Hardware | One NVIDIA L4 or A10G; 16 GiB host RAM |
| Image | Shared serving image on `pytorch:2.14-cu133-amzn2023-sagemaker` (PyTorch 2.14, CUDA 13) |
| Inference AMI | `al2023-ami-sagemaker-inference-gpu-4-1` (NVIDIA driver 580, CUDA 13.0) |
| Model | `vllm-sr/Decision-2.0-Sol-2B` |
| Checkpoint and code revision | `64235bef55dad29387dd16da7c90e038bf2f0972` |
| Loaded parameters | 1,883,930,944 |
| Published context budget | 16,384 tokens |
| Worker | One process; native inference serialized |
| Precision | Native BF16-resident eligible Linear weights; FP32 head |
| Startup timeout | 1,800 seconds for the SageMaker container health check |

The published loader first constructs FP32 weights on the CPU and then reduces
eligible Linear weights to BF16 before moving to the GPU. The FP32 parameter
copy alone is about 7.0 GiB; diagnostics report the worker's measured high-water
resident memory. GPU peak allocated memory is reported for the most recent
inference. These measurements do not include every process on the instance.

The model requires `trust_remote_code=True`. The example pins both weights and
code to one revision; review that published code before adopting it.

## Inference and validation

Invoke through SageMaker's IAM-authenticated `InvokeEndpoint` API with JSON
containing `state` and named `questions`. The handler calls `model.system_one`
and preserves native answer fields. `{"operation": "diagnostics"}` returns
runtime metadata and memory measurements.

The notebook checks all three answer types, three basic expected answers,
repeatability, and rejection of an oversized state. It also measures three
requests each with 1, 8, 32, and 64 Yes/No questions over a short state, reporting
server and SDK round-trip latency separately.

These are hosting smoke checks, not a representative accuracy or calibration
benchmark. The surrounding harness owns authorization, decision thresholds,
tool-argument validation, and execution. A short request with many questions
does not establish capacity for long contexts or concurrent clients.

### Observed hosting smoke results

The notebook completed on October 5, 2026 in `us-west-2` on the previous PyTorch 2.6
container, including cleanup.
All three output types, 3/3 basic expected answers, repeatability, and model
HTTP 422 for oversized state passed.

For the short-state Yes/No workload, medians of three requests were:

| Questions | Server latency (ms) | SDK round trip (ms) | GPU peak allocated (GiB) |
|---|---:|---:|---:|
| 1 | 46.76 | 246.99 | 4.49 |
| 8 | 85.56 | 276.14 | 4.62 |
| 32 | 444.25 | 632.82 | 5.06 |
| 64 | 1,046.93 | 1,238.46 | 5.66 |

Worker lifetime peak RSS was 9.07 GiB. SDK timings were measured from the local
Jupyter client and include network latency. These results use the reference
NVIDIA path described below. A separate `us-east-1` attempt failed with
`InsufficientInstanceCapacity` before producing model logs; that is not a model
memory result.

### Serving image results

The same notebook on the serving image (PyTorch 2.14.0, CUDA 13.0, Transformers 5.17.0),
on `ml.g6.xlarge` in `eu-west-1` on October 9, 2026, passed every check, including
model HTTP 422 for oversized state, and cleaned up. Medians of three requests:

| Questions | Server latency (ms) | SDK round trip (ms) | GPU peak allocated (GiB) |
|---|---:|---:|---:|
| 1 | 55.05 | 76.80 | 4.49 |
| 8 | 89.66 | 110.45 | 4.62 |
| 32 | 448.98 | 473.23 | 5.06 |
| 64 | 1,052.08 | 1,077.92 | 5.66 |

Worker lifetime peak RSS was 9.21 GiB. SDK round trips depend on where the client runs;
this one ran on a laptop in London against `eu-west-1`.

On `ml.g5.xlarge` (A10G) in `us-east-1` the notebook also passed, with server medians of
63.79, 104.37, 380.49 and 718.32 ms for 1, 8, 32 and 64 questions, the same GPU peak
allocation, and a worker peak RSS of 8.93 GiB.

## Runtime and performance scope

The default exact runtime batches questions, with the supplied state repeated
in each question's sequence. It can split large requests into several forward
batches. Optional shared-context reuse is disabled by default and may change
answers slightly; this example retains the exact path.

The published accelerated path targets a verified ROCm configuration. NVIDIA
uses eager execution, and this example does not install optional
`flash-linear-attention` or `causal-conv1d` kernels. The measured latencies should
not be compared directly with the authors' optimized GPU or API speedup claims.

The example runs on the shared serving image, built `FROM` the maintained
`pytorch:2.14-cu133-amzn2023-sagemaker` DLC ([image entry](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch/2.14-cuda-sagemaker.yml),
supported until September 28, 2027), with Transformers 5.17.0 pinned. PyTorch is no longer
installed at startup. The previous path used the PyTorch 2.6 inference DLC, which reached
end of support on June 30, 2026.

PyTorch 2.14 is built for CUDA 13, which ships with NVIDIA driver 580. The endpoint sets
`InferenceAmiVersion` to `al2023-ami-sagemaker-inference-gpu-4-1`, which ships that
driver. The default AMI of `ml.g5` and `ml.g4dn` (driver 470) cannot start the image: the
container failed with `CannotStartContainerError` on both. The image is about 10 GB compressed because the base
DLC carries the PyTorch training stack. This is an evaluation sample: review the
dependencies and the image before production use.

## Cleanup

The final cell deletes the endpoint, waits for deletion, and removes its model
and endpoint configuration. It deletes the uploaded S3 object by its version
when versioning is enabled, including older versions and delete markers for
this exact unique key. It removes the local archive and deletes the endpoint
log group. Independent cleanup is attempted even if another operation fails;
failures are reported. Only confirmed missing resources are ignored. The IAM
role and shared bucket are preserved.

Late endpoint log delivery can recreate the CloudWatch group after deletion.
Rerun cleanup after delivery settles if it reappears.

The cleanup cell keeps the serving image, its Amazon ECR repository
(`decision-model-serving`), and the AWS CodeBuild project (`decision-model-serving-build`)
so later runs can reuse them. The image is about 10 GB and ECR bills for its storage.
To remove them:

```bash
aws ecr delete-repository --repository-name decision-model-serving --force
aws codebuild delete-project --name decision-model-serving-build
```

## References

- [Model card and Apache-2.0 license](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B)
- [Pinned model manifest](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B/blob/64235bef55dad29387dd16da7c90e038bf2f0972/MODEL_MANIFEST.json)
- [Pinned Qwen runtime](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B/blob/64235bef55dad29387dd16da7c90e038bf2f0972/decision2/qwen.py)
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
- [AWS G6 specifications](https://aws.amazon.com/ec2/instance-types/g6/)
