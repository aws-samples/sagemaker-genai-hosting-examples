# Deploy Strands Decider 2B on Amazon SageMaker AI

Deploy [Strands Decider 2B](https://github.com/strands-labs/strands-decider) to a
SageMaker AI real-time endpoint on one `ml.g5.xlarge`, falling back to `ml.g6.xlarge`
when g5 capacity is unavailable. It runs on the maintained AWS PyTorch 2.14 SageMaker
DLC through the shared
[decision-model serving image](../../../03-features/decision-model-serving-image/), on
the SageMaker inference AMI with NVIDIA driver 580 that the image's CUDA 13 needs.

Decider answers bounded questions about supplied state: `choice` selects a named
option, `noul` returns P(true), and `score` rates state on an ordered rubric.
It loads the published checkpoint's LoRA adapter, pointer head, and calibration
on the Qwen3.5-2B base. It does not use a chat-completion or text-generation
handler.

## Prerequisites

- A Python 3.10+ Jupyter environment, such as SageMaker Studio.
- AWS credentials with SageMaker hosting, S3 artifact, and `iam:PassRole`
  permissions, plus permission to inspect the execution role.
- A SageMaker execution role that trusts `sagemaker.amazonaws.com` and can read
  the artifact bucket, pull the serving image from your Amazon ECR, and write
  inference logs. Set `SAGEMAKER_EXECUTION_ROLE_ARN` when the notebook's own role
  is not a SageMaker execution role.
- An AWS CodeBuild service role for the image build; set `CODEBUILD_ROLE_ARN`. See
  the [serving image README](../../../03-features/decision-model-serving-image/README.md#build)
  for its permissions. The notebook's caller also needs permission to create the
  ECR repository and the CodeBuild project. Set `SAGEMAKER_INFERENCE_IMAGE_URI`
  instead to reuse an image you already built.
- Quota for one `ml.g5.xlarge` or `ml.g6.xlarge` real-time endpoint.
- Outbound access from the container to GitHub, PyPI, and Hugging Face during
  startup (the default notebook). The example uses public, ungated artifacts and does
  not require a Hugging Face token.

The notebooks were validated in `eu-west-1`. `SAGEMAKER_INSTANCE_TYPES` sets the
instance types in priority order (default `ml.g5.xlarge,ml.g6.xlarge`). With more
than one, the endpoint uses a
[capacity-aware instance pool](../../../03-features/capacity-aware-instance-pool/),
available in 16 commercial Regions; in other Regions set a single instance type.
`SAGEMAKER_INFERENCE_AMI_VERSION` sets the inference AMI (default `al2023-ami-sagemaker-inference-gpu-4-1`).
Do not use the default AMIs of `ml.g5` and `ml.g4dn` (NVIDIA driver 470): they cannot start the CUDA 13 image.

## Run the example

1. Open [deploy_strands_decider_2b_sagemaker.ipynb](deploy_strands_decider_2b_sagemaker.ipynb)
   from this directory.
2. Run the setup, packaging, deployment, and inference cells in order.
3. Inspect the diagnostics and typed judgments.
4. Run the cleanup cell, including after a failed deployment once SageMaker
   permits endpoint deletion.

The notebook resolves account and Region at runtime, uses unique resource
names, and preserves an existing execution role and shared artifact bucket.
It uses `boto3` directly, without requiring SageMaker Python SDK v2.

## Files

| File | Purpose |
|---|---|
| `deploy_strands_decider_2b_sagemaker.ipynb` | Setup, package, deploy, invoke, validate, and clean up |
| `inference.py` | SageMaker `model_fn` / `transform_fn` adapter for the official Decider engine |
| `requirements.txt` | The pinned upstream Decider runtime source; the serving image provides PyTorch and the other libraries |
| `example-request.json` | Tool-call readiness example containing all three question types |
| `deploy_strands_decider_2b_sagemaker_network_isolated.ipynb` | The same deployment with `EnableNetworkIsolation=True`; see [Network-isolated deployment](#network-isolated-deployment) |
| `package_offline.py` | Stages the Decider runtime wheel and the pinned model files for the network-isolated notebook |

## Network-isolated deployment

The default notebook lets the container install the Decider runtime and download weights
at startup, so the endpoint needs outbound access to GitHub, PyPI (build requirements
for the Decider runtime), and Hugging Face. Where endpoints cannot reach the public internet, or where
only reviewed artifacts may run, use
[deploy_strands_decider_2b_sagemaker_network_isolated.ipynb](deploy_strands_decider_2b_sagemaker_network_isolated.ipynb).

It deploys the same image, adapter, pins, and revisions with these changes:

| | Default | Network-isolated |
|---|---|---|
| Container network | Outbound internet at startup | `EnableNetworkIsolation=True` |
| Python packages | Decider runtime installed from GitHub at startup; the rest is in the serving image | Decider runtime and `huggingface_hub` installed with `--no-index` from wheels staged in S3 |
| Decider runtime | Pinned source archive from GitHub; `direct_url.json` checked | Wheel built from the same pinned archive; SHA-256 checked against `code/provenance.json` |
| Checkpoint and base weights | Downloaded from Hugging Face at startup | Staged in S3 as a Hugging Face cache; read with `HF_HUB_OFFLINE=1` |
| Model data | `model.tar.gz` (code only) | Uncompressed S3 prefix (code, two wheels, and 4.6 GB of weights) |

The notebook environment, not the endpoint, needs internet access while staging. It
downloads about 4.6 GB of weights; allow about 5 GB of free disk. The notebook installs
the same `huggingface_hub` version (1.33.0) that the serving image pins, and staging
installs that version in the container, because the offline cache layout differs across
versions.

The execution role needs read access to the artifact prefix, as in the default
notebook. The notebook's caller also needs `s3:ListBucketVersions` and
`s3:DeleteObjectVersion` on the artifact bucket: cleanup deletes every object version
under the example's prefix. Network isolation also blocks the container's own AWS API
calls; this adapter makes none.

Network isolation removes the container's runtime downloads. It does not by itself
review the staged packages or weights: scan and approve the staged directory under
your organization's process before upload if that is required.

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one instance: `ml.g5.xlarge`, then `ml.g6.xlarge` |
| GPU | NVIDIA A10G or L4, BF16 torso |
| Image | Shared serving image on `pytorch:2.14-cu133-amzn2023-sagemaker` (PyTorch 2.14, CUDA 13) |
| Inference AMI | `al2023-ami-sagemaker-inference-gpu-4-1` (NVIDIA driver 580, CUDA 13.0) |
| Model worker | One; engine access serialized |
| Context | Strict 4,096-token checkpoint window |
| Internal question batch | At most four questions |
| Startup timeout | 1,800 seconds for the SageMaker container health check |
| Checkpoint | `StrandsAgents/strands-decider-2B-hobson-v19` |
| Checkpoint revision | `bb282d786bc251fd4e3068de3ada9ddbb38127cd` |
| Runtime source | `75c9fd32e664954cdc18481434018aa507eee8fb` |
| Base revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, recorded in checkpoint provenance |

The published checkpoint records the base revision as inferred at export time;
this example pins that recorded revision for reproducibility.

## Inference

The endpoint accepts the official System One JSON request shape through
SageMaker's IAM-authenticated `InvokeEndpoint` API. See `example-request.json`.
It also accepts `{"operation": "diagnostics"}` to inspect effective package
versions, model revisions, device, and context settings.
The handler verifies the installed runtime's `direct_url.json` provenance
against the pinned source URL before loading the model.

`noul` returns a probability, not a separate confidence field. Choice confidence
is normalized from the top probability; it is not identical to that probability.
Score returns an expected index over the supplied ordered levels. Probabilities
are rounded to four decimal places.

The notebook validates answer structure, three basic expected-answer cases,
repeatability, and rejection of oversized state. SageMaker wraps model-side
HTTP 422 errors in `ModelError`; inspect `OriginalStatusCode`, since it may omit
the model error body.

These are hosting smoke checks. They do not establish decision accuracy,
calibration, or safe autonomous execution on a new workload. The surrounding
application owns validation, authorization, confidence thresholds, and execution.

## Container and performance scope

The example runs on the shared serving image, built `FROM` the maintained
`pytorch:2.14-cu133-amzn2023-sagemaker` DLC ([image entry](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch/2.14-cuda-sagemaker.yml),
supported until September 28, 2027), with Transformers 5.17.0 and PEFT 0.21.0 pinned.
PyTorch is no longer installed at startup. The previous path used the PyTorch 2.6
inference DLC, which reached end of support on June 30, 2026.

PyTorch 2.14 is built for CUDA 13, which ships with NVIDIA driver 580. The endpoint sets
`InferenceAmiVersion` to `al2023-ami-sagemaker-inference-gpu-4-1`, which ships that driver. On `ml.g4dn.xlarge`
the default AMI (driver 470) failed with `CannotStartContainerError`; with this AMI it
started and answered. `ml.g6.xlarge` was also tested. The default `ml.g5` AMI shares the
driver 470 default, so the AMI is set explicitly for every instance type.

In `eu-west-1` on `ml.g6.xlarge`, the example request returns the same answers as on the
2.6 path within 0.004, and median warm server latency is about 150 ms (about 125 ms
before). The image is about 10 GB compressed because the base DLC carries the PyTorch
training stack. This is an evaluation example: review the dependencies and the image
before production use.

The optional `flash-linear-attention` and `causal-conv1d` kernels are absent.
Inference uses the reference fallbacks, so measured latency is not directly
comparable with the authors' optimized GPU benchmarks. Startup downloads the
runtime source and model weights in the default notebook.

## Cleanup

The final notebook cell deletes the endpoint and waits for it to disappear
before deleting its endpoint configuration and model. It then deletes the
example's S3 archive, specifying its uploaded version when S3 versioning is
enabled. Only confirmed missing resources are ignored; permission
and network errors are surfaced. The shared bucket and existing IAM role are
preserved. The cleanup cell also removes this example's CloudWatch endpoint log
group when the caller has permission.
Endpoint log delivery is asynchronous: late log events can recreate the group
after deletion. If it reappears, rerun the cleanup cell after delivery settles.

The cleanup cell keeps the serving image, its Amazon ECR repository
(`decision-model-serving`), and the AWS CodeBuild project (`decision-model-serving-build`)
so later runs can reuse them. The image is about 10 GB and ECR bills for its storage.
To remove them:

```bash
aws ecr delete-repository --repository-name decision-model-serving --force
aws codebuild delete-project --name decision-model-serving-build
```

## References

- [Official inference instructions](https://github.com/strands-labs/strands-decider/blob/75c9fd32e664954cdc18481434018aa507eee8fb/docs/inference.md)
- [Model card and license](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
- [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- [Official Strands agent intervention example](https://github.com/strands-labs/strands-decider/blob/75c9fd32e664954cdc18481434018aa507eee8fb/examples/strands/README.md)
- [Evaluation results and limitations](https://github.com/strands-labs/strands-decider/blob/75c9fd32e664954cdc18481434018aa507eee8fb/evaluation/results.md)
- [SageMaker PyTorch inference toolkit](https://github.com/aws/sagemaker-pytorch-inference-toolkit)
