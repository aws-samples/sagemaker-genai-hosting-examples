# Deploy Strands Decider 2B on Amazon SageMaker AI

Deploy [Strands Decider 2B](https://github.com/strands-labs/strands-decider) to a
SageMaker AI real-time endpoint on one `ml.g5.xlarge`.

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
  the artifact bucket, pull the AWS inference container, and write inference
  logs. Set `SAGEMAKER_EXECUTION_ROLE_ARN` when the notebook's own role is not a
  SageMaker execution role.
- Quota for one `ml.g5.xlarge` real-time endpoint.
- Outbound access from the container to PyPI, the PyTorch wheel index, GitHub,
  and Hugging Face during startup. The example uses public, ungated artifacts
  and does not require a Hugging Face token.

The example was validated in `us-east-1`. For another commercial AWS Region,
verify the availability of the specified DLC and instance type. The notebook
accepts `SAGEMAKER_INFERENCE_IMAGE_URI` to override the image.

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
| `requirements.txt` | Container dependencies and pinned upstream runtime source |
| `example-request.json` | Tool-call readiness example containing all three question types |
| `deploy_strands_decider_2b_sagemaker_network_isolated.ipynb` | The same deployment with `EnableNetworkIsolation=True`; see [Network-isolated deployment](#network-isolated-deployment) |
| `package_offline.py` | Stages the wheels and pinned model files for the network-isolated notebook |

## Network-isolated deployment

The default notebook lets the container install packages and download weights at
startup, so the endpoint needs outbound access to PyPI, the PyTorch wheel index,
GitHub, and Hugging Face. Where endpoints cannot reach the public internet, or where
only reviewed artifacts may run, use
[deploy_strands_decider_2b_sagemaker_network_isolated.ipynb](deploy_strands_decider_2b_sagemaker_network_isolated.ipynb).

It deploys the same image, adapter, pins, and revisions with these changes:

| | Default | Network-isolated |
|---|---|---|
| Container network | Outbound internet at startup | `EnableNetworkIsolation=True` |
| Python packages | Installed from PyPI and the PyTorch index at startup | Installed with `--no-index` from wheels staged in S3 |
| Decider runtime | Pinned source archive from GitHub; `direct_url.json` checked | Wheel built from the same pinned archive; SHA-256 checked against `code/provenance.json` |
| Checkpoint and base weights | Downloaded from Hugging Face at startup | Staged in S3 as a Hugging Face cache; read with `HF_HUB_OFFLINE=1` |
| Model data | `model.tar.gz` (code only) | Uncompressed S3 prefix (code, about 3 GB of wheels, and 4.6 GB of weights) |

The notebook environment, not the endpoint, needs internet access while staging: it
downloads about 7.6 GB (about 3 GB of wheels and 4.6 GB of weights) and needs that much free disk. The execution role needs read
access to the artifact prefix, as in the default notebook. Network isolation also
blocks the container's own AWS API calls; this adapter makes none.

Network isolation removes the container's runtime downloads. It does not by itself
review the staged packages or weights: scan and approve the staged directory under
your organization's process before upload if that is required.

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one `ml.g5.xlarge` |
| GPU | NVIDIA A10G, BF16 torso |
| Model worker | One; engine access serialized |
| Context | Strict 4,096-token checkpoint window |
| Internal question batch | At most four questions |
| Startup timeouts | 1,800 seconds for SageMaker and the model worker |
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

This evaluation path uses the AWS PyTorch 2.6 inference DLC for SageMaker's
hosting interface and installs Torch 2.7.1 / Transformers 5.17.0 / PEFT 0.21.0
at startup. CUDA 11.8 Torch wheels are used for driver compatibility.

The [AWS image entry](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch-inference/2.6-gpu-sagemaker.yml)
records June 30, 2026 as the base image's end of support. The upgraded runtime
does not extend support for the base image. Use this example for evaluation;
production packaging requires a maintained container and dependency review.

The optional `flash-linear-attention` and `causal-conv1d` kernels are absent.
Inference uses the reference fallbacks, so measured latency is not directly
comparable with the authors' optimized GPU benchmarks. Startup downloads the
runtime source and model weights.

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

## References

- [Official inference instructions](https://github.com/strands-labs/strands-decider/blob/75c9fd32e664954cdc18481434018aa507eee8fb/docs/inference.md)
- [Model card and license](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
- [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- [Official Strands agent intervention example](https://github.com/strands-labs/strands-decider/blob/75c9fd32e664954cdc18481434018aa507eee8fb/examples/strands/README.md)
- [Evaluation results and limitations](https://github.com/strands-labs/strands-decider/blob/75c9fd32e664954cdc18481434018aa507eee8fb/evaluation/results.md)
- [SageMaker PyTorch inference toolkit](https://github.com/aws/sagemaker-pytorch-inference-toolkit)
