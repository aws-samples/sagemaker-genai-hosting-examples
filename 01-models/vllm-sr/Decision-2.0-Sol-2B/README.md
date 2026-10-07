# Deploy Decision 2.0 Sol 2B on Amazon SageMaker AI

Deploy [Decision 2.0 Sol 2B](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B)
to a real-time endpoint on one `ml.g6.xlarge`, using its published Transformers
runtime and native `system_one` API.

The model returns Choice, Yes/No (`noul`), and Score judgments over supplied
state without generating text. Its checkpoint, head, and bundled inference code
are loaded together from a pinned Hugging Face revision.

## Prerequisites

- Python 3.10+ in Jupyter, such as SageMaker Studio.
- AWS credentials with SageMaker hosting, S3 artifact and object-version listing,
  `iam:PassRole`, and
  execution-role inspection permissions.
- An existing execution role that trusts `sagemaker.amazonaws.com`, can read the
  artifact bucket, pull the AWS inference image, and write endpoint logs.
  Outside Studio, set `SAGEMAKER_EXECUTION_ROLE_ARN`.
- Quota for one `ml.g6.xlarge` real-time endpoint.
- Outbound access from the container to PyPI, the PyTorch wheel index, and
  Hugging Face. No Hugging Face token is needed for these public artifacts.
- CloudWatch log-group deletion permission for log cleanup.

The example was validated in `us-west-2`. The notebook uses your configured AWS
Region, falling back to `us-west-2` when none is configured. In another
commercial AWS Region, verify image and instance availability. The notebook accepts
`SAGEMAKER_INFERENCE_IMAGE_URI`, `SAGEMAKER_EXECUTION_ROLE_ARN`, and
`SAGEMAKER_ARTIFACT_BUCKET` overrides.

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
| `requirements.txt` | Container dependencies |
| `example-request.json` | Grounding, next-action, and readiness questions |

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one `ml.g6.xlarge` |
| Hardware | One NVIDIA L4; 16 GiB host RAM |
| Model | `vllm-sr/Decision-2.0-Sol-2B` |
| Checkpoint and code revision | `64235bef55dad29387dd16da7c90e038bf2f0972` |
| Loaded parameters | 1,883,930,944 |
| Published context budget | 16,384 tokens |
| Worker | One; native inference serialized |
| Precision | Native BF16-resident eligible Linear weights; FP32 head |
| Startup timeouts | 1,800 seconds for SageMaker and the model worker |

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

The notebook completed on October 5, 2026 in `us-west-2`, including cleanup.
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

## Runtime and performance scope

The default exact runtime batches questions, with the supplied state repeated
in each question's sequence. It can split large requests into several forward
batches. Optional shared-context reuse is disabled by default and may change
answers slightly; this example retains the exact path.

The published accelerated path targets a verified ROCm configuration. NVIDIA
uses eager execution, and this example does not install optional
`flash-linear-attention` or `causal-conv1d` kernels. The measured latencies should
not be compared directly with the authors' optimized GPU or API speedup claims.

This example uses the AWS PyTorch 2.6 inference DLC and installs Torch 2.7.1 /
Transformers 5.17.0 at startup. CUDA 11.8 wheels support the inference driver.
The [AWS image entry](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch-inference/2.6-gpu-sagemaker.yml)
records June 30, 2026 as the base image's end of support. Upgrading packages does
not extend that support. This is an evaluation sample; production packaging
requires a maintained container and dependency review.

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

## References

- [Model card and Apache-2.0 license](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B)
- [Pinned model manifest](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B/blob/64235bef55dad29387dd16da7c90e038bf2f0972/MODEL_MANIFEST.json)
- [Pinned Qwen runtime](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B/blob/64235bef55dad29387dd16da7c90e038bf2f0972/decision2/qwen.py)
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
- [AWS G6 specifications](https://aws.amazon.com/ec2/instance-types/g6/)
