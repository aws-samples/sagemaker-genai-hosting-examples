# Deploy Decision 2.0 Kai 0.6B on a CPU endpoint on Amazon SageMaker AI

Deploy [Decision 2.0 Kai 0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B)
to a real-time endpoint on one `ml.c7i.2xlarge`, with no GPU, using its published
Transformers runtime and native `system_one` API. The same notebook deploys
[Decision 2.0 Eos 0.8B](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B)
when a longer (16,384-token) context is needed.

Decision models are not chat or text-generation models. They answer bounded
questions about supplied state in one forward pass and return typed probability
distributions: `choice` picks one named candidate, `noul` returns P(true), and
`score` rates the state on an ordered rubric. That replaces "prompt an LLM to
classify or route" for routing, gating, scoring, and policy checks. Kai 0.6B
loads as about 2.2 GiB of FP32 weights (about 3 GiB of worker memory with
Torch), so the decision layer fits on a
general-purpose CPU instance instead of a GPU endpoint. One use that motivated
this example is a model-tier router: a gateway asks Kai which tier should handle
a request and whether it needs live web data, then rewrites the target.

The checkpoint, head, and bundled inference code are loaded together from a
pinned Hugging Face revision. The sibling
[Decision 2.0 Sol 2B](../Decision-2.0-Sol-2B/) example hosts the 2B model of
the same family on a GPU.

## Prerequisites

- Python 3.10+ in Jupyter, such as SageMaker Studio.
- AWS credentials with SageMaker hosting, S3 artifact and object-version listing,
  `iam:PassRole`, and execution-role inspection permissions.
- An existing execution role that trusts `sagemaker.amazonaws.com`, can read the
  artifact bucket, pull the serving image from your Amazon ECR, and write
  endpoint logs. Outside Studio, set `SAGEMAKER_EXECUTION_ROLE_ARN`.
- An AWS CodeBuild service role for the one-time image build; set
  `CODEBUILD_ROLE_ARN`. See the
  [serving image README](../../../03-features/decision-model-serving-image/README.md#build)
  for its permissions. The notebook's caller also needs permission to create the
  ECR repository and the CodeBuild project. Set `SAGEMAKER_INFERENCE_IMAGE_URI`
  instead to reuse a CPU image you already built.
- Quota for one `ml.c7i.2xlarge` real-time endpoint (or the instance you choose).
- Outbound access from the container to Hugging Face at startup. No Hugging Face
  token is needed for these public artifacts.
- CloudWatch log-group deletion permission for log cleanup.

The example was validated in `us-east-1`. The notebook uses your configured AWS
Region, falling back to `us-east-1` when none is configured. CPU instances do not
have the capacity constraints of `ml.g6`.
The notebook accepts these overrides:

| Variable | Default | Purpose |
|---|---|---|
| `DECISION_MODEL` | `kai` | `kai` (0.6B, 8,192 tokens) or `eos` (0.8B, 16,384 tokens) |
| `SAGEMAKER_INSTANCE_TYPE` | `ml.c7i.2xlarge` | Any CPU instance with at least about 4 GiB free for the worker |
| `SAGEMAKER_CPU_THREADS` | every vCPU | Torch intra-op threads in the worker |
| `SAGEMAKER_INFERENCE_IMAGE_URI` | build | Reuse an existing CPU serving image |
| `SAGEMAKER_SERVING_BASE_IMAGE` | PyTorch 2.14 CPU DLC | Base of the serving image build |
| `CODEBUILD_ROLE_ARN`, `SAGEMAKER_EXECUTION_ROLE_ARN`, `SAGEMAKER_ARTIFACT_BUCKET` | | As in the sibling examples |

## Run the example

1. Open [deploy_decision_kai_0_6b_cpu_sagemaker.ipynb](deploy_decision_kai_0_6b_cpu_sagemaker.ipynb)
   from this directory.
2. Run the setup, image build, packaging, deployment, and inference cells in order.
3. Inspect typed answers, runtime diagnostics, and bounded scaling measurements.
4. Run cleanup, including after a failed deployment once SageMaker permits deletion.

The notebook resolves account and Region at runtime and uses unique resource
names. It preserves the existing IAM role and shared artifact bucket. It uses
`boto3` directly and does not require SageMaker Python SDK v2.

## Files

| File | Purpose |
|---|---|
| `deploy_decision_kai_0_6b_cpu_sagemaker.ipynb` | Setup, build the CPU image, package, deploy, invoke, validate, and clean up |
| `inference.py` | Native `AutoModel` / `system_one` adapter for SageMaker on CPU |
| `requirements.txt` | Installs nothing; documents what the serving image and the model repository provide |
| `example-request.json` | Support-ticket routing, receipt, and urgency questions (the model card's own example) |

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one `ml.c7i.2xlarge` (8 vCPUs, 16 GiB) |
| Hardware | 4th generation Intel Xeon Scalable (Sapphire Rapids); no GPU |
| Image | Shared [decision-model serving image](../../../03-features/decision-model-serving-image/) built on `pytorch:2.14-cpu-amzn2023-sagemaker` |
| Model | `vllm-sr/Decision-2.0-Kai-0.6B` (or `vllm-sr/Decision-2.0-Eos-0.8B`) |
| Checkpoint and code revision | Kai `cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b`; Eos `52cad37ef21f598302fe964cbfa164739cb965d5` |
| Loaded parameters | Kai 597,103,104; Eos 753,446,208 |
| Published context budget | Kai 8,192 tokens; Eos 16,384 tokens |
| Worker | One; native inference serialized |
| Precision | FP32 throughout (fixed by the runtime on CPU) |
| Threads | Torch intra-op threads set to the instance's vCPU count unless `DECISION_THREADS` is set |
| Startup timeout | 1,800 seconds for the SageMaker container health check |

On CPU the published runtime keeps every parameter in FP32 and runs each
request as one padded forward batch. It rejects `dtype`/`torch_dtype`
overrides and dtype casts, so this example does not quantize or cast the
weights: BF16 or INT8 inference would mean modifying the module behind the
runtime's verified numerics, and is out of scope here.

The model requires `trust_remote_code=True`. The remote code copies the
repository's `decision2/` runtime, verifies every file against
`MODEL_MANIFEST.json`, and refuses partial downloads. The example pins both
weights and code to one revision; review that published code before adopting
it. Two startup warnings are expected and harmless: Transformers reports that a
model of type `decision2` instantiates a model of an empty type, and the
tokenizer loader mentions `fix_mistral_regex`.

## Inference and validation

Invoke through SageMaker's IAM-authenticated `InvokeEndpoint` API with JSON
containing `state` and named `questions`. The handler calls `model.system_one`
and preserves native answer fields: `noul` is a probability with no separate
confidence; `choice.confidence` is normalized from the top probability and is
not identical to it; `score` returns an expected index over the supplied
levels. `{"operation": "diagnostics"}` returns runtime metadata, the host CPU
model, thread settings, and the worker's peak resident memory.

The notebook checks all three answer types, three basic expected answers,
repeatability, and rejection of an oversized state. It also measures three
requests each with 1, 8, 32, and 64 Yes/No questions over a short state,
reporting server and SDK round-trip latency separately.

These are hosting smoke checks, not a representative accuracy or calibration
benchmark. The surrounding harness owns authorization, decision thresholds,
tool-argument validation, and execution. A short request with many questions
does not establish capacity for long contexts or concurrent clients.

Question formulation matters more than on a generative model. In a small
routing check, one comparative `choice` whose candidate descriptions contained
concrete examples outperformed both a vaguely described `choice` and a cascade
of binary `noul` questions, and the same pattern held for
[Vela 2.0 0.3B](../Vela-2.0-0.3B-cpu/), where a "needs live web data" Yes/No
was right on 10 of 12 probes and the same question as a two-way `choice` was
right on 17 of 17. Prefer one well-described comparative `choice` over bare
yes/no questions on both model families.

### Observed hosting smoke results

The notebook completed on October 14, 2026 in `us-east-1` on one
`ml.c7i.2xlarge` (Intel Xeon Platinum 8488C, 8 vCPUs, 8 Torch threads, Torch
2.14.0+cpu, Transformers 5.17.0, FP32), including cleanup. All three output
types, 3/3 basic expected answers, repeatability, and model HTTP 422 for
oversized state passed.
The example request returned `route = returns` (p = 0.962), `receipt = 0.916`,
and `urgency = 1.48` on the 0–2 scale, identical to four decimal places to a
local FP32 run on a different CPU architecture. Median warm server latency for
that three-question request was 658.56 ms. Worker peak RSS after startup was
3.19 GiB.

For the short-state Yes/No workload, medians of three requests were:

| Questions | Server latency (ms) | SDK round trip (ms) | Worker peak RSS (GiB) |
|---|---:|---:|---:|
| 1 | 294.40 | 306.55 | 3.24 |
| 8 | 1,032.55 | 1,044.93 | 3.29 |
| 32 | 3,859.87 | 3,872.25 | 3.51 |
| 64 | 8,730.57 | 8,743.90 | 3.63 |

Worker RSS is the lifetime high-water mark. SDK timings were measured from a
SageMaker Studio client in the same Region, which is why they sit about 12 ms
above the server figures. Latency grows roughly linearly with the number of
questions because each question repeats the state in its own sequence of the
FP32 forward batch; the 64-question request stays well inside the 60-second
invocation limit, but a request that long is a capacity signal, not a target.
A 10-core Apple laptop ran the same workload about 2.4 times faster, so these
numbers are specific to this instance's FP32 CPU path; measure before sizing.

Two further runs the same day on `ml.c7i.4xlarge` (16 vCPUs) with
`SAGEMAKER_CPU_THREADS` set to 4 and to 16 returned identical answers and
isolated the instance size from the thread count. Server-latency medians:

| Questions | c7i.2xlarge, 8 threads | c7i.4xlarge, 4 threads | c7i.4xlarge, 16 threads |
|---|---:|---:|---:|
| Example request (3 questions) | 658.56 | 552.47 | 397.53 |
| 1 | 294.40 | 216.90 | 186.85 |
| 8 | 1,032.55 | 1,096.46 | 611.84 |
| 32 | 3,859.87 | 4,442.68 | 1,924.68 |
| 64 | 8,730.57 | 9,701.91 | 4,417.34 |

Worker peak RSS on the 4xlarge runs was 3.66 and 3.85 GiB. Doubling the vCPUs
and the threads halved the batched latencies, so the FP32 forward scales with
threads for a batch, while a single short question gains only about a third.
Fewer threads than vCPUs shortened tiny requests slightly and lengthened every
batched one; a larger instance without more threads bought nothing. Keep the
default of one thread per vCPU, and pick the instance size by the question
counts you expect per request: at roughly twice the hourly price, the 4xlarge
costs about the same per batched question as the 2xlarge and answers it in
half the time.

## Runtime and performance scope

The default exact runtime batches questions, with the supplied state repeated
in each question's sequence. Optional shared-context reuse is disabled by
default and may change answers slightly; this example retains the exact path.
The published GPU speed figures (about 5 ms per single-question request) use a
verified ROCm configuration and should not be compared with CPU latencies here.

Thread count is the main CPU tuning knob. The adapter passes the instance's
vCPU count to the runtime's `threads` option; `SAGEMAKER_CPU_THREADS` sets a
different value. In the measured runs, batched requests scaled almost
linearly with threads and single questions much less (see the observed results
above). The
diagnostics operation reports the effective `torch_threads` and the processor
model so results can be attributed to a configuration.

The example runs on the shared serving image, built `FROM` the maintained
`pytorch:2.14-cpu-amzn2023-sagemaker` DLC
([image entry](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch/2.14-cpu-sagemaker.yml),
supported until September 28, 2027), with Transformers 5.17.0 pinned. Nothing
is installed at startup except the model download. This is an evaluation
sample; production packaging requires a dependency review of the image and of
the model's remote code.

## Cleanup

The final cell deletes the endpoint, waits for deletion, and removes its model
and endpoint configuration. It deletes the uploaded S3 object by its version
when versioning is enabled, including older versions and delete markers for
this exact unique key. It removes the local archive and deletes the endpoint
log group. Independent cleanup is attempted even if another operation fails;
failures are reported. Only confirmed missing resources are ignored. The IAM
role and shared bucket are preserved.

The cleanup cell keeps the serving image, its Amazon ECR repository
(`decision-model-serving`), and the CodeBuild project
(`decision-model-serving-build`) so later runs, including the Strands Decider
example, can reuse them. To remove them:

```bash
aws ecr delete-repository --repository-name decision-model-serving --force
aws codebuild delete-project --name decision-model-serving-build
```

Late endpoint log delivery can recreate the CloudWatch group after deletion.
Rerun cleanup after delivery settles if it reappears.

## References

- [Kai 0.6B model card and Apache-2.0 license](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B)
- [Eos 0.8B model card](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B)
- [Decision 2.0 collection](https://huggingface.co/collections/vllm-sr/decision-20)
- [Pinned Kai model manifest](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b/MODEL_MANIFEST.json)
- [Pinned Kai Qwen runtime](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b/decision2/qwen.py)
- [Decision models: architecture and the three answer types](https://vllm-sr.ai/blog/decision-models/)
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
- [AWS C7i specifications](https://aws.amazon.com/ec2/instance-types/c7i/)
- [Decision-model serving image](../../../03-features/decision-model-serving-image/)
