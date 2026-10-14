# Deploy Vela 2.0 0.3B on a CPU endpoint on Amazon SageMaker AI

Deploy [Vela 2.0 0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B) to a
real-time endpoint on one `ml.c7i.2xlarge`, with no GPU, using its published
Transformers runtime and native `system_one` API.

Vela 2.0 is the compact multilingual encoder of the vLLM Semantic Router
family: a 22-layer bidirectional ModernBERT with schema-conditioned readouts and
an 8,192-token window. Like the Decision 2.0 models it answers bounded questions
about supplied state in one pass (`choice`, `noul`, `score`), and it adds two
types: `span` returns labelled text with character offsets (personal
information, claims unsupported by a source), and `set` selects any number of
labels with independent probabilities. The state can carry typed parts
(`request`, `source`, `answer`), so one request can check the user's input, the
grounding, and the model's answer together. At 309M FP32 parameters and under
3 GiB of worker memory it is built for CPU hosting, which makes it a fit for
routing, guardrail, and PII-redaction layers in front of a larger model.

The checkpoint, calibration, and bundled engine code are loaded together from a
pinned Hugging Face revision. The sibling
[Decision 2.0 Kai 0.6B CPU](../Decision-2.0-Kai-0.6B-cpu/) example hosts a
Qwen3-based decision model on the same serving image.

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
  instead to reuse a CPU image you already built, for example from the Kai example.
- Quota for one `ml.c7i.2xlarge` real-time endpoint (or the instance you choose).
- Outbound access from the container to Hugging Face at startup. No Hugging Face
  token is needed for these public artifacts.
- CloudWatch log-group deletion permission for log cleanup.

The example was validated in `us-east-1`. The notebook uses your configured AWS
Region, falling back to `us-east-1` when none is configured. The notebook
accepts these overrides:

| Variable | Default | Purpose |
|---|---|---|
| `SAGEMAKER_INSTANCE_TYPE` | `ml.c7i.2xlarge` | Any CPU instance with at least about 4 GiB free for the worker |
| `SAGEMAKER_CPU_THREADS` | every vCPU | Torch intra-op threads in the worker (`VELA_THREADS` in the container) |
| `SAGEMAKER_INFERENCE_IMAGE_URI` | build | Reuse an existing CPU serving image |
| `SAGEMAKER_SERVING_BASE_IMAGE` | PyTorch 2.14 CPU DLC | Base of the serving image build |
| `CODEBUILD_ROLE_ARN`, `SAGEMAKER_EXECUTION_ROLE_ARN`, `SAGEMAKER_ARTIFACT_BUCKET` | | As in the sibling examples |

## Run the example

1. Open [deploy_vela_2_0_0_3b_cpu_sagemaker.ipynb](deploy_vela_2_0_0_3b_cpu_sagemaker.ipynb)
   from this directory.
2. Run the setup, image build, packaging, deployment, and inference cells in order.
3. Inspect typed answers and spans, runtime diagnostics, the windowing check, and
   bounded scaling measurements.
4. Run cleanup, including after a failed deployment once SageMaker permits deletion.

The notebook resolves account and Region at runtime and uses unique resource
names. It preserves the existing IAM role and shared artifact bucket. It uses
`boto3` directly and does not require SageMaker Python SDK v2.

## Files

| File | Purpose |
|---|---|
| `deploy_vela_2_0_0_3b_cpu_sagemaker.ipynb` | Setup, build the CPU image, package, deploy, invoke, validate, and clean up |
| `inference.py` | Native `AutoModel` / `system_one` adapter for SageMaker on CPU, with 413/422 mapping of engine errors |
| `requirements.txt` | Installs nothing; documents what the serving image and the model repository provide |
| `example-request.json` | The model card's quickstart: PII spans over the request, unsupported spans in the answer, and a subject-area route |

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one `ml.c7i.2xlarge` (8 vCPUs, 16 GiB) |
| Hardware | 4th generation Intel Xeon Scalable (Sapphire Rapids); no GPU |
| Image | Shared [decision-model serving image](../../../03-features/decision-model-serving-image/) built on `pytorch:2.14-cpu-amzn2023-sagemaker` |
| Model | `vllm-sr/Vela-2.0-0.3B` |
| Checkpoint and code revision | `d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409` |
| Loaded parameters | 309,114,371 |
| Backbone | ModernBERT, 22 layers, hidden size 768, SDPA attention |
| Input window | 8,192 tokens including schema and state; 512-token overlap when windowing |
| Question types | `choice`, `noul`, `score`, `span`, `set` |
| Worker | One; native inference serialized |
| Precision | FP32 (the model card does not support BF16/FP16 Torch loading) |
| Threads | Torch intra-op threads set to the instance's vCPU count unless `VELA_THREADS` is set |
| Startup timeout | 1,800 seconds for the SageMaker container health check |

Loading goes through Transformers' standard weight loader (`model.safetensors`,
strict); the repository's `vela2_inference.py` engine then handles tokenization,
schema assembly, windowing, calibration, and span decoding. The ONNX exports in
the repository (fp32 and fp16) are not downloaded. The model requires
`trust_remote_code=True`; the example pins weights and code to one revision.
Review the published code before adopting it.

## Inference and validation

Invoke through SageMaker's IAM-authenticated `InvokeEndpoint` API with JSON
containing `state` and named `questions`. The state is a string, a JSON object
whose keys map onto typed parts, or an array. Each question may name the state
field it is asked `over`. The handler calls `model.system_one` and preserves
native fields: `noul` is a probability; `choice.confidence` is normalized from
the top probability; `score` is an expected index over the supplied levels;
`span` questions add `spans` (label, text, code-point `start`/`end`,
probability) and `thresholds`, plus a `noul` view holding the highest span
probability; `set` questions add `sets` (selected labels and per-label
probabilities) and one `noul` view per label named `<question>.<label>`.
`{"operation": "diagnostics"}` returns runtime metadata, the host CPU model,
thread settings, calibration, cumulative windowing counters, and the worker's
peak resident memory.

The engine validates requests and reports the failing location. The adapter
maps an invalid question to HTTP 422 and a schema that alone exceeds the window
to HTTP 413, as the published SystemOne server does. SageMaker wraps both in
`ModelError`; inspect `OriginalStatusCode`.

Long state is **windowed, not rejected**: a targeted part that does not fit is
scored in overlapping 8,192-token windows and the logits are averaged (Set
logits max-pooled). `usage.input_tokens` then exceeds 8,192 and latency grows
by one full encoder pass per window. The notebook demonstrates a two-window
state and the adapter does not cap it; a harness should bound state length so
requests stay inside SageMaker's 60-second invocation limit.

The notebook checks all five answer types, six basic expected answers,
repeatability, both error mappings, windowing, and a scaling run of 1, 8, 32,
and 64 Yes/No questions over a short state with server and SDK round-trip
latency reported separately.

These are hosting smoke checks, not a representative accuracy or calibration
benchmark. The surrounding harness owns authorization, decision thresholds,
redaction, and execution. A short request with many questions does not
establish capacity for long contexts or concurrent clients.

### Ask comparative questions

Question formulation matters more than on a generative model, and Vela's bare
Yes/No answers are conservatively calibrated. In the notebook, "Did the user
explicitly provide a city?" scores 0.35 for a request that names Paris against
0.07 for one that names no city, while the same distinction as a two-way
`choice` with concrete examples in each candidate description scores 0.97
against 0.00. In a separate routing check, a "needs live web data" Yes/No was
right on 10 of 12 probes with a greeting sitting at 0.55, and the same question
as a two-way `choice` (web vs. static, with examples) was right on 17 of 17 at
0.99–1.00 against 0.00–0.03. The same pattern held for Decision 2.0 Kai. Prefer
one comparative `choice`, or a `noul` with explicit `true`/`false` criteria,
over bare Yes/No questions.

### Observed hosting smoke results

The notebook completed on October 14, 2026 in `us-east-1` on one
`ml.c7i.2xlarge` (Intel Xeon Platinum 8488C, 8 vCPUs, 8 Torch threads, Torch
2.14.0+cpu, Transformers 5.17.0, tokenizers 0.23.3, FP32), including cleanup.
All five output types, 6/6 basic expected answers, repeatability, the HTTP 422
and 413 mappings, and windowing passed. The example request returned `domain =
health` (p = 0.727), PII spans `Tom Baker` (PERSON, 1.000) and
`tom.baker@example.com` (EMAIL_ADDRESS, 0.983), and the unsupported span
`up to 6 grams` (0.895), identical to a local FP32 run on a different CPU
architecture. Its median warm server latency was 299.65 ms for two sequences;
the Set/Yes-No/Score ticket took 66.97 ms. The two-way Choice separated the city
cases at P(named) 0.974 against 0.000, the bare Yes/No at 0.351 against 0.067.
Worker peak RSS after startup was 1.31 GiB.

For the short-state Yes/No workload, medians of three requests were:

| Questions | Input tokens | Server latency (ms) | SDK round trip (ms) | Worker peak RSS (GiB) |
|---|---:|---:|---:|---:|
| 1 | 49 | 46.44 | 57.56 | 2.63 |
| 8 | 294 | 146.80 | 159.07 | 2.63 |
| 32 | 1,134 | 601.57 | 613.62 | 2.63 |
| 64 | 2,254 | 1,496.23 | 1,509.11 | 2.63 |

Worker RSS is the lifetime high-water mark; the windowed request set it. SDK
timings were measured from a SageMaker Studio client in the same Region, about
11 ms above the server figures. Non-span questions share one sequence, so
latency grows with the schema rather than with a repeated state: 64 questions
cost about 32 times one question, and the same 64-question request on
[Decision 2.0 Kai 0.6B](../Decision-2.0-Kai-0.6B-cpu/) on the same instance
took 8,731 ms, about six times longer.

The two-window state (9,663 input tokens, 2 windows) took **27.32 s**
server-side on this instance. That is inside SageMaker's 60-second invocation
limit, but one such request costs about 90 example requests and a slightly
longer state would add a third window. Bound state length in the harness.

A second run the same day on `ml.c7i.4xlarge` (16 vCPUs, 16 threads,
including cleanup) returned identical answers. Server-latency medians:

| Request | c7i.2xlarge, 8 threads | c7i.4xlarge, 16 threads |
|---|---:|---:|
| Example request (2 sequences) | 299.65 | 165.67 |
| Ticket (Set, Yes/No, Score) | 66.97 | 48.53 |
| 1 question | 46.44 | 27.57 |
| 8 questions | 146.80 | 79.69 |
| 32 questions | 601.57 | 315.78 |
| 64 questions | 1,496.23 | 766.33 |
| Two-window state (9,663 tokens) | 27,318 | 14,822 |

Doubling the vCPUs and threads halved every latency except the smallest
requests, at roughly twice the hourly price, so cost per request is about
equal and the choice is about latency and headroom for long inputs.

## Runtime and performance scope

The engine runs the Torch backend in FP32 with SDPA attention and the
calibration shipped in `calibration.json` (one temperature per question type;
Span and Set thresholds, including PII length buckets). Each sequence carries
all non-span questions and at most one Span question, so a request with two
Span questions runs two sequences. The authors report ONNX fp32 and fp16 CPU
paths as faster alternatives; this example keeps the Torch path the remote code
exposes and does not install `onnxruntime`.

Thread count is the main CPU tuning knob. The adapter sets Torch intra-op
threads to the instance's vCPU count; `SAGEMAKER_CPU_THREADS` sets a different
value, for example the physical-core count on an SMT instance. The diagnostics
operation reports the effective `torch_threads` and the processor model.

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
(`decision-model-serving-build`) so later runs, including the other
decision-model examples, can reuse them. To remove them:

```bash
aws ecr delete-repository --repository-name decision-model-serving --force
aws codebuild delete-project --name decision-model-serving-build
```

Late endpoint log delivery can recreate the CloudWatch group after deletion.
Rerun cleanup after delivery settles if it reappears.

## Licensing

The model's weights, code, and documentation are Apache-2.0. It is derived from
Decision-1.0-Kai-0.6B (Apache-2.0) and Vela-1.0-Encoder-307M (MIT, from
mmBERT); the tokenizer carries the Gemma Terms of Use, and some training data
are CC-BY-SA. Read `LICENSE`, `NOTICE`, `DISTRIBUTION_TERMS.md`, and
`LICENSING_STATUS.md` in the model repository before redistributing.

## References

- [Model card](https://huggingface.co/vllm-sr/Vela-2.0-0.3B)
- [Usage: SystemOne request and response, Span and Set, typed parts, limits](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409/USAGE.md)
- [Pinned engine](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409/vela2_inference.py)
- [Evaluation and export parity](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409/EVALUATION.md)
- [Vela 2.0 technical overview](https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/)
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
- [AWS C7i specifications](https://aws.amazon.com/ec2/instance-types/c7i/)
- [Decision-model serving image](../../../03-features/decision-model-serving-image/)
