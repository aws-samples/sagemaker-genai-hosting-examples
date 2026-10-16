# Deploy Decision 3.0 (d3-mini) on Amazon SageMaker AI

Deploy [d3-mini](https://huggingface.co/vllm-sr/d3-mini), the 4B multimodal
foundation decision model of Decision 3.0, to a real-time endpoint on one
NVIDIA L4 instance (`ml.g6.2xlarge`, with `ml.g6.4xlarge` and `ml.g6e.2xlarge`
as capacity fallbacks), using its published Transformers runtime and native
`system_one` API. The same notebook deploys
[d3-nano](https://huggingface.co/vllm-sr/d3-nano) (2B) and
[d3-flash](https://huggingface.co/vllm-sr/d3-flash) (9B).

Decision 3.0 keeps the Decision 2.0 contract (Choice, Yes/No `noul`, and Score
answers over supplied state, one probability per option, no text generation) and
adds **images and videos** as inputs: a request carries any number of each, and
every question of the request sees them. The models are Qwen3.5 fine-tunes with
a 255-way answer-code readout: one forward pass per question, `batch_size`
questions per pass, non-causal full attention over the prompt. Typical uses are
routing and policy checks that must look at an attachment, a screenshot, or a
clip as well as the text.

The checkpoint, readout, and bundled runtime code are loaded together from a
pinned Hugging Face revision and verified against `MODEL_MANIFEST.json`. The
sibling [Decision 2.0 Sol 2B](../Decision-2.0-Sol-2B/) example hosts the
previous generation on a GPU; [Kai 0.6B](../Decision-2.0-Kai-0.6B-cpu/) and
[Vela 2.0 0.3B](../Vela-2.0-0.3B-cpu/) show the CPU path.

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
  instead to reuse the CUDA image you already built, for example from the Strands
  Decider example.
- Quota for the `ml.g6` (and, for d3-flash, `ml.g6e`) real-time endpoint instances
  you choose.
- Outbound access from the container to PyPI (the kernels and OpenCV) and Hugging
  Face at startup. No Hugging Face token is needed for these public artifacts.
- CloudWatch log-group deletion permission for log cleanup.
- The notebook environment needs `huggingface_hub` to fetch the two small example
  assets (a 66 KB receipt PNG and a 26 KB clip); the first cell installs it.

The example was validated in `us-east-1`. The notebook uses your configured AWS
Region, falling back to `us-east-1` when none is configured. The notebook accepts
these overrides:

| Variable | Default | Purpose |
|---|---|---|
| `DECISION_MODEL` | `mini` | `nano` (2B), `mini` (4B) or `flash` (9B) |
| `SAGEMAKER_INSTANCE_TYPES` | `ml.g6.2xlarge,ml.g6.4xlarge,ml.g6e.2xlarge` (`ml.g6e.2xlarge` for flash) | Comma-separated priority list, no brackets or quotes; more than one creates a capacity-aware instance pool |
| `SAGEMAKER_INFERENCE_AMI_VERSION` | `al2023-ami-sagemaker-inference-gpu-4-1` | NVIDIA driver 580 AMI for the CUDA 13 image |
| `D3_BATCH_SIZE` | `8` | Questions per forward pass |
| `D3_WARMUP` | `1` | Compile the kernels and warm the vision tower at startup |
| `SAGEMAKER_INFERENCE_IMAGE_URI` | build | Reuse an existing CUDA serving image |
| `CODEBUILD_ROLE_ARN`, `SAGEMAKER_EXECUTION_ROLE_ARN`, `SAGEMAKER_ARTIFACT_BUCKET` | | As in the sibling examples |

An `ml.g6` instance (one NVIDIA L4, 24 GB) holds d3-nano and d3-mini with images
and videos: d3-mini peaked at 9.66 GiB of GPU memory in the validation run,
including the warm-up's four 1.6-megapixel images. d3-flash, whose BF16 weights
alone are 16.8 GB, needs the L40S (48 GB) of `ml.g6e`, which is also the fallback
for the smaller models when `ml.g6` capacity is short. The capacity-aware pool is
available in 16 commercial Regions; elsewhere set a single type.

## Run the example

1. Open [deploy_d3_mini_sagemaker.ipynb](deploy_d3_mini_sagemaker.ipynb)
   from this directory.
2. Run the setup, image build, packaging, deployment, and inference cells in order.
3. Inspect typed answers over text, an image, and a video; runtime diagnostics;
   and bounded scaling measurements.
4. Run cleanup, including after a failed deployment once SageMaker permits deletion.

The notebook resolves account and Region at runtime and uses unique resource
names. It preserves the existing IAM role and shared artifact bucket. It uses
`boto3` directly and does not require SageMaker Python SDK v2.

## Files

| File | Purpose |
|---|---|
| `deploy_d3_mini_sagemaker.ipynb` | Setup, build the image, package, deploy, invoke (text, image, video), validate, and clean up |
| `inference.py` | Native `AutoModel` / d3 runtime adapter for SageMaker, with the package server's strict media handling and 422 mapping |
| `requirements.txt` | Adds the optional `flash-linear-attention` kernels and OpenCV for video decoding; everything else comes from the image |
| `example-request.json` | Support-ticket routing, receipt, and urgency questions (the model card's text example; identical to the Kai example's request) |

## Serving configuration

| Setting | Value |
|---|---|
| Endpoint | Real-time, one instance from the pool `ml.g6.2xlarge`, `ml.g6.4xlarge`, `ml.g6e.2xlarge` (validated on `ml.g6.4xlarge`) |
| Hardware | One NVIDIA L4 (24 GB) on `ml.g6`; one L40S (48 GB) on `ml.g6e` |
| Image | Shared [decision-model serving image](../../../03-features/decision-model-serving-image/) on `pytorch:2.14-cu133-amzn2023-sagemaker` |
| Inference AMI | `al2023-ami-sagemaker-inference-gpu-4-1` (NVIDIA driver 580, CUDA 13.0) |
| Model | `vllm-sr/d3-mini` (or `vllm-sr/d3-nano`, `vllm-sr/d3-flash`) |
| Checkpoint and code revision | mini `f15a639c070dccd1da611d83c0618c77be8c23d6`; nano `7bdafa1db05271702efdd4fa2bd2f66a03680763`; flash `c7b9fa351ffe8d17f1f97ecacf1e36a9256bad8c` |
| Loaded parameters | mini 4,539,918,336; nano 2,213,763,904; flash 8,393,739,504 (text + vision tower + readout) |
| Base model | Qwen/Qwen3.5-4B, -2B, -9B |
| Input limit | None declared (`max_input_tokens` null); 262,144 positions |
| Images | Any number per request; 65,536 to 1,638,400 pixels each after resizing; PNG, JPEG, WebP data URLs up to 8 MB |
| Videos | Any number per request; 2 frames/s, 4 to 32 frames, up to 0.2 MP per frame, 16,384 tokens per request; MP4, WebM, QuickTime, Matroska data URLs up to 32 MB |
| Worker | One; native inference serialized |
| Precision | BF16 backbone loaded straight to the GPU; FP32 readout (fixed by the runtime) |
| Batch | 8 questions per forward pass |
| Startup timeout | 1,800 seconds for the SageMaker container health check |

The runtime verifies the package against `MODEL_MANIFEST.json` before loading,
loads the backbone in BF16 directly onto the GPU (no FP32 staging on the host),
and rejects `dtype` overrides and dtype casts. With `D3_WARMUP=1` (the default)
it compiles and autotunes the Gated DeltaNet kernels for every batch size and
runs three image requests before the endpoint reports healthy, so the first real
request does not pay for that. The package's ROCm fast path (`d3_fast.py`)
disables itself on CUDA. The model requires `trust_remote_code=True`; the
example pins weights and code to one revision. Review the published code before
adopting it.

### Gated DeltaNet kernels

Qwen3.5 backbones alternate linear-attention (Gated DeltaNet) and full-attention
layers. Transformers binds the Gated DeltaNet operations to the
`flash-linear-attention` kernels when that package is installed and to its
PyTorch reference implementation otherwise, which the authors describe as
correct but much slower. `requirements.txt` installs `flash-linear-attention==0.5.2`
and `einops` at startup, as the model's own requirements do; both leave the
image's Transformers pin untouched. The diagnostics operation reports which
implementation is bound under `kernels`, so a result can be attributed to a
configuration. The optional `causal-conv1d` kernel is not installed, so that
operation runs the PyTorch reference. `requirements.txt` also installs
`opencv-python-headless`, which the runtime needs to decode video files and
which the DLC does not ship; without it, video requests are refused with HTTP
422 while text and image requests work.

## Inference and validation

Invoke through SageMaker's IAM-authenticated `InvokeEndpoint` API with JSON
containing `questions` and optionally `state`, `images`, and `videos`; images
and videos are lists of base64 data URLs. The adapter runs the runtime's
`prepare`, `run`, and `respond` steps as the package's own `d3_server.py` does
and preserves native answer fields: `noul` is a probability; `choice.confidence`
is normalized from the top probability; `score` is an expected index over the
supplied levels. Each response also carries `latency_ms` and the request's
`gpu_peak_allocated_bytes`. `{"operation": "diagnostics"}` returns runtime
metadata, the GPU, kernel bindings, image and video contracts, load and warm-up
times, and memory use.

Malformed requests are refused before anything runs with HTTP 422 and a `detail`
body: an unknown question type, an unknown field, or media that is not a valid
data URL. **Images and videos are accepted only as data URLs**, never as file
paths or web addresses, so the endpoint does not read its own disk or fetch from
the network on a caller's behalf. SageMaker wraps the error in `ModelError`;
inspect `OriginalStatusCode`. SageMaker's `InvokeEndpoint` payload limit (6 MB)
bounds what can be sent inline; larger media need a different ingestion path.

The checkpoint declares no input limit, so a long state is answered in full
rather than rejected (the Decision 2.0 behavior) or truncated. The notebook
sends a 15,000-token state and reports its latency and GPU peak. Bound state
length in the harness to stay inside the 60-second invocation limit.

The notebook checks all three answer types over text, an image, and a video;
nine basic expected answers; repeatability per modality; four error mappings;
the long state; and a scaling run of 1, 8, 32, and 64 Yes/No questions over a
short state with server and SDK round-trip latency reported separately.

These are hosting smoke checks, not a representative accuracy or calibration
benchmark. The surrounding harness owns authorization, decision thresholds,
tool-argument validation, and execution. A short request with many questions
does not establish capacity for long contexts, large media, or concurrent
clients.

### Observed hosting smoke results

The notebook completed on October 14, 2026 in `us-east-1` for **d3-mini** and
**d3-nano**, each on one `ml.g6.4xlarge` (NVIDIA L4; Torch 2.14.0+cu130, CUDA
13.0, Transformers 5.17.0, flash-linear-attention 0.5.2 bound for
`chunk_gated_delta_rule`, PyTorch reference for `causal_conv1d`, OpenCV 4.14),
including cleanup. The pool placed both endpoints on the L4 because no `ml.g6e`
capacity was available at the time; d3-flash was not run. For both models, all
three answer types over text, an image, and a video, 9/9 basic expected answers,
repeatability per modality, and the four HTTP 422 mappings passed.

| | d3-mini | d3-nano |
|---|---:|---:|
| Load / warm-up (s) | 2.5 / 85 | 1.9 / 55 |
| GPU resident after startup (GiB) | 8.47 | 4.13 |
| GPU peak after startup (GiB) | 9.66 | 5.32 |
| Worker peak RSS (GiB) | 9.73 | 5.34 |

Both models gave the same decisions. d3-mini: text `route = returns`
(p = 0.999), `receipt = 0.999`, `urgency = 1.84` on the 0–2 scale; image
`route = returns`, `on_receipt = 0.993`, `payment = card` (p = 0.9998); video
`direction = right` (p = 0.992), `color_change = 0.959`. d3-nano: `returns`
(0.998), `receipt = 0.9998`, `urgency = 1.78`; `returns`, `on_receipt = 0.999`,
`card` (0.9994); `right` (0.987), `color_change = 0.973`, identical to a CPU run
of the same model. Median warm server latencies over three repeats:

| Request | Input tokens | d3-mini (ms) | d3-nano (ms) | GPU peak mini / nano (GiB) |
|---|---:|---:|---:|---:|
| Text, 3 questions | 345 | 81.65 | 40.99 | 8.52 / 4.17 |
| Image (receipt PNG), 3 questions | 2,938 | 1,049.86 | 628.76 | 8.96 / 4.62 |
| Video (test clip), 2 questions | 1,720 | 456.68 | 230.83 | 8.71 / 4.28 |
| Text, 15,092 tokens, 1 question | 15,092 | 12,711 | 9,711 | 9.66 / 5.22 |

For the short-state Yes/No workload, medians of three requests were:

| Questions | Input tokens | d3-mini server (ms) | d3-mini round trip (ms) | d3-nano server (ms) | d3-nano round trip (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 102 | 52.45 | 61.22 | 40.35 | 50.25 |
| 8 | 816 | 172.60 | 180.93 | 62.31 | 70.58 |
| 32 | 3,264 | 709.23 | 718.75 | 258.78 | 267.50 |
| 64 | 6,528 | 1,419.69 | 1,429.23 | 524.64 | 535.19 |

GPU peak during the scaling run was 8.49–8.58 GiB for d3-mini and 4.15–4.20 GiB
for d3-nano. SDK timings were measured from a SageMaker Studio client in the same
Region, about 9 ms above the server figures. One question is one forward pass,
run 8 per batch, so latency grows linearly with the question count once a batch
is full; d3-nano answers 8 questions in little more than one (62 vs 40 ms) and
is about 2.7 times faster than d3-mini on the batched requests. Against the
Decision 2.0 Kai 0.6B CPU example on the same 64-question request (8,731 ms on
`ml.c7i.2xlarge`), d3-mini on an L4 is about six times faster and d3-nano about
seventeen. The 15,000-token state took 12.7 s (mini) and 9.7 s (nano), within
the 60-second invocation limit but a reminder to bound state length in the
harness.

## Runtime and performance scope

One forward pass decides one question; a request's questions run in order,
`batch_size` per pass. Images are read by the checkpoint's own processor at up
to 1.6 megapixels (one token per 32 × 32 pixels) and placed before the text;
videos follow the images at 2 frames per second. The authors' published figures
(18 ms per text question, 112 ms per image, 336 ms per 10-second video, one
request at a time on an AMD Instinct MI325X with their ROCm fast path) should
not be compared with CUDA latencies measured here; the fast path is off on
NVIDIA, and the kernel binding differs.

The example runs on the shared serving image, built `FROM` the maintained
`pytorch:2.14-cu133-amzn2023-sagemaker` DLC
([image entry](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch/2.14-cuda-sagemaker.yml),
supported until September 28, 2027), with Transformers 5.17.0 pinned. PyTorch
2.14 is built for CUDA 13, which needs NVIDIA driver 580; the endpoint sets
`InferenceAmiVersion` accordingly. This is an evaluation sample; production
packaging requires a dependency review of the image, the kernels, and the
model's remote code.

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

## References

- [d3-mini model card and Apache-2.0 license](https://huggingface.co/vllm-sr/d3-mini)
- [d3-nano](https://huggingface.co/vllm-sr/d3-nano) and [d3-flash](https://huggingface.co/vllm-sr/d3-flash) model cards
- [Pinned d3-mini runtime](https://huggingface.co/vllm-sr/d3-mini/blob/f15a639c070dccd1da611d83c0618c77be8c23d6/d3_runtime.py) and [package server](https://huggingface.co/vllm-sr/d3-mini/blob/f15a639c070dccd1da611d83c0618c77be8c23d6/d3_server.py)
- [Pinned d3-mini manifest](https://huggingface.co/vllm-sr/d3-mini/blob/f15a639c070dccd1da611d83c0618c77be8c23d6/MODEL_MANIFEST.json)
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [AWS G6 specifications](https://aws.amazon.com/ec2/instance-types/g6/) and [G6e specifications](https://aws.amazon.com/ec2/instance-types/g6e/)
- [Decision-model serving image](../../../03-features/decision-model-serving-image/)
