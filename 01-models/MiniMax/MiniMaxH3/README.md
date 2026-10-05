# Deploy MiniMax-H3 on Amazon SageMaker AI

This example deploys [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — an omni-modal video-and-audio generation system — on a SageMaker **asynchronous** inference endpoint.

## Model

MiniMax-H3 takes text, images, video and audio as context and generates video with **natively synchronized stereo audio** in a single pass. Video and audio latents are jointly predicted by one transformer; the audio is not a separately generated track muxed on afterwards.

| Property | Value |
| :--- | :--- |
| Model | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| H3-Omni-Transformer | 33B dense, single-stream; ~13B in AdaLN branches (not loaded for inference-only) |
| Text encoder | Qwen3-VL-32B, hidden states from layer 50 |
| Visual VAE | f16t4d24, temporally causal, ViT-based decoder; patchify 1×2×2 |
| Audio VAE | 32 kHz, 40 Hz latent, per-channel stereo |
| Positional encoding | 3D MM-RoPE over (t, h, w) |
| Precision | Mixed BF16 / FP32 — FP32 preserved on patch, timestep and output projections |
| Output | One MP4: H.264 video @ 24 fps + AAC stereo @ 32 kHz |
| Duration | 4–15 seconds inclusive |
| Resolution | 768-pixel short edge (2K only via the hosted Regenerate API) |
| Languages | Stable in 11; others to varying degrees |
| License | **MiniMax H3 Community License** — not Apache 2.0 |

Architecture figures come from the model card; serving recipes and all measured numbers below come from the [SGLang MiniMax-H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3).

## Three things that make this unlike the other examples in this repo

### 1. Only the middle third of the system is open

| Module | Role | Open? |
| :--- | :--- | :--- |
| H3-Context-IR | Refines free-form multimodal input into the structured representation H3-Base consumes | **No** — hosted API |
| H3-Base | Generates 768p video + audio | **Yes** — this example |
| H3-Regenerate-2K | In-context regeneration to 2K | **No** — hosted API |

MiniMax describe H3-Context-IR as critical to output quality. A self-hosted H3-Base fed raw user prompts will underperform the hosted product, and the gap is a prompt-engineering gap rather than a serving bug. Either call their Context-IR API or build an equivalent from the [base](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) and [reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) prompting guides. Set that expectation with customers up front.

### 2. Two checkpoint partitions means two endpoints

`--model-variant` is fixed at launch:

| Partition | Serves | Conditioning |
| :--- | :--- | :--- |
| `fl2va` | `t2va`, `fl2va` | none, or first/last keyframes |
| `ref2va` | `ref2va` (includes video-to-video) | images, videos, audio as references |

No single endpoint covers both. The serving shim rejects a mismatched task explicitly rather than generating something wrong.

### 3. Async, not real-time

A single 5-second 768p request is tens of seconds of GPU work; model load alone is ~2 minutes. Async is also the right shape independently: reference media arrives as S3 objects, output is a multi-megabyte MP4 that belongs in S3, and requests queue naturally.

## Instance selection

**`ml.g6e.12xlarge` does not work.** This is the instance the [Wan example](../../Wan/Wan2.1-T2V-1.3B-Diffusers/) uses and the obvious first guess for a video model in this repo.

SGLang's measured peak memory per GPU, lossless BF16/FP32, 1344×768, 124 frames, 50 steps:

| Topology | Pipeline latency | Peak / GPU |
| :--- | ---: | ---: |
| 4× H100 — TP2 + Ulysses2 | 13.25 s | 66.04 GB |
| 4× H100 — FSDP + Ulysses4 | 13.36 s | 57.01 GB |
| 4× H100 — TP4 + Ulysses1 | 13.86 s | **49.80 GB** |

49.80 GB is the most frugal measured Hopper topology, against 48 GB on L40S — short by ~1.8 GB. L40S also has no NVLink, and Ulysses sequence parallelism is all-to-all heavy, so the topology would suffer over PCIe even if memory fit.

| Instance | GPUs | HBM / GPU | Verdict | Verified topology |
| :--- | :--- | ---: | :--- | :--- |
| `ml.g6e.12xlarge` | 4× L40S | 48 GB | **Infeasible** | — |
| `ml.p5.48xlarge` | 8× H100 | 80 GB | Fits | `--num-gpus 4 --tp-size 2 --ulysses-degree 2` |
| `ml.p5e.48xlarge` | 8× H200 | 141 GB | Fits | `--num-gpus 4 --ulysses-degree 4` |
| `ml.p6-b200.48xlarge` | 8× B200 | 180 GB | Fits | `--num-gpus 8 --ulysses-degree 8` |

Topology constraints worth encoding rather than discovering:

- **Ulysses, not Ring.** Ring attention is incompatible with H3's packed multi-segment attention.
- **CFG parallelism is rejected.** The released checkpoints are CFG-distilled and run a single denoising branch; `--enable-cfg-parallel true` errors out.
- **`--vae-config.parallel-decode-mode spatial` / `spatial_shard` are rejected** — validation found output mismatches. Keep the default tiled decode.
- **Don't enable `torch.compile`.** It changes numerical output and its steady-state benefit measured below noise on H200. Never use it to produce consistency ground truth.

## Container

There is no AWS DLC carrying SGLang's diffusion extras today, so `container/` builds one: upstream `lmsysorg/sglang:dev`, plus the diffusion extra (absent from the base image — SGLang's own Docker recipe installs it at launch for the same reason), plus `ffmpeg`, plus a Flask shim providing SageMaker's `/ping` and `/invocations` contract on port 8080.

The shim handles two things that are easy to get wrong:

- **S3 → `file://` rewriting.** SGLang resolves `conditions[].uri` inside its own filesystem, so an `s3://` URI passed straight through fails. The shim downloads reference assets to server-local scratch first.
- **Task/partition validation.** A `ref2va` request sent to an `fl2va` endpoint gets a clear error, not a silently wrong result.

Pin the base image to a digest before production use — `:dev` moves.

## Quality profiles

`quality` is request-scoped, so one resident endpoint switches per request. Measured on 4× H200, 1344×768, 124 frames, 50 steps, averaged over three prompt/seed pairs:

| `quality` | Mean latency | Speedup | SSIM vs lossless | PSNR vs lossless |
| :--- | ---: | ---: | ---: | ---: |
| `lossless` | 75.10 s | 1.00× | 1.000 | exact |
| `high` | 53.70 s | 1.40× | 0.931 | 28.16 dB |
| `medium` | 30.23 s | 2.48× | 0.818 | 20.40 dB |
| `low` | 25.81 s | 2.91× | 0.794 | 19.25 dB |

Two caveats when quoting these:

- SSIM/PSNR measure **same-seed trajectory deviation**, not absolute perceptual quality. An approximate profile can produce a different but equally plausible result.
- Both cover **video only**, while the profiles also change the joint audio-video denoise trajectory. Listen before shipping an approximate profile.

The named profiles are fail-closed to the audited 4× H200 workload — other hardware, task modes, step counts or flow shifts are rejected before denoising rather than silently degrading.

## Notebook contents

| Section | Description |
| :--- | :--- |
| Read this first | Partial open-sourcing, two partitions, async rationale |
| Instance selection | Sizing math and the `g6e` negative result |
| Stage weights | Scoped `snapshot_download` + `aws s3 sync` |
| Build container | ECR repo, build, push |
| Create model | Topology via `SGLANG_ARGS`, no rebuild per instance |
| Async endpoint | `AsyncInferenceConfig`, generous startup timeout |
| `t2va` | Text to video + audio, inline playback |
| `fl2va` | Keyframe conditioning from S3 |
| `ref2va` | Multimodal references (second endpoint) |
| Video to video | A `ref2va` use case, with expectation-setting |
| Quality profiles | Per-request latency/fidelity tradeoff |
| Scale to zero | Backlog-based autoscaling |
| Cleanup | Delete endpoint, config, model |

## Gotchas

1. **Scope the download.** The repo hosts `FL2VA/`, `Ref2VA/` **and** a diffusers copy side by side. Pulling everything wastes a lot of time and disk.
2. **Set startup timeouts generously.** Load is 112–124 s plus 26–39 s warmup *after* the S3 download. A tight `ContainerStartupHealthCheckTimeoutInSeconds` is the most common deployment failure for this model.
3. **`ref2va` is not an img2img-style editor.** It treats input video as reference material, can resynthesize or reorder motion and cuts, and exposes no denoising-strength control.
4. **Condition order is semantic.** `<Picture 1>` / `<Video 1>` / `<Audio 1>` are one-based *within each modality* and must match the order in `conditions`.
5. **Reference limits:** ≤9 images, ≤3 video clips, ≤3 audio clips, ≤12 files total; clips 2–15 s each, ≤15 s total; audio cannot be the sole input.
6. **Check the licence.** The MiniMax H3 Community License is not Apache 2.0 and carries use restrictions. Review it — and the [licence Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md) — before recommending commercial use.
7. **Sparse attention isn't in this release.** H3 trains with native sparse attention but the initial open-source release is full-attention inference only; it ships later.

## Unresolved

- **Latency figures are not directly comparable across the published tables.** The 4× H100 numbers are labelled "pipeline latency" (13.25 s) while the 4× H200 quality sweep reports 75.10 s lossless and the 8× B300 sweep reports 19.04 s, all nominally at 1344×768 / 124 frames / 50 steps. These almost certainly measure different spans. Benchmark on your own target instance before quoting any of them to a customer.
- **No SageMaker-native DLC.** The custom container works but carries maintenance cost. Worth tracking whether an SGLang diffusion DLC ships.
- **FP8 is only verified on B200/B300**, where it cut peak memory from ~83.5 GB to ~51.9 GB per GPU. Not validated on Hopper.

## References

- [MiniMax-H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [SGLang MiniMax-H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
- [vLLM MiniMax-H3 recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Wan2.1-T2V example in this repo](../../Wan/Wan2.1-T2V-1.3B-Diffusers/) — the closest existing pattern
