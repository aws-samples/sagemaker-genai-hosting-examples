# DeBERTa + XGBoost Triton Ensemble

A two-stage NLI (Natural Language Inference) classification ensemble served by [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) on Amazon SageMaker using [Inference Components](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-components.html).

1. **NLI DeBERTa** — scores input text against 50 candidate labels via zero-shot cross-encoder (`cross-encoder/nli-deberta-v3-base`), running on GPU
2. **XGBoost** — binary classifier that takes the NLI confidence vector as input, running on CPU

Both models are exported to ONNX so Triton serves them with the `onnxruntime` backend — no Python dependencies at inference time.

**Example use case: email threat detection.** Each email body is scored against a **50-category threat taxonomy** — `phishing`, `credential harvesting`, `business email compromise`, `invoice fraud`, `gift-card scam`, etc. (see `NLI_LABELS` in [workspace/config.py](workspace/config.py)). One request fans out to 50 NLI forward passes in a single GPU batch, then feeds the per-category confidence vector to XGBoost as a final malicious-or-benign arbiter. Zero-shot NLI is a natural fit when the threat taxonomy changes faster than you'd want to retrain an encoder — new threat categories are onboarded by editing the label list, and the XGBoost head is refreshed periodically as the attack mix evolves.

**Why XGBoost on top of the NLI scores, instead of just thresholding the max confidence?** Two reasons:

- **Category co-occurrence is diagnostic.** An email that scores 0.6 on `gift-card scam` *and* 0.5 on `executive impersonation` *and* 0.4 on `urgent action request` is almost certainly BEC — but no single category crosses a naive threshold. A linear rule over the confidence vector can't learn that pattern; a tree ensemble can.
- **Taxonomy evolves without retraining the encoder.** Add a new threat category → re-score historical emails through NLI → retrain only the XGBoost head on the new `N+1`-dim vectors. The expensive DeBERTa fine-tune stays frozen; the cheap classifier moves at the speed of the threat landscape.

> **Proof-of-concept disclaimer:** The XGBoost second stage is trained on `sklearn.make_classification` synthetic data — it has no real-world skill and the final pipeline predictions are effectively random. This repo exists to demonstrate the *ensemble wiring pattern*, not to solve a classification task. To adapt it, drop your own trained XGBClassifier into `convert_xgboost_to_onnx()` — see ["Adapting to your workload"](#adapting-to-your-workload) below.
>
> **How you'd train XGBoost for real:** take a labeled corpus of historical emails (or whatever inputs your workload uses), run each one through the NLI stage to produce its `N`-dim confidence vector, and train XGBoost on the resulting `(confidence_vector, malicious/benign)` pairs. Standard supervised practice applies from there — stratified splits, class-imbalance handling (threats are rare), cross-validation, and periodic retraining as the label distribution drifts. The NLI stage stays frozen; only the cheap classifier is retrained.

---

## Prerequisites

- **AWS credentials** configured 
- **SageMaker execution role** with permissions for: SageMaker full access, `ecr:GetAuthorizationToken` + `BatchGetImage` against the Triton DLC account above, and read/write on the SageMaker session's default bucket (`sagemaker-{region}-{account-id}`).
- **`ml.g5.2xlarge` service quota** in your region — the notebook's default deployment target. Check in the [Service Quotas console](https://console.aws.amazon.com/servicequotas/) before running.
- **Python 3.10+**

---

## Benchmarks

A base HuggingFace or PyTorch container runs the full Python stack on every request, adding framework and interpreter overhead before the GPU touches the model. Triton removes that: once loaded, the hot path is pure C++, which unlocks lower-level GPU optimisations (TensorRT, FP16, graph fusion) that can't be applied to a live Python model.

#### N=50 labels, seq_len=128, concurrency=1

| Instance | Configuration | Mean | Median | p99 | vs. baseline |
|---|---|---|---|---|---|
| `ml.g4dn.2xlarge` (T4) | Baseline — FP32 ONNX Runtime | 393.9 ms | 387.2 ms | 439.5 ms | — |
| `ml.g4dn.2xlarge` (T4) | TensorRT FP16 + ONNX graph level 3 | 113.3 ms | 96.4 ms | 192.5 ms | −71 % mean / 3.5× |
| `ml.g5.2xlarge` (A10G) | TensorRT FP16 + ONNX graph level 3 | 90.3 ms | 57.3 ms | 270.4 ms | −77 % mean vs. g4dn baseline / 4.4× |

TensorRT kernel fusion and FP16 (which unlocks ~65 TFLOPS on T4 vs. 8.1 TFLOPS at FP32) account for most of the gain. The Inference Component is allocated 8 GB memory, 1 GPU, and 2 CPUs — fits both 2xlarge sizes above. The notebook defaults to `ml.g5.2xlarge`; `ml.g4dn.2xlarge` is the cheaper T4 alternative.

---

## What changes from HuggingFace / PyTorch

No retraining required. Five changes get you from a stock DLC to this ensemble:

| Change | What it does | Where in the code |
|---|---|---|
| **Export models to ONNX** | Removes Python + PyTorch dispatch overhead; bakes postprocessing (softmax + entailment extraction) into the graph; unlocks downstream optimisations | [`NLIWithPostprocess`](workspace/export_models.py) wrapper + `torch.onnx.export()` for NLI; [`convert_xgboost_to_onnx()`](workspace/export_models.py) for XGBoost |
| **ONNX graph optimisation (level 3)** | Fuses adjacent ops (e.g. `MatMul + Add + Gelu`), drops redundant transposes, folds constants | `graph { level: 3 }` in [`get_nli_config()`](workspace/config.py) |
| **TensorRT FP16** | Swaps generic ORT kernels for TRT; JIT-compiles layer fusion for the specific GPU; FP16 is the bulk of the 3.5× speedup | `optimization` block in [`get_nli_config()`](workspace/config.py) |
| **Write Triton configs** | A `config.pbtxt` per model + an ensemble config wires NLI → XGBoost, replacing Python glue | Generators in [`config.py`](workspace/config.py); written by [`build_triton_repo.py`](workspace/build_triton_repo.py) |
| **Move tokenization to the client** | Triton serves raw tensors, so the client tokenizes and sends `[N, 128]` arrays directly | [`tokenize_nli_pairs()`](workspace/run_benchmark.py) |
| **Swap the SageMaker container** | Use the Triton DLC instead of the HuggingFace/PyTorch DLC; no `inference.py` needed | `ModelBuilder` setup in the [deployment notebook](deberta_xgboost_triton_ensemble.ipynb) |

> **TensorRT cold starts:** First load JIT-compiles TRT kernels (~67s on A10G) and isn't cached across restarts. To eliminate this for autoscaling, pre-compile offline with [`trtexec`](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) and deploy the `.plan` file directly via Triton's [`tensorrt` backend](https://github.com/triton-inference-server/tensorrt_backend) — engine files are GPU-architecture specific (an A10G plan won't run on a T4).

---

## Architecture

```
 ┌─────────────────────────────────────┐
 │              Client                 │
 │                                     │
 │   Input text + 50 labels            │
 │          │                          │
 │          ▼                          │
 │   Tokenizer (client-side)           │
 └──────────┬──────────────────────────┘
            │
            │  input_ids [50, 128]
            │  attention_mask [50, 128]
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │              Triton Inference Server                     │
 │                                                          │
 │   ensemble_model (platform: ensemble)                    │
 │          │                                               │
 │          ▼                                               │
 │   ┌──────────────────────────────────────────────┐       │
 │   │  Stage 1 — GPU                               │       │
 │   │                                              │       │
 │   │  nli_deberta (backend: onnxruntime)          │       │
 │   │  cross-encoder/nli-deberta-v3-base           │       │
 │   │  + baked-in postprocessing                   │       │
 │   │    (softmax + entailment extraction)         │       │
 │   └──────────────────┬───────────────────────────┘       │
 │                      │                                   │
 │                      │  confidence_vector [1, 50]        │
 │                      ▼                                   │
 │   ┌──────────────────────────────────────────────┐       │
 │   │  Stage 2 — CPU                               │       │
 │   │                                              │       │
 │   │  xgboost_classifier (backend: onnxruntime)   │       │
 │   │  Binary classifier on confidence patterns    │       │
 │   └──────────────────┬───────────────────────────┘       │
 │                      │                                   │
 │                      │  PREDICTION [1]                   │
 └──────────────────────┼───────────────────────────────────┘
                        │
                        ▼
                     Result
```

### Fan-Out / Fan-In Pattern

For each request, the client tokenizes 50 `(premise, hypothesis)` pairs and sends them as a single `[50, 128]` tensor batch. All 50 label inferences run **in parallel** in one GPU forward pass. The NLI model's baked-in postprocessing (2-class softmax over contradiction/entailment logits) produces a `[1, 50]` confidence vector, which Triton's ensemble scheduler routes directly to XGBoost.

**`max_batch_size: 0`** on all models — Triton's ensemble scheduler propagates the batch dimension uniformly, which is incompatible with the fan-out (1 text → 50 pairs) / fan-in (50 scores → 1 vector) pattern. Setting it to 0 disables Triton's batch management so tensor shapes are controlled explicitly.

## Repository Structure

```
├── workspace/
│   ├── config.py              # Constants (N_LABELS, MAX_SEQ_LEN, model ID) + Triton config generators
│   ├── export_models.py       # ONNX export for NLI DeBERTa (with postprocessing) and XGBoost
│   ├── build_triton_repo.py   # Assembles Triton model repository from exported artifacts
│   └── run_benchmark.py       # Client-side tokenization + SageMaker endpoint benchmarking
├── triton-serve-ensemble/     # Triton model repository
│   ├── nli_deberta/           # config.pbtxt + 1/model.onnx (+ model.onnx.data)
│   ├── xgboost_classifier/    # config.pbtxt + 1/model.onnx
│   └── ensemble_model/        # config.pbtxt + 1/ (ensemble is config-only, no artifact)
└── deberta_xgboost_triton_ensemble.ipynb  # Full deployment notebook (export → deploy → benchmark → cleanup)
```

> `config.pbtxt` files are committed for browsing but regenerated by `build_triton_repo.py` — edit the generators in `config.py`, not the configs directly.

## When this pattern fits

Good fit when a single request fans out to N forward passes through an encoder (e.g. scoring one text against N candidate labels) and a downstream classical model consumes the per-label scores — zero-shot NLI, content moderation, intent routing, compliance flagging. Works best when the encoder is stable and the label taxonomy evolves faster than you'd want to retrain.

Probably not a fit when the pipeline is a single model with no fan-out (a plain Triton or HuggingFace DLC is simpler), or when pre/postprocessing requires behaviour that can't be expressed in ONNX (stateful tokenization, external service calls).

## Adapting to your workload

Most adaptations will change **one or more of these coupled constants**. Changing any one of them requires updating the others and re-running the export + build steps:

| What you want to change | Where to edit | What else needs to change |
|---|---|---|
| **Number of candidate labels (N)** | `N_LABELS` in [`workspace/config.py`](workspace/config.py) | Re-run both ONNX exports (shapes are baked into the graph), rebuild the Triton repo. XGBoost stage input dim must match. |
| **Label taxonomy** | `NLI_LABELS` list in [`workspace/config.py`](workspace/config.py) | Length must equal `N_LABELS`. No re-export needed — labels are client-side only. |
| **Max sequence length** | `MAX_SEQ_LEN` in [`workspace/config.py`](workspace/config.py) | Re-run NLI ONNX export (shape is baked in), rebuild Triton repo. |
| **Swap the NLI encoder** | `NLI_MODEL_ID` in [`workspace/config.py`](workspace/config.py) | Confirm the new model's label indices match `ENTAILMENT_IDX` / `CONTRADICTION_IDX`. Re-export. |
| **Swap the classifier** (recommended — the demo one is synthetic) | Replace the call to `train_demo_xgboost()` with your own trained `XGBClassifier`; pass it to `convert_xgboost_to_onnx()` in [`workspace/export_models.py`](workspace/export_models.py) | Input dim must equal `N_LABELS`. The Triton config expects outputs named `label` and `probabilities` — standard onnxmltools output. |
| **Change the hypothesis template** | `HYPOTHESIS_TEMPLATE` in [`workspace/config.py`](workspace/config.py) | No re-export needed — applied client-side during tokenization. |

After any `config.py` change that affects shapes: re-run `export_models.py` and `build_triton_repo.py`, then redeploy.

## Getting Started

The notebook **[deberta_xgboost_triton_ensemble.ipynb](deberta_xgboost_triton_ensemble.ipynb)** walks through the entire workflow end-to-end: environment setup, ONNX export, Triton repository build, S3 upload, SageMaker deployment, benchmarking, and cleanup. Start there.
