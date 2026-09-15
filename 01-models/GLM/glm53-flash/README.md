# Hosting GLM-5.3-Flash on Amazon SageMaker AI

A single self-contained notebook that builds a serving container, stages 306 GiB of weights
into S3, deploys `zai-org/GLM-5.3-Flash` to a SageMaker AI real-time endpoint, smoke tests it,
reads GPU telemetry back, and derives cost per token.

Written from an actual bring-up on `ml.p5en.48xlarge` in `us-east-2`. Every figure labelled
*measured* came off that run. The constraints that are not obvious from the documentation are
called out inline rather than left to be discovered at roughly 30 minutes per failed attempt.

| | |
| --- | --- |
| Notebook | `glm53-flash.ipynb` |
| Model | `zai-org/GLM-5.3-Flash`, 321.3 B total / 18 B active, natively FP8 |
| Weight footprint | **~306 GiB**, measured |
| Serving stack | vLLM OpenAI-compatible server in a custom container |
| Validated instance | `ml.p5en.48xlarge` (8x H200 SXM 141 GiB, sm90) |
| Validated topology | TP=4, DP=2, expert parallel on, so EP=8 |

---

## Prerequisites

- An AWS account with SageMaker AI access, and **quota for an 8-GPU instance type**.
- An execution role carrying `AmazonSageMakerFullAccess`. See the IAM naming rules below,
  because getting the names right avoids needing any custom policy at all.
- A Hugging Face token with access to the model repo, stored in Secrets Manager.
- `boto3`. Nothing else is required; the notebook has no dependency on the wider repo.

```bash
pip install boto3
jupyter lab glm53-flash.ipynb
```

Credentials come from the default chain. Set `PROFILE` in the config cell if you use a named
profile. Account ID, bucket name, role ARN, and ECR URI are all derived at runtime from
`sts:GetCallerIdentity`, so there is nothing account-specific to edit.

---

## What the notebook does

Five stages, in order. The first two are one-time.

| stage | what happens | wall clock |
| --- | --- | --- |
| 1 | Measure the model footprint from the HF API | seconds |
| 2 | Write `Dockerfile` + `serve` + `buildspec.yml`, build and push via CodeBuild | ~10 min |
| 3 | Stream the weights into S3 with a Processing job | ~15 min |
| 4 | Create model, endpoint config, endpoint; wait; smoke test | ~15 min |
| 5 | Benchmark, GPU telemetry, cost per token, teardown | ~25 min |

Measured on the real run: **305.8 GiB moved in 14.8 minutes at 353 MiB/s**, weight load
40.2 s, engine init 404.4 s.

### Cost guards

Stages 2 and 3 provision billable resources, so both are gated:

```python
RUN_BUILD = False   # CodeBuild minutes
RUN_STAGE = False   # an ml.m5.12xlarge Processing job
```

A run-all cannot start either by accident. The preflight cell checks whether the ECR image and
the staged weights already exist and both stages self-skip when they do.

**Stage 4 is not gated and does provision an 8-GPU endpoint.** Read the cell before running it.

---

## The five things that will cost you a day if you do not know them

### 1. A bare Hugging Face repo id cannot work

vLLM's multimodal processor for this architecture does a literal local open:

```python
# vllm/transformers_utils/processors/glm5next.py
with open(os.path.join(model_path, "processor_config.json")) as f:
```

There is no Hub resolution. Passing `zai-org/GLM-5.3-Flash` makes it look for a *relative
directory* of that name and die during multimodal profiling, before any weights load. The
weights must be an S3 prefix mounted at `/opt/ml/model`. The container crash-loops until the
startup health check expires, so you pay the full timeout to learn this.

### 2. MLA replicates its KV cache across tensor-parallel ranks

The single most important sizing fact. Measured at TP=8:

```
82.07 GiB / 7,253,058 tokens = 11.86 KiB per token
```

The pool equals **one GPU's** KV memory, not eight, because MLA caches a compressed latent that
is not head-partitioned. So:

- Raising **TP does not raise KV capacity**, it duplicates the same cache more times.
- Raising **DP does**, because each attention replica holds a distinct cache.

Confirmed twice: DP=2 raised total usable KV from 7,253,058 to **9,857,994 tokens** even though
utilisation dropped from 0.90 to 0.85.

Verify on any new config by dividing `Available KV cache memory` by `GPU KV cache size` and
checking whether the pool tracks one GPU or all of them.

### 3. DP=2 is not two replicas

A common and expensive misreading. The 288 routed experts shard **exactly once** across all 8
GPUs; only attention and dense weights duplicate per DP group. Per-GPU weight cost moves 38.3
to 39.9 GiB between TP=8 and TP=4/DP=2, which is why you choose topology on throughput rather
than memory.

It is one endpoint, one instance, one copy of the expert weights. **Setting DP=1 does not
reduce the bill.**

Related: vLLM has no `--expert-parallel-size`. EP width is derived, **EP = TP x DP**, so
"TP=4, EP=2" is not expressible. Also `--max-num-seqs` is **per DP replica**.

### 4. 306 GiB of weights is a hard floor, independent of throughput

The model must be resident before it serves a single token, which is what forces 8 GPUs. There
is **no 4-GPU H200 instance** on SageMaker or EC2; `p5e` and `p5en` ship only as `48xlarge`. So
an 8-GPU shape is the minimum viable deployment even for modest traffic, and configuring TP=4
would idle four GPUs at the same hourly rate.

### 5. IAM: three resource names are load-bearing

The managed policies are prefix-scoped. Name things correctly and you need no policy changes.

| resource | must be named | why |
| --- | --- | --- |
| ECR repository | `sagemaker-*` | the CodeBuild service role allows ECR push only on `repository/sagemaker-*` |
| S3 staging bucket | contains `sagemaker` | `AmazonSageMakerFullAccess` scopes S3 object access to `*sagemaker*` |
| Secrets Manager secret | `AmazonSageMaker-*` | that policy scopes `GetSecretValue` to `secret:AmazonSageMaker-*` |

That same CodeBuild role is **not** granted `ecr:DescribeImages`, so a post-push verification
call returns an error even though the push succeeded. Keep such checks non-fatal; the notebook
does.

---

## Configuration reference

All engine tuning happens through the Model's `Environment` map. The container entrypoint
translates every `SM_VLLM_<FLAG>` variable into `--<flag>` for `vllm serve`, and
`SM_VLLM_MODEL` into the positional argument. **No image rebuild is needed to change context
length, batch size, or topology.**

| variable | validated value | notes |
| --- | --- | --- |
| `SM_VLLM_MODEL` | `/opt/ml/model` | a local directory, never a repo id |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | `4` | |
| `SM_VLLM_DATA_PARALLEL_SIZE` | `2` | attention replicas |
| `SM_VLLM_ENABLE_EXPERT_PARALLEL` | `true` | with TP=4, DP=2 gives EP=8 |
| `SM_VLLM_MAX_MODEL_LEN` | `32768` | native window is 1,048,576 |
| `SM_VLLM_MAX_NUM_SEQS` | `128` | per DP replica, so 256 total |
| `SM_VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | |
| `SM_VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | chunked-prefill chunk size |
| `SM_VLLM_KV_CACHE_DTYPE` | `auto` | never an fp8 value; see below |

Two endpoint-config values, both capped at **3600 s** by the service:
`ModelDataDownloadTimeoutInSeconds` and `ContainerStartupHealthCheckTimeoutInSeconds`.
Staging to S3 first is what lets 306 GiB of download and engine init use separate budgets
instead of racing inside one.

Do **not** set `VolumeSizeInGB`. These instance types ship local NVMe and SageMaker rejects the
parameter for instance types that provide instance storage.

---

## Measured results

`ml.p5en.48xlarge`, TP=4/DP=2/EP=8, 8192 in / 256 out, streaming.

| concurrency | TTFT p50 | TTFT p90 | ITL p50 | output tok/s |
| --- | --- | --- | --- | --- |
| 16 | 1428 ms | 12,316 ms | 20.1 ms | 311 |
| 32 | 979 ms | 4,672 ms | 25.0 ms | 842 |
| 64 | 634 ms | 2,724 ms | 32.3 ms | 1,458 |
| 128 | **415 ms** | 1,561 ms | 43.4 ms | **2,359** |

### Read these three caveats before quoting the numbers

**The GPUs were saturated the whole time.** CloudWatch showed **94 to 99% per-GPU utilisation
from the first minute of load, at every concurrency level including 16**. The rise from 311 to
2,359 tok/s came from better batching efficiency on already-busy GPUs, not from filling idle
silicon. So "we only use 13% of peak throughput" describes batch occupancy, not reclaimable
hardware.

Note that SageMaker's `GPUUtilization` is **summed across GPUs**, so the ceiling on an 8-GPU
instance is 800%. Divide, or the numbers look impossible.

**The tail is 9x the median at low concurrency.** TTFT p50 1428 ms against p90 12,316 ms at
concurrency 16, and it is not cold start because warm-up ran. It is head-of-line blocking: with
8192-token inputs and `--max-num-batched-tokens 8192` only one prefill fits per scheduler step.
If your traffic sits at low concurrency, that tail is your user experience.

**Every output token above is a reasoning token.** The model defaults to
`reasoning_effort=max`, and this workload capped output at 256 tokens while a max-effort
response has a p50 of about 2053 tokens. So responses were truncated deep inside the reasoning
trace and never reached an answer. Throughput and cost figures therefore describe
reasoning-token production, and TTFT is time to the first *reasoning* token. ITL is unaffected.

`reasoning_effort` is a **per-request** body field, so it can be tuned per call:

```json
{"model": "glm-5.3-flash", "messages": [...], "reasoning_effort": "low"}
```

Measured separately at concurrency 16 with a 2048-token cap, `max` emits **1.46x** the output
tokens of `low` on identical prompts, which is a lower bound because 66% of `max` responses
still hit the cap. There is **no quality evaluation** across effort levels, only token counts.

---

## Known failure modes

### `pe_dim must be 64 for fp8_ds_mla`

Seen on 96 GiB cards at utilisation 0.90 and 131072 context. The endpoint provisions, then
crash-loops until the health check expires.

```
RuntimeError: concat_and_cache_mla, csrc/libtorch_stable/cache_kernels.cu:866,
              pe_dim must be 64 for fp8_ds_mla
```

vLLM's FP8 DeepSeek-MLA KV-cache write kernel asserts the positional-encoding dimension is 64.
This model sets `qk_rope_head_dim: 0` and `mla_use_nope: true`, so `pe_dim` is **0** and the
kernel has no NoPE path.

**It is memory pressure, not an architecture mismatch.** The preceding error reads
`OOM on device 0 ... (free: 5701632, total: 101973819392)`, i.e. a 95 GiB card with 5.7 MB free.
The profiler exhausted its budget, vLLM fell back to an FP8 KV cache to fit, and that routed
into the unusable kernel. On 141 GiB cards there was room, so it stayed on the BF16 MLA path.

Mitigate with lower `--gpu-memory-utilization`, shorter `--max-model-len`, lower
`--max-num-seqs`, and an explicit `--kv-cache-dtype auto`.

A corollary worth internalising: **a config sized against the tightest member of an instance
pool is not automatically safe across it.** A config that boots on 141 GiB cards can die on
96 GiB ones.

### `no kernel image is available for execution on the device`

Not this container, but it catches people moving to newer GPUs. Blackwell (sm_120) needs
**PyTorch 2.7.0 or later built with CUDA 12.8 or newer**. A `-cu128` image tag does **not**
prove the bundled torch wheel has sm_120 cubins; the toolkit version and the compiled
architecture list are different things. Settle it in one command instead of a 700 s
deploy-and-fail:

```bash
docker run --rm --entrypoint python3 <image> \
  -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
```

Separately compiled extensions each carry their own architecture list, so `flash-attn`,
`bitsandbytes`, and `deepspeed` can fail this way after torch itself is fine.

### `InsufficientInstanceCapacity`

The dominant practical obstacle, and it takes about **31 minutes** to surface. Quota and
capacity are unrelated; quota was 24 for every type involved.

Two mitigations. **Instance pools** give ordered fallback across up to 5 types, but yield one
endpoint on whichever type had capacity, so they cannot pin a type for comparison work.
**Training plans** reserve capacity via `CapacityReservationConfig`, but the plan must target
resource `endpoint`, and they are **p-family only**. Note also that
`VariantInstanceProvisionTimeoutInSeconds` does not fast-fail a pinned single-instance-type
variant; it appears to apply to pools only.

### Never let a transient error reach a teardown handler

Two real incidents on this project. A check-mark emoji raised `UnicodeEncodeError` on a cp1252
console, a broad `except Exception` read it as a deploy failure, and **deleted a healthy, fully
warmed endpoint**. Separately, a transient `EndpointConnectionError` during polling caused a
rollback that deleted a still-provisioning endpoint along with its `FailureReason`, making a
local network blip indistinguishable from a capacity failure.

The notebook's wait loop retries connection errors, never deletes anything, and leaves a
still-`Creating` endpoint in place so its `FailureReason` survives.

---

## Cleanup

The last cell deletes the endpoint, then the config, then the model. **An idle endpoint bills at
the full hourly rate**, so run it. At the validated instance type that is $72.795/hr in
`us-east-2`, roughly $1,750 a day.

Cost per token from the measured throughput, assuming a continuously loaded endpoint:

| concurrency | $/1M output tokens |
| --- | --- |
| 16 | 65.02 |
| 32 | 24.02 |
| 64 | 13.87 |
| 128 | **8.57** |

Duty cycle governs real cost per token far more than any engine setting. An endpoint at 25%
utilisation costs four times these figures per token.

---

## Scope and limits

- **One workload shape**, 8192 in / 256 out. A 256-in / 2048-out chat workload spends far more
  of the GPU on decode and would produce very different figures. These numbers do not transfer.
- **One run per point.** No confidence intervals. Roughly 40% run-to-run variance was observed
  in TTFT at low concurrency.
- **Prefix-cache assisted.** 79.2% of prompt tokens were served from vLLM's prefix cache, so a
  workload with genuinely unique prompts would be slower and cost more.
- **Single instance, no autoscaling**, and startup time excluded from cost figures.
- **`ml.p5en.48xlarge` only.** Attempts on 96 GiB and H100 instance types never obtained
  capacity, so there is no cross-hardware comparison here.
