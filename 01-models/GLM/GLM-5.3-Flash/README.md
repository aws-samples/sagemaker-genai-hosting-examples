# Hosting GLM-5.3-Flash on Amazon SageMaker AI

A self-contained notebook that builds a serving container, stages 306 GiB of weights into S3,
deploys `zai-org/GLM-5.3-Flash` to a SageMaker AI real-time endpoint, verifies it serves,
measures throughput and cost, and tears it down.

Every figure labelled *measured* came from an actual bring-up on `ml.p5en.48xlarge` in
`us-east-2`.

| | |
| --- | --- |
| Notebook | `GLM-5.3-Flash.ipynb` |
| Model | `zai-org/GLM-5.3-Flash`, 321 B total / 18 B active, natively FP8 |
| Weight footprint | **~306 GiB**, measured |
| Serving stack | vLLM OpenAI-compatible server in a custom container |
| Validated instance | `ml.p5en.48xlarge` (8x H200 SXM 141 GiB) |
| Validated topology | TP=4, DP=2, expert parallel on, so EP=8 |

---

## Prerequisites

- An AWS account with SageMaker AI access, and **quota for an 8-GPU instance type**.
- An execution role carrying `AmazonSageMakerFullAccess`.
- A Hugging Face token with access to the model repo, stored in Secrets Manager.
- `boto3`. Nothing else; the notebook has no dependency on the wider repo.

```bash
pip install boto3
jupyter lab GLM-5.3-Flash.ipynb
```

Credentials come from the default chain. Set `PROFILE` in step 1 if you use a named profile.
Account ID, bucket name, role ARN, and ECR URI are all derived at runtime from
`sts:GetCallerIdentity`, so there is nothing account-specific to edit.

### Name three resources carefully

The managed IAM policies are prefix-scoped. Get these names right and you need no custom policy.

| resource | must be named | why |
| --- | --- | --- |
| ECR repository | `sagemaker-*` | the CodeBuild service role permits ECR push only on `repository/sagemaker-*` |
| S3 staging bucket | contains `sagemaker` | `AmazonSageMakerFullAccess` scopes S3 object access to `*sagemaker*` |
| Secrets Manager secret | `AmazonSageMaker-*` | that policy scopes `GetSecretValue` to `secret:AmazonSageMaker-*` |

That CodeBuild role is not granted `ecr:DescribeImages`, so treat any post-push verification
call as advisory rather than authoritative. The notebook does.

---

## The nine steps

| step | what it does | run time |
| --- | --- | --- |
| 1 | Configure the session | instant |
| 2 | Check the model's memory footprint | seconds |
| 3 | See which one-time setup is already done | seconds |
| 4 | Build and push the serving container | ~10 min |
| 5 | Stage 306 GiB of weights into S3 | ~15 min |
| 6 | Deploy the endpoint | ~15 min |
| 7 | Confirm the engine came up as intended | seconds |
| 8 | Send requests, measure throughput and cost | ~25 min |
| 9 | Delete everything | ~5 min |

Steps 4 and 5 are one-time. Once the image and the weights exist, step 3 detects them and later
runs go straight to step 6.

Measured on the real run: **305.8 GiB staged in 14.8 min at 353 MiB/s**, weight load 40.2 s,
engine init 404.4 s.

### Cost controls

Steps 4 and 5 provision billable resources, so both are gated:

```python
RUN_BUILD = False   # CodeBuild minutes
RUN_STAGE = False   # an ml.m5.12xlarge Processing job
```

Running every cell cannot start either by accident.

**Step 6 is not gated and provisions an 8-GPU endpoint.** Read it before running it.

---

## Requirements that drive the design

### 306 GiB of weights is a hard floor

The model must be fully resident before it serves a token, which is what forces 8 GPUs. It is
independent of how much throughput you need. There is **no 4-GPU H200 instance** on SageMaker or
EC2; `p5e` and `p5en` ship only as `48xlarge`. So an 8-GPU shape is the minimum viable
deployment even for modest traffic, and configuring TP=4 would leave four GPUs idle at the same
hourly rate.

### Weights must be a local directory, not a Hub repo id

This architecture's multimodal processor reads `processor_config.json` from a local path, so
`SM_VLLM_MODEL` must point at `/opt/ml/model` backed by an S3 prefix.

Staging first also splits the work across two separate budgets, both capped at **3600 s** by the
service: `ModelDataDownloadTimeoutInSeconds` for the S3 copy and
`ContainerStartupHealthCheckTimeoutInSeconds` for engine init. Pulling weights at container start
races both inside one budget.

For any container that does resolve from the Hub at load time, including DJL LMI driven by
`HF_MODEL_ID`, set the download filter so config and processor files come along with the weights:

```
HF_HUB_DOWNLOAD_ALLOW_PATTERNS = *.json,*.jinja,*.safetensors,*.model,*.txt,*.py
```

The notebook sets this by default.

### DP raises KV capacity, TP does not

The most important sizing fact. MLA caches a compressed latent that is not head-partitioned, so
vLLM **replicates it on every tensor-parallel rank**. Measured at TP=8:

```
82.07 GiB / 7,253,058 tokens = 11.86 KiB per token
```

That pool is one GPU's KV memory, not eight. So raising TP duplicates the same cache more times,
while raising DP adds distinct caches. Confirmed twice: DP=2 raised total usable KV from
7,253,058 to **9,857,994 tokens** even with utilisation reduced from 0.90 to 0.85.

To check this on any new config, divide `Available KV cache memory` by `GPU KV cache size` and
see whether the pool tracks one GPU or all of them.

### DP=2 is not two replicas

A costly misreading. The 288 routed experts shard **exactly once** across all 8 GPUs; only
attention and dense weights duplicate per DP group. Per-GPU weight cost moves 38.3 to 39.9 GiB
between TP=8 and TP=4/DP=2, which is why you choose topology on throughput rather than memory.

One endpoint, one instance, one copy of the expert weights. **Setting DP=1 does not reduce the
bill.**

Related: vLLM derives expert-parallel width, **EP = TP x DP**, so "TP=4, EP=2" is not
expressible. And `--max-num-seqs` is **per DP replica**.

### Other constraints

- **Do not set `VolumeSizeInGB`.** These instance types ship local NVMe and SageMaker rejects the
  parameter for instance types that provide their own instance storage.
- **`kv-cache-dtype` must not be an fp8 value.** This model is NoPE MLA (`qk_rope_head_dim: 0`),
  which the FP8 MLA cache kernel does not support. Leave it at `auto`.
- **Capacity is not quota.** An 8-GPU request can return `InsufficientInstanceCapacity` with
  quota to spare, and it takes about 31 minutes to surface. `InstancePools` gives ordered
  fallback across up to 5 types but yields one endpoint on whichever type had capacity, so it
  cannot pin a type for comparison work. Training plans reserve capacity but must target
  resource `endpoint`, and are p-family only.

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
| `SM_VLLM_KV_CACHE_DTYPE` | `auto` | see above |

---

## Measured results

`ml.p5en.48xlarge`, TP=4/DP=2/EP=8, 8192 in / 256 out, streaming.

| concurrency | TTFT p50 | TTFT p90 | ITL p50 | output tok/s | req/s |
| --- | --- | --- | --- | --- | --- |
| 16 | 1428 ms | 12,316 ms | 20.1 ms | 311 | 1.2 |
| 32 | 979 ms | 4,672 ms | 25.0 ms | 842 | 3.3 |
| 64 | 634 ms | 2,724 ms | 32.3 ms | 1,458 | 5.7 |
| 128 | **415 ms** | 1,561 ms | 43.4 ms | **2,359** | 9.3 |

Benchmarked with **Amazon SageMaker AI optimized generative AI benchmarking**. Drive it through
the `CreateAIBenchmarkJob` API rather than the SDK wrapper: `concurrency` accepts a list so one
job sweeps every level, results persist in S3 independently of the job, and it returns
end-to-end latency percentiles the wrapper omits.

### Reading these numbers correctly

**The GPUs were saturated throughout.** CloudWatch showed **94 to 99% per-GPU utilisation from
the first minute of load, at every concurrency level including 16**. The rise from 311 to 2,359
tok/s came from better batching efficiency on already-busy GPUs, not from filling idle silicon.
A "percentage of peak throughput" therefore describes batch occupancy, not reclaimable hardware.

Note that SageMaker's `GPUUtilization` is **summed across GPUs**, so the ceiling on an 8-GPU
instance is 800%. Divide, or the numbers look impossible.

**The tail is 9x the median at low concurrency.** TTFT p50 1428 ms against p90 12,316 ms at
concurrency 16, with warm-up in place. With 8192-token inputs and `--max-num-batched-tokens 8192`
only one prefill fits per scheduler step, so a queued request waits whole prefill cycles. If your
traffic sits at low concurrency, raising `--max-num-batched-tokens` is the lever to test.

**Every output token above is a reasoning token.** The model defaults to `reasoning_effort=max`,
and this workload capped output at 256 tokens while a max-effort response has a p50 of about
2053. Responses were therefore truncated inside the reasoning trace. Throughput and cost figures
describe reasoning-token production, and TTFT is time to the first *reasoning* token. ITL is
unaffected.

`reasoning_effort` is a **per-request** field, so it can be tuned per call:

```json
{"model": "glm-5.3-flash", "messages": [...], "reasoning_effort": "low"}
```

Measured separately at concurrency 16 with a 2048-token cap, `max` emits **1.46x** the output
tokens of `low` on identical prompts. That is a lower bound, because 66% of `max` responses still
reached the cap. There is **no quality evaluation** across effort levels, only token counts.

---

## Cost

Derived from the measured throughput at $72.795/hr in `us-east-2`, assuming a continuously
loaded endpoint.

| concurrency | $/1M output tokens |
| --- | --- |
| 16 | 65.02 |
| 32 | 24.02 |
| 64 | 13.87 |
| 128 | **8.57** |

Duty cycle governs real cost per token far more than any engine setting. An endpoint at 25%
utilisation costs four times these figures per token.

Step 9 deletes the endpoint, then the config, then the model. **An idle endpoint bills at the
full hourly rate**, roughly $1,750 a day at this instance type, so run it.

One Pricing API detail: SageMaker uses the **`instanceName`** attribute, not `instanceType`.
Filtering on `instanceType` returns zero results with no error. The Pricing API also lives in
`us-east-1` regardless of the region you are pricing.

---

## Scope and limits

- **One workload shape**, 8192 in / 256 out. A 256-in / 2048-out chat workload spends far more of
  the GPU on decode and would produce very different figures. These numbers do not transfer.
- **One run per point.** No confidence intervals. Roughly 40% run-to-run variance was observed in
  TTFT at low concurrency.
- **Prefix-cache assisted.** 79.2% of prompt tokens were served from vLLM's prefix cache, so a
  workload with genuinely unique prompts would be slower and cost more per token.
- **Single instance, no autoscaling.** No allowance for the provisioning plus model load a new
  instance needs before serving, and startup time is excluded from the cost figures.
- **`ml.p5en.48xlarge` only.** No cross-hardware comparison; other instance types were attempted
  but never obtained capacity.
- **No quality evaluation** of any kind, at any `reasoning_effort` level.
