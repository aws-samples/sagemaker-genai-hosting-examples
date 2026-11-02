# Deploy Reflection Beam 1.0 with Amazon SageMaker JumpStart

This sample deploys the BF16, FP8, or NVFP4 variant of Reflection Beam 1.0
501B-A23B to an Amazon SageMaker AI real-time endpoint, invokes it, runs a
bounded smoke test, and cleans up the endpoint.

The sample uses the released model IDs from `SageMakerPublicHub`. It defaults to
NVFP4; use `--variant fp8` or `--variant bf16` to select another variant.

This folder also contains a custom-container sample for the NVFP4 checkpoint on
`ml.g7e.24xlarge` (4x NVIDIA RTX PRO 6000 Blackwell). Use it when you need that
instance type or a specific vLLM configuration, such as the full 1M-token
context with an FP8 KV cache or DFlash2 speculative decoding. See
[Deploy NVFP4 on ml.g7e.24xlarge with a custom vLLM container](#deploy-nvfp4-on-mlg7e24xlarge-with-a-custom-vllm-container).

For agent-guided hardware, capacity, cost, deployment, testing, and cleanup
decisions, use the
[SageMaker generative AI deployment Skill](../../../.agents/skills/sagemaker-generative-ai-deploy/SKILL.md)
with its
[Reflection Beam worked example](../../../.agents/skills/sagemaker-generative-ai-deploy/references/examples/reflection-beam.md).

## Prerequisites

- Use Python 3.11 or later.
- Configure AWS credentials and a default AWS Region for the SageMaker Python
  SDK.
- Use an IAM principal that can create and delete SageMaker models, endpoint
  configurations, and endpoints, and pass the required SageMaker execution role.
- Confirm that the released Beam JumpStart model and its supported instance type
  are available in your Region.
- Confirm sufficient SageMaker endpoint quota and accelerated instance capacity.

## Install the dependency

1. **Create and activate a virtual environment.**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the tested SageMaker Python SDK range.**

   ```bash
   python -m pip install -r requirements.txt
   ```

   The requirements file installs `sagemaker>=2.257.7,<3`.

## Choose the checkpoint variant

Choose the variant that matches your GPU hardware, memory requirements, and
performance goals. The script maps each CLI value to its published JumpStart
model ID:

| Variant | CLI value | JumpStart model ID |
|---|---|---|
| BF16 | `bf16` | `huggingface-llm-beam-1-0-501b-a23b` |
| FP8 | `fp8` | `huggingface-llm-beam-1-0-501b-a23b-fp8` |
| NVFP4 | `nvfp4` | `huggingface-llm-beam-1-0-501b-a23b-nvfp4` |

Keep credentials, account IDs, role ARNs, and endpoint names out of the source
file.

## Run the lifecycle CLI

1. **Run the deploy, smoke-test, and cleanup lifecycle with the default NVFP4 variant.**

   ```bash
   python jumpstart_real_time.py \
     --endpoint-name reflection-beam-smoke
   ```

2. **Select FP8 when your target configuration supports that variant.**

   ```bash
   python jumpstart_real_time.py \
     --variant fp8 \
     --endpoint-name reflection-beam-fp8-smoke
   ```

3. **Override an individual timeout when your selected variant needs more time.**

   ```bash
   python jumpstart_real_time.py \
     --variant nvfp4 \
     --endpoint-name reflection-beam-smoke \
     --endpoint-timeout-seconds 10800
   ```

The sample uses independent deadlines for model download, container startup,
endpoint deployment, and cleanup:

| CLI option | Default |
|---|---:|
| `--model-download-timeout-seconds` | 3,600 seconds |
| `--container-startup-timeout-seconds` | 3,600 seconds |
| `--endpoint-timeout-seconds` | 7,200 seconds |
| `--cleanup-timeout-seconds` | 7,200 seconds |

This separation lets a large checkpoint take longer to deploy without
shortening cleanup or changing the container health-check deadline. Each value
must be a positive integer.

The CLI deploys an endpoint, runs `smoke_test()`, prints the returned text, and
then exits. It always attempts cleanup before printing the result. If cleanup
cannot complete, it reports the endpoint name and exits with an error so you can
inspect the endpoint in SageMaker. Use the Python functions directly when you
need to keep an endpoint running.

The CLI generates a unique endpoint name unless you supply `--endpoint-name`.
If the supplied name already exists, deployment stops before creating or
deleting resources. Automatic cleanup runs only after the deployment call
returns a predictor owned by the current run.

## Use the Python functions

| Function | Behavior |
|---|---|
| `deploy_endpoint(...)` | Resolves the selected public model ID, validates the independent download, startup, endpoint, and cleanup timeouts, creates a `JumpStartModel`, deploys it, and returns its predictor. |
| `invoke_endpoint(predictor, prompt, max_tokens=64, timeout_seconds=120)` | Sends one chat request with bounded connection and read timeouts, then returns `choices[0].message.content` as text. |
| `smoke_test(predictor)` | Requests a short greeting with at most 64 output tokens and rejects empty output. |
| `cleanup_endpoint(predictor, timeout_seconds=7200)` | Retries endpoint deletion while SageMaker finishes a transitional operation, waits for deletion, then deletes the endpoint configuration and model. Passing `None` is safe. |
| `run_lifecycle(...)` | Deploys the selected variant with independent timeouts, smoke tests it, and calls cleanup from `finally` on success or smoke-test failure. |

Use `try` and `finally` when you call the functions independently:

```python
from jumpstart_real_time import (
    cleanup_endpoint,
    deploy_endpoint,
    invoke_endpoint,
)

predictor = deploy_endpoint(
    endpoint_name="reflection-beam-development",
    endpoint_timeout_seconds=7200,
    cleanup_timeout_seconds=7200,
)
try:
    output = invoke_endpoint(
        predictor,
        "Reply with a short greeting.",
        max_tokens=64,
    )
    print(output)
finally:
    cleanup_endpoint(predictor)
```

The invocation helper expects the released JumpStart model package to accept a
chat request with `messages` and `max_tokens`, and to return an OpenAI-style
`choices[0].message.content` field. Confirm this schema against the public model
documentation before release.

## Understand cleanup behavior

1. **Use the lifecycle CLI for disposable smoke-test endpoints.**
2. **Call `cleanup_endpoint()` in `finally` when you manage the functions yourself.**
3. **Inspect SageMaker for partial resources if deployment raises before returning a predictor.**

Cleanup requests endpoint deletion and waits for it to complete before deleting
the endpoint configuration and SageMaker model. This order prevents cleanup
from removing resources that an endpoint still uses. If SageMaker temporarily
rejects deletion while the endpoint is being created or updated, cleanup waits
and retries within its own deadline. If cleanup still cannot complete, the
error reports the endpoint name, current status, Region when available, and an
AWS CLI deletion command.

If deployment fails before returning a predictor, the sample leaves existing
and partial resources untouched because it cannot prove that the current run
created them. Cleanup failures are reported instead of suppressed.

## Run the tests

1. **Run the standard-library unit tests from this directory.**

   ```bash
   python -m unittest discover -s tests -v
   ```

The tests stub the SageMaker SDK and do not require AWS credentials or a live
endpoint. They verify the configured model ID, credential hygiene,
bounded deployment timeouts, request and response validation, smoke-test
behavior, retryable transitional states, and cleanup on success and failure.

## Deploy NVFP4 on ml.g7e.24xlarge with a custom vLLM container

This path deploys the NVFP4 checkpoint to a SageMaker AI real-time endpoint on
`ml.g7e.24xlarge` (4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB
each, SM120) with a bring-your-own-container (BYOC) image and weights that you
stage to Amazon S3. It is a custom runtime, so you manage the image, weights,
and vLLM flags yourself. Prefer the JumpStart lifecycle CLI above when its
supported instance types meet your needs: JumpStart lists the FP8 variant on
`ml.g7e.48xlarge` (8 GPUs) with no custom image (not tested in this sample).

**Why a custom image for NVFP4 on `ml.g7e.24xlarge`.** JumpStart supports the
NVFP4 variant only on `ml.p6-b200.48xlarge`. The same AWS vLLM container
(`vllm:server-sagemaker-cuda-v2.7`) was tested with the NVFP4 checkpoint on
EC2 `g7e.24xlarge` using the settings in this sample. Its first start spent
about 55 minutes autotuning the FlashInfer MoE kernels on these GPUs (SM120)
and then ran out of GPU memory at `--gpu-memory-utilization 0.92`. That is
close to the endpoint's 3,600-second startup health-check limit. The image in
[`container-g7e/`](container-g7e/) (the model-specific vLLM image, with FlashInfer
0.7.2 instead of 0.7.0) becomes healthy in about 5–6 minutes on the same instance.

| Path | Purpose |
|---|---|
| `Beam-1.0-501B-A23B-g7e.ipynb` | Deploy, invoke (text, tool calling, streaming), and clean up |
| `container-g7e/` | Serving image: `Dockerfile`, `serve` entrypoint, `nginx.conf` |
| `buildspec-image-g7e.yml` | AWS CodeBuild: build and push the serving image to Amazon ECR |
| `buildspec-weights.yml` | AWS CodeBuild: stream the checkpoint from Hugging Face to S3 |

### Requirements

- **Serving image built from [`container-g7e/`](container-g7e/).** It is the
  public model-specific image `vllm/vllm-openai:reflection-beam`, pinned by
  digest, plus nginx for the SageMaker `/ping` and `/invocations` contract.
  This image carries `BeamForCausalLM` and the `beam` reasoning and tool-call
  parsers. The `serve` entrypoint maps `SM_VLLM_*` environment variables to
  `vllm serve` flags, the same convention as the AWS vLLM DLC.
- **Instance `ml.g7e.24xlarge`** with quota for `ml.g7e.24xlarge for endpoint
  usage` (request an increase in Service Quotas if it is 0).
- **Inference AMI `al2023-ami-sagemaker-inference-gpu-4-1`** (driver 580 /
  CUDA 13), which matches the image's CUDA 13 build.
- An IAM execution role that trusts `sagemaker.amazonaws.com`, with read access
  to the weights bucket, pull access to your ECR repository, and CloudWatch Logs
  write access. In SageMaker Studio the notebook uses the Studio execution role;
  elsewhere, set `role` in the notebook to this role's ARN. The Beam weights are
  public; a Hugging Face token is optional and only raises download rate limits.
- **One Region for everything.** The weights bucket, the ECR image, and the
  endpoint must be in the same Region. SageMaker does not pull images across
  Regions.

### Stage the weights to S3

The ~294 GB checkpoint is streamed from Hugging Face to S3 by an AWS CodeBuild
job, so you do not need local disk. The job copies files in parallel, checks
each object's size against the Hugging Face metadata, and skips files that are
already staged, so you can safely re-run it.

1. **Set the shell variables.**

   ```bash
   export AWS_REGION=<YOUR_REGION>
   export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
   export BUCKET=<YOUR_BUCKET>            # existing bucket in the same Region
   export HF_REPO=reflection/Beam-1.0-501B-A23B-NVFP4
   export WEIGHTS_PREFIX=models/beam-1.0-501b-nvfp4
   export CB_ROLE=<YOUR_CODEBUILD_ROLE_NAME>
   ```

2. **Create a CodeBuild service role scoped to your bucket and the CodeBuild
   log groups, and upload the buildspec.**

   ```bash
   aws iam create-role --role-name ${CB_ROLE} \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
   aws iam put-role-policy --role-name ${CB_ROLE} --policy-name weights-staging \
     --policy-document '{"Version":"2012-10-17","Statement":[
       {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload","s3:ListBucket"],
        "Resource":["arn:aws:s3:::'"${BUCKET}"'","arn:aws:s3:::'"${BUCKET}"'/*"]},
       {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
        "Resource":"arn:aws:logs:'"${AWS_REGION}"':'"${ACCOUNT_ID}"':log-group:/aws/codebuild/*"}]}'
   export CB_ROLE_ARN=$(aws iam get-role --role-name ${CB_ROLE} --query Role.Arn --output text)
   sleep 15  # IAM propagation

   zip /tmp/beam-weights-src.zip buildspec-weights.yml
   aws s3 cp /tmp/beam-weights-src.zip s3://${BUCKET}/build/beam-weights-src.zip
   ```

3. **Create and run the staging build.** The 8-hour timeout leaves headroom for
   slow Hugging Face throughput.

   ```bash
   aws codebuild create-project \
     --name beam-weights \
     --source "type=S3,location=$BUCKET/build/beam-weights-src.zip,buildspec=buildspec-weights.yml" \
     --artifacts type=NO_ARTIFACTS \
     --service-role "$CB_ROLE_ARN" \
     --timeout-in-minutes 480 \
     --environment "type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux-x86_64-standard:5.0,computeType=BUILD_GENERAL1_LARGE,environmentVariables=[{name=HF_REPO,value=${HF_REPO}},{name=BUCKET,value=${BUCKET}},{name=PREFIX,value=${WEIGHTS_PREFIX}}]"

   aws codebuild start-build --project-name beam-weights
   ```

The notebook mounts `s3://${BUCKET}/${WEIGHTS_PREFIX}/` read-only at
`/opt/ml/model`. To use a Hugging Face token, store it in AWS Secrets Manager
and add it to the project as a `SECRETS_MANAGER` environment variable named
`HF_TOKEN`; do not pass it in plain text.

### Build the serving image

Run once in the same Region as the endpoint, reusing the variables above:

```bash
export REPO=<YOUR_ECR_REPOSITORY>
export REGISTRY=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
export IMAGE=${REGISTRY}/${REPO}:reflection-beam
aws ecr create-repository --repository-name ${REPO}

# Allow the CodeBuild role to push to the repository.
aws iam put-role-policy --role-name ${CB_ROLE} --policy-name ecr-push \
  --policy-document '{"Version":"2012-10-17","Statement":[
    {"Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"},
    {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:InitiateLayerUpload","ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload","ecr:PutImage","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],
     "Resource":"arn:aws:ecr:'"${AWS_REGION}"':'"${ACCOUNT_ID}"':repository/'"${REPO}"'"}]}'

zip -r /tmp/beam-g7e-src.zip buildspec-image-g7e.yml container-g7e
aws s3 cp /tmp/beam-g7e-src.zip s3://${BUCKET}/build/beam-g7e-src.zip

aws codebuild create-project \
  --name beam-g7e-image \
  --source "type=S3,location=$BUCKET/build/beam-g7e-src.zip,buildspec=buildspec-image-g7e.yml" \
  --artifacts type=NO_ARTIFACTS \
  --service-role "$CB_ROLE_ARN" \
  --timeout-in-minutes 120 \
  --environment "type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux-x86_64-standard:5.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true,environmentVariables=[{name=REGISTRY,value=${REGISTRY}},{name=IMAGE,value=${IMAGE}}]"

aws codebuild start-build --project-name beam-g7e-image
```

The SageMaker execution role also needs `ecr:BatchGetImage`,
`ecr:GetDownloadUrlForLayer`, and `ecr:GetAuthorizationToken` to pull the image.

### Deploy with the notebook

Open [`Beam-1.0-501B-A23B-g7e.ipynb`](Beam-1.0-501B-A23B-g7e.ipynb), set
`bucket` and `ecr_repository` (and `role` outside SageMaker Studio), and run
the cells. The notebook uses your session's default Region, so set
`AWS_DEFAULT_REGION` to the Region of your bucket and image before you start
it. The notebook creates the
model, endpoint configuration, and endpoint; runs text, tool-calling, and
streaming requests; and deletes all three resources in its cleanup cell.

The notebook sets this vLLM configuration through `SM_VLLM_*` variables:

| Setting | Value |
|---|---|
| Tensor parallelism | 4 |
| Context length | `--max-model-len 1048576` |
| KV cache | `--kv-cache-dtype fp8_e4m3`, `--enable-prefix-caching` |
| Batching | `--enable-chunked-prefill`, `--max-num-batched-tokens 16384`, `--max-num-seqs 16` |
| Memory and loading | `--gpu-memory-utilization 0.92`, `--safetensors-load-strategy eager` |
| Parsers | `--reasoning-parser beam`, `--tool-call-parser beam`, `--enable-auto-tool-choice` |
| Environment | `VLLM_ENGINE_READY_TIMEOUT_S=3600` |

vLLM detects NVFP4 from the checkpoint, so no `--quantization` flag is needed.
The endpoint configuration sets 3,600-second model download and container
startup health-check timeouts. Synchronous `invoke_endpoint` calls time out
after 60 seconds, so use `invoke_endpoint_with_response_stream` for long
reasoning generations. Reasoning tokens count against `max_tokens`, so keep it
at 2,048 or more.

**If the endpoint fails with `InsufficientInstanceCapacity`**, retry later or
in another Region, or let SageMaker fall back to `ml.g7e.48xlarge`. To fall
back, create a second model with `SM_VLLM_TENSOR_PARALLEL_SIZE` set to `8`,
then replace `InstanceType` in the production variant with instance pools:

```python
"InstancePools": [
    {"InstanceType": "ml.g7e.24xlarge", "Priority": 1, "ModelNameOverride": model_name_tp4},
    {"InstanceType": "ml.g7e.48xlarge", "Priority": 2, "ModelNameOverride": model_name_tp8},
],
```

SageMaker tries the pools in priority order. No instance is provisioned for a
failed endpoint, but delete it before you retry.

The endpoint bills for as long as it exists. Run the cleanup cell when you are
done. The staged weights in S3 and the image in ECR persist and incur storage
cost until you delete them.

### Optional: DFlash2 speculative decoding

Stage the
[Beam-1.0-501B-A23B-DFlash2](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2)
drafter into a `dflash2/` folder next to the target weights, for example with
the same staging project:

```bash
aws codebuild start-build --project-name beam-weights \
  --environment-variables-override \
    name=HF_REPO,value=reflection/Beam-1.0-501B-A23B-DFlash2 \
    name=PREFIX,value=${WEIGHTS_PREFIX}/dflash2
```

Then add this setting to `vllm_env` in the notebook:

```python
"SM_VLLM_SPECULATIVE_CONFIG": '{"method":"dflash","model":"/opt/ml/model/dflash2","num_speculative_tokens":7,"enable_adaptive_verification":false}',
```

On SM120 GPUs, keep `enable_adaptive_verification` set to `false`. Do not force
`attention_backend`: FlashAttention rejects the FP8 KV cache on SM120, and vLLM
selects the Triton backend when you leave it unset.

### Validation and limitations

**SageMaker endpoint.** The sample was run end to end on a real-time endpoint
on `ml.g7e.24xlarge`: the weights and drafter were staged with the CodeBuild
job, the image was built with the image buildspec, and the endpoint was
deployed from the notebook cells. It reached `InService` in about 20 minutes,
including weight loading from S3 and engine warmup.

- The notebook's text (reasoning returned separately from the answer), tool
  calling, and streaming cells passed.
- Needle retrieval from a ~105K-token prompt passed.
- Natural-text prompts (up to 1,536 output tokens, temperature 0) through the
  SageMaker runtime, streaming: 93 output tokens/s per user at 1 concurrent
  request, and 65 per user (259 total) at 4 concurrent requests.

DFlash2 on the endpoint was not run: every attempt to provision `ml.g7e.24xlarge`
or `ml.g7e.48xlarge` with the drafter enabled failed with
`InsufficientInstanceCapacity` during testing. The `SM_VLLM_SPECULATIVE_CONFIG`
setting above uses the configuration validated on EC2 below with the same
image.

**Engine performance on Amazon EC2 `g7e.24xlarge`** with the same base image
and vLLM flags, served directly against the engine. Measured with
[AIPerf](https://github.com/ai-dynamo/aiperf) at 12.7K input and up to 5.3K
output tokens per request, streaming, with 16 / 32 / 64 requests at 2 / 4 / 8
concurrent requests:

| Concurrent requests | Time to first token | Inter-token latency | Output tokens/s per user | Total output tokens/s |
|---:|---:|---:|---:|---:|
| 2 | 1.33 s | 12.6 ms | 76 | 148 |
| 4 | 1.60 s | 15.6 ms | 62 | 231 |
| 8 | 2.09 s | 21.3 ms | 45 | 346 |

**DFlash2 on EC2 `g7e.24xlarge`**, measured with and without the drafter on the
same instance:

| Workload | Concurrency | Tokens/s per user without DFlash2 | With DFlash2 |
|---|---:|---:|---:|
| Natural-text prompts | 1 | 96 | 175 |
| Natural-text prompts | 4 | 67 | 103 |
| Synthetic AIPerf profile | 2 | 76 | 107 |
| Synthetic AIPerf profile | 4 | 62 | 75 |
| Synthetic AIPerf profile | 8 | 45 | 51 |

The drafter accepted 34% of drafted tokens on natural text. Synthetic AIPerf
prompts are random tokens and understate the speculative-decoding gain. Startup
with the drafter took about 7 minutes on EC2.

`ml.g7e.48xlarge` (TP8) is not faster at low concurrency: with the same AIPerf
profile on EC2 it measured 76 / 66 / 55 output tokens/s per user at 2 / 4 / 8
concurrent requests.

These results are indicative for this environment only, not general
performance claims.
