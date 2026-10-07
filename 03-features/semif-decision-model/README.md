# Deploy decision models with llama.cpp on Amazon SageMaker AI

Use the AWS Deep Learning Container (DLC) for llama.cpp to deploy a model on an
Amazon SageMaker AI real-time endpoint. This sample provides two paths:

1. Deploy the published DLC directly for text generation through the
   OpenAI-compatible `llama-server` Chat Completions API. You do not need to
   build a custom image for this path.
2. Build an optional child image with
   [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev). This path
   returns probabilities for a declared set of choices, which applications can
   use for routing, prioritization, policy selection, and other structured
   decisions.

Both paths use the same model packaged in GGUF, a file format designed for
efficient inference with llama.cpp. Start with the direct text-generation
deployment to verify the model and DLC. Build the SemIf image only when your
application needs structured option scores and lower-level access to model
logits and state.

## How it works

1. AWS CodeBuild downloads pinned model and tokenizer revisions,
   creates `model.tar.gz`, and uploads it to Amazon Simple Storage Service
   (Amazon S3).
2. The text-generation path points a SageMaker model directly at the published
   DLC and the S3 model archive.
3. The optional SemIf path uses CodeBuild to build a child image and push it
   to Amazon Elastic Container Registry (Amazon ECR).
4. SageMaker AI downloads the model archive and starts the selected
   digest-pinned container image.

When several decisions use the same exact state, send them together. SemIf
prefills that state once and restores the saved llama.cpp state before scoring
each question.

The published DLC maps `/invocations` to its OpenAI-compatible
`/v1/chat/completions` API. That API does not expose the full-vocabulary logits
and state operations that SemIf uses. The child image calls SemIf and the DLC's
`libllama` libraries in-process. SemIf support is an extension built by this
sample, not a built-in endpoint mode in the DLC.

## Prerequisites

- Python 3.10 or later and the AWS CLI.
- An AWS account with permissions for SageMaker AI, Amazon ECR, Amazon S3,
  CodeBuild, AWS Identity and Access Management (IAM), and Amazon CloudWatch.
- An S3 bucket in your deployment Region.
- A CodeBuild service role that can read the source object, write build logs,
  obtain an ECR authorization token, and create, describe, and push images.
- A SageMaker execution role that can read the model object and pull the
  private ECR image.
- Quota for an x86 CPU or NVIDIA GPU SageMaker real-time endpoint. Start with
  `ml.c6i.2xlarge` for CPU. For GPU, use an instance pool so SageMaker AI can
  try compatible instance types in your preferred order.

Create purpose-specific IAM roles with the permissions required by your
environment. Do not reuse broad development roles for a production deployment.
The endpoint and CodeBuild jobs incur charges in your account.

## Install the deployment tools

Clone the repository and enter this sample directory:

```bash
git clone https://github.com/aws-samples/sagemaker-genai-hosting-examples.git
cd sagemaker-genai-hosting-examples/03-features/semif-decision-model
```

Create a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r deploy/requirements.txt
```

```bash
export AWS_REGION=your-region
export ARTIFACT_BUCKET=your-artifact-bucket
export CODEBUILD_ROLE_ARN=arn:aws:iam::123456789012:role/semif-codebuild-role
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/semif-sagemaker-role
```

## Package the model with CodeBuild

Start a CodeBuild job that downloads the pinned Qwen3.5-4B GGUF and tokenizer
revisions and writes the SageMaker model archive to your bucket:

```bash
python deploy/prepare_model_build.py \
  --region "$AWS_REGION" \
  --bucket "$ARTIFACT_BUCKET" \
  --codebuild-role-arn "$CODEBUILD_ROLE_ARN" \
  --model-s3-uri "s3://$ARTIFACT_BUCKET/semif-llamacpp/model.tar.gz"
```

The command returns a build ID. Check the build until it reaches `SUCCEEDED`:

```bash
aws codebuild batch-get-builds \
  --region "$AWS_REGION" \
  --ids BUILD_ID
```

You can also run `deploy/package_model.py` locally, but the download and
archive require several GiB of disk space. The model build records the exact
source revisions and checksums in `model-manifest.json` and rejects a GGUF
whose SHA-256 checksum differs from the value validated with this sample.

## Deploy the DLC for text generation

Resolve a digest-pinned CPU image URI from the
[AWS DLC for llama.cpp documentation](https://aws.github.io/deep-learning-containers/llama-cpp/),
then create the SageMaker model, endpoint configuration, and endpoint:

```bash
export STANDARD_IMAGE_URI=public.ecr.aws/deep-learning-containers/llama-cpp@sha256:IMAGE_DIGEST

python deploy/deploy_standard_endpoint.py \
  --region "$AWS_REGION" \
  --image-uri "$STANDARD_IMAGE_URI" \
  --model-data-url "s3://$ARTIFACT_BUCKET/semif-llamacpp/model.tar.gz" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --instance-type ml.c6i.2xlarge \
  --name llamacpp-standard
```

The DLC auto-detects the first `.gguf` file in the extracted model archive.
Invoke the endpoint with a standard OpenAI Chat Completions request:

```bash
python deploy/invoke_standard_endpoint.py \
  --region "$AWS_REGION" \
  --endpoint-name llamacpp-standard
```

The request uses `messages`, `max_tokens`, and `stream`. The response uses the
OpenAI-compatible `choices[].message.content` shape:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "A customer cannot receive a password reset email. Reply with the best support queue."
    }
  ],
  "max_tokens": 64,
  "stream": false
}
```

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Account access support."
      }
    }
  ]
}
```

The response above is abbreviated. The endpoint returns the remaining
OpenAI-compatible completion metadata. See `example-standard-request.json` for
the complete example request.

## Build the optional SemIf child image

Start the CPU container build:

```bash
python deploy/prepare_build.py \
  --region "$AWS_REGION" \
  --bucket "$ARTIFACT_BUCKET" \
  --codebuild-role-arn "$CODEBUILD_ROLE_ARN" \
  --image-tag cpu
```

For a CUDA image, select the GPU Dockerfile and use a distinct tag:

```bash
python deploy/prepare_build.py \
  --region "$AWS_REGION" \
  --bucket "$ARTIFACT_BUCKET" \
  --codebuild-role-arn "$CODEBUILD_ROLE_ARN" \
  --dockerfile Dockerfile.gpu \
  --image-tag gpu
```

Wait for both builds to succeed. Resolve each tag to its immutable digest so
the CPU and GPU deployment commands cannot use the wrong image:

```bash
export ECR_REPOSITORY_URI="$(
  aws ecr describe-repositories \
    --region "$AWS_REGION" \
    --repository-names semif-llamacpp-dlc \
    --query 'repositories[0].repositoryUri' \
    --output text
)"

export CPU_IMAGE_DIGEST="$(
  aws ecr describe-images \
    --region "$AWS_REGION" \
    --repository-name semif-llamacpp-dlc \
    --image-ids imageTag=cpu \
    --query 'imageDetails[0].imageDigest' \
    --output text
)"

export GPU_IMAGE_DIGEST="$(
  aws ecr describe-images \
    --region "$AWS_REGION" \
    --repository-name semif-llamacpp-dlc \
    --image-ids imageTag=gpu \
    --query 'imageDetails[0].imageDigest' \
    --output text
)"

export CPU_IMAGE_URI="${ECR_REPOSITORY_URI}@${CPU_IMAGE_DIGEST}"
export GPU_IMAGE_URI="${ECR_REPOSITORY_URI}@${GPU_IMAGE_DIGEST}"
```

The deployment script rejects mutable image tags.

## Choose CPU or GPU for the SemIf endpoint

Use CPU when the workload is not latency-sensitive and can tolerate seconds of
response time. Use GPU for interactive applications, long prompts, or
workloads with tight tail-latency targets. The instance types below are
starting points, not sizing recommendations. Benchmark your own prompts,
decision schemas, concurrency, and cost requirements.

### Deploy on CPU

Create the SageMaker model, endpoint configuration, and endpoint:

```bash
python deploy/deploy_endpoint.py \
  --region "$AWS_REGION" \
  --image-uri "$CPU_IMAGE_URI" \
  --model-data-url "s3://$ARTIFACT_BUCKET/semif-llamacpp/model.tar.gz" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --instance-type ml.c6i.2xlarge
```

Record the returned endpoint name and wait for it to enter `InService`:

```bash
aws sagemaker wait endpoint-in-service \
  --region "$AWS_REGION" \
  --endpoint-name ENDPOINT_NAME
```

The default `ml.c6i.2xlarge` instance keeps the walkthrough cost lower. For
more CPU throughput, test a larger instance and set the llama.cpp thread count:

```bash
python deploy/deploy_endpoint.py \
  --region "$AWS_REGION" \
  --image-uri "$CPU_IMAGE_URI" \
  --model-data-url "s3://$ARTIFACT_BUCKET/semif-llamacpp/model.tar.gz" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --instance-type ml.c6i.8xlarge \
  --threads 32
```

In the validated run, more CPU threads improved median latency, but long
prompts still dominated tail latency.

### Deploy on GPU with an instance pool

Deploy the CUDA image with an ordered instance pool:

```bash
python deploy/deploy_endpoint.py \
  --region "$AWS_REGION" \
  --image-uri "$GPU_IMAGE_URI" \
  --model-data-url "s3://$ARTIFACT_BUCKET/semif-llamacpp/model.tar.gz" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --instance-pool ml.g6e.xlarge \
  --instance-pool ml.g6.2xlarge \
  --instance-pool ml.g5.2xlarge \
  --instance-pool ml.g4dn.2xlarge \
  --gpu-layers=-1
```

SageMaker AI tries the compatible instance types in order during endpoint
provisioning and selects one available type for the production variant. An
instance pool improves placement flexibility when your first preference is
unavailable. It does not load balance inference across the listed instance
types.

## Compare the validated CPU and GPU configurations

The following results use the 231-case public
[JevBench](https://github.com/fstandhartinger/jevbench) dataset with the same
model artifacts and request path. Each configuration processed the full
dataset once. Treat these figures as a measured sample comparison, not a
service-level performance claim.

| Configuration | AWS DLC for llama.cpp | Correct | Client p50 | Client p95 |
|---|---|---:|---:|---:|
| `ml.c6i.8xlarge`, 32 CPU threads | [`llama-cpp:server-sagemaker-cpu-v1`](https://aws.github.io/deep-learning-containers/llama-cpp/) | 179/231 (77.5%) | 1.715 seconds | 19.291 seconds |
| `ml.g6e.xlarge`, all model layers on GPU | [`llama-cpp:server-sagemaker-cuda-v1`](https://aws.github.io/deep-learning-containers/llama-cpp/) | 180/231 (77.9%) | 0.352 seconds | 0.599 seconds |

Compared with the tuned CPU configuration, the GPU configuration reduced
median latency by 4.9 times and p95 latency by 32.2 times. Accuracy differed by
one case, so the result does not establish a quality advantage.

## Invoke the SemIf endpoint

Send one JSON object with exactly one of `decision` or `decisions`. A
`decision` declares its context, question, and allowed options. SemIf returns
conditional probabilities in the same option order.

### Single-decision input

```json
{
  "decision": {
    "id": "support-route",
    "state": "The customer cannot sign in.",
    "question": "Which queue should handle this request?",
    "options": [
      {"id": "account-access", "description": "Account access support."},
      {"id": "billing", "description": "Billing support."}
    ]
  }
}
```

The fields are:

- `id`: A caller-defined identifier returned unchanged.
- `state`: The unstructured context that the model interprets.
- `question`: The decision that the model evaluates.
- `options`: An ordered list of stable `id` and `description` pairs.

Invoke the endpoint with the included example:

```bash
python deploy/invoke_endpoint.py \
  --region "$AWS_REGION" \
  --endpoint-name ENDPOINT_NAME
```

### Single-decision output

```json
{
  "result": {
    "id": "support-route",
    "option_ids": ["account-access", "billing"],
    "probabilities": [0.98, 0.02]
  }
}
```

The full response can also include raw option logits, prompt details, timing,
and pinned model metadata from SemIf. `option_ids`, `probabilities`, and
`option_logits` use the same order.

Use the probabilities with application-level thresholds and fallback rules.
Do not treat them as a substitute for authorization or deterministic policy
enforcement.

### Score several decisions over one state

For shared-state scoring, replace `decision` with a nonempty `decisions` list.
Use this mode when your application asks several questions about the same
document, conversation, or agent state. Every `state` value must be exactly
equal and every decision ID must be unique. The endpoint accepts at most 8
decisions by default.

```bash
python deploy/invoke_endpoint.py \
  --region "$AWS_REGION" \
  --endpoint-name ENDPOINT_NAME \
  --request example-shared-request.json
```

The endpoint returns one result per decision and aggregate timing for the
shared operation:

```json
{
  "results": [
    {
      "id": "route-request",
      "option_ids": ["account-access", "billing"],
      "probabilities": [0.98, 0.02]
    },
    {
      "id": "set-priority",
      "option_ids": ["urgent", "standard"],
      "probabilities": [0.91, 0.09]
    }
  ],
  "timing": {
    "batch_size": 2,
    "prefix_tokens": 22,
    "prefill_seconds": 0.31,
    "suffix_forward_seconds": 0.08,
    "max_request_tokens": 8192,
    "max_tokens_per_decision": 4096,
    "total_seconds": 0.42
  }
}
```

The abbreviated values above illustrate the response shape; your endpoint
returns the full model metadata and measured timings. Shared-state scoring
improves throughput when decisions reuse a long state. It does not reduce the
latency of a single decision with a unique state.

### Handle endpoint backpressure

The container admits one model operation at a time by default because the
llama.cpp backend owns one stateful context. It returns HTTP `503` with
`"retryable": true` when another operation is active. Retry with capped
exponential backoff and jitter, or scale the SageMaker endpoint across
instances for more concurrent capacity.

Set `SEMIF_MAX_IN_FLIGHT` only if you deliberately want a small bounded queue.
Increasing it does not make the single model context execute in parallel.

### Understand the request and runtime guardrails

The SemIf adapter requires a declared `Content-Length` between 1 byte and
256 KiB and accepts at most 8 decisions in one shared request by default.
Configure these limits with the positive integer environment variables
`SEMIF_MAX_BODY_BYTES` and `SEMIF_MAX_SHARED_DECISIONS`.

`SEMIF_MAX_REQUEST_TOKENS` provides a request-wide prompt allowance of 8,192
tokens by default. A single decision can use up to the lower of
`SEMIF_MAX_TOKENS` and `SEMIF_MAX_REQUEST_TOKENS`. For a shared request with
`N` decisions, every prompt must fit within the lower of `SEMIF_MAX_TOKENS`
and `floor(SEMIF_MAX_REQUEST_TOKENS / N)`. The adapter returns HTTP `400`
instead of truncating a prompt that exceeds this per-decision limit. This
allocation caps the sum of the prompt allowances even though shared scoring
computes the common state prefix once. Shared responses report
`max_request_tokens` and `max_tokens_per_decision` in `timing`.

After parsing the JSON and before constructing prompts, the adapter recursively
rejects every case-sensitive special token registered by the pinned tokenizer,
including `<|im_start|>`, `<|im_end|>`, and `<|endoftext|>`. It checks object
keys and values anywhere inside `decision` or `decisions`, including IDs,
state, questions, and options.

Both deployment scripts enable SageMaker network isolation, so the inference
containers cannot make outbound network calls at runtime. Package the model
and tokenizer artifacts in `model.tar.gz`; do not depend on runtime downloads.

## Clean up

Run the cleanup command once for each endpoint name that you deployed. It
deletes the endpoint, endpoint configuration, and model with that exact name:

```bash
python deploy/cleanup_endpoint.py \
  --region "$AWS_REGION" \
  --name ENDPOINT_NAME
```

The cleanup script removes only those three explicitly named SageMaker
resources. Delete the CodeBuild projects, ECR repository, model object, build
source objects, logs, and IAM roles separately when you no longer need them.

## Limitations and design constraints

- **CPU and GPU options:** the CPU image defaults to `n_gpu_layers=0`. The CUDA
  image uses `--gpu-layers=-1` to offload every compatible layer and verifies
  that the CUDA backend loaded before it accepts traffic.
- **One serialized model context:** the HTTP server accepts concurrent
  connections, but one stateful llama.cpp context performs scoring. The
  container rejects excess work with HTTP `503` by default instead of building
  an unbounded request queue. Scale out with endpoint instances for concurrent
  capacity.
- **Shared-state scope:** sending several decisions in `decisions` reuses one
  exact state prefix inside that request. This improves repeated-state
  throughput and queue latency, not unique-state single-request latency.
- **Pinned compatibility:** each Dockerfile pins its AWS DLC, SemIf, and
  llama-cpp-python revisions. Native llama.cpp structures change. Update the
  pins together and rerun end-to-end CPU and GPU tests before upgrading.
- **Model-specific contract:** SemIf verifies tokenizer compatibility and
  requires each answer letter to map to one shared token. Test other GGUF
  models before substituting them.
- **No dynamic batching:** this sample does not combine unrelated requests
  into a model batch. Scale the endpoint across instances when the workload
  needs more concurrent capacity.
- **Not a policy engine:** model probabilities are nondeterministic signals.
  Keep authorization, compliance rules, and other hard constraints in
  deterministic application code.
- **SemIf-specific API:** the structured request and response schema in this
  sample is not a drop-in replacement for another decision-model API. Adapt
  your client or add a compatibility layer when migrating an existing
  application.

## Test

Run the focused unit tests without downloading the model:

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q
```

## Third-party software

This sample downloads SemIf-OpenJev, llama-cpp-python, Qwen3.5-4B, and a
community GGUF conversion. Review each upstream repository, model card,
license, acceptable-use terms, and security posture before production use.
AWS does not maintain those third-party projects or model artifacts.
Scan the final child image in Amazon ECR and remediate findings before
production deployment.
