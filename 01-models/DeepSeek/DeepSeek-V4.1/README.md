# Deploy DeepSeek-V4.1 on Amazon SageMaker AI

This example deploys **DeepSeek-V4.1** series models on an Amazon SageMaker AI real-time endpoint using the **AWS vLLM Deep Learning Container**, with weights you stage to S3.

## Models

DeepSeek-V4.1 is a family of multimodal (image + text input) Mixture-of-Experts (MoE) language models supporting a context length of **one million tokens**.

| Model | Backbone Parameters | Activated Parameters | Weights |
| :--- | :--- | :--- | :--- |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 552B (+196B Engram memory) | 8B prefill / 16B decode | FP8 + FP4 experts, ~475 GiB |

### Key Architecture Highlights

- **Causal Encoder-Decoder (CED)** — a 20-layer causal encoder followed by a 20-layer decoder whose global KV cache is projected from the encoder output, so only 8B parameters are active per token during prefill (16B during decode).
- **Compressed Sparse Attention 2 (CSA2)** — layers share KV and reuse sparse-attention indices; with FP4 KV caching the global KV footprint is ~890 bytes per token, which keeps 1M-token contexts tractable.
- **Mixture-of-Experts + Engram memory** — 1 shared and 384 routed experts per MoE layer (6 active per token), plus a 196B-parameter Engram conditional memory accessed by token lookup.
- **Natively quantized checkpoint** — FP8 weights with FP4 experts, so vLLM loads them without a separate quantization step.

See the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) for details and benchmarks.

## Requirements

- **Serving container: the AWS vLLM Deep Learning Container (DLC).** DeepSeek-V4.1-Flash is registered in vLLM starting with **v0.30.0**, which ships in the SageMaker vLLM DLC (image tag `0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker`). SageMaker pulls it directly from the AWS DLC registry (account `763104351884`) — there is nothing to build.
- **Instance: `ml.p6-b200.48xlarge`** (8× NVIDIA B200 GPUs). The ~475 GiB checkpoint is ~60 GiB/GPU at tensor-parallel 8, leaving most of the B200's 192 GiB/GPU of HBM for the KV cache; the Engram memory table is offloaded to host memory (`VLLM_PLE_CPU_OFFLOAD=1`). A capacity reservation is recommended for availability.
- **Inference AMI: `al2023-ami-sagemaker-inference-gpu-4-1`** (driver 580 / CUDA 13). The DLC is built against CUDA 13 (`cu130`) and requires this driver; the default SageMaker GPU AMI ships an older CUDA 12 driver.
- **AWS account** with quota for the instance above, and an IAM role with SageMaker execution permissions (with read access to the weights bucket). The model weights are public (MIT license); a HuggingFace token is optional and only raises download rate limits.

## Repository layout

```
.
├── README.md
├── DeepSeek-V4.1.ipynb         # deploy + inference notebook (DeepSeek-V4.1-Flash)
└── buildspec-weights.yml       # CodeBuild: stream the weights HF Hub -> S3
```

## One-time setup: stage the weights to S3

The ~475 GiB checkpoint is streamed straight from the HuggingFace Hub to S3 by an **AWS CodeBuild** job, so you do not need local disk or bandwidth. The job copies files in parallel, checks every object's size against the HuggingFace metadata, and skips files already staged, so it is safe to re-run. Run the steps below once from a shell with AWS credentials (the notebook only verifies the weights are present). Set these shell variables first:

```bash
export AWS_REGION=<YOUR_REGION>                   # same region as the endpoint
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET=<YOUR_BUCKET>                       # must exist, same region
export HF_REPO=deepseek-ai/DeepSeek-V4.1-Flash
export WEIGHTS_PREFIX=models/deepseek-v41-flash
```

Create a CodeBuild service role once, scoped to your bucket and the CodeBuild log groups, and upload the buildspec as the build source:

```bash
aws iam create-role --role-name deepseek-v41-flash-codebuild \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name deepseek-v41-flash-codebuild --policy-name weights-staging \
  --policy-document '{"Version":"2012-10-17","Statement":[
    {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload","s3:ListBucket"],
     "Resource":["arn:aws:s3:::'"${BUCKET}"'","arn:aws:s3:::'"${BUCKET}"'/*"]},
    {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
     "Resource":"arn:aws:logs:'"${AWS_REGION}"':'"${ACCOUNT_ID}"':log-group:/aws/codebuild/*"}]}'
export CB_ROLE_ARN=$(aws iam get-role --role-name deepseek-v41-flash-codebuild --query Role.Arn --output text)
sleep 15  # IAM propagation

zip /tmp/deepseek-v41-src.zip buildspec-weights.yml
aws s3 cp /tmp/deepseek-v41-src.zip s3://${BUCKET}/build/deepseek-v41-src.zip
```

Then create and run the weight-staging build. On `BUILD_GENERAL1_LARGE` the full ~475 GiB copy takes about 20 minutes; the 8-hour timeout leaves headroom for slower HuggingFace throughput (the default 60 minutes can be too short):

```bash
aws codebuild create-project \
  --name deepseek-v41-flash-weights \
  --source "type=S3,location=$BUCKET/build/deepseek-v41-src.zip,buildspec=buildspec-weights.yml" \
  --artifacts type=NO_ARTIFACTS \
  --service-role "$CB_ROLE_ARN" \
  --timeout-in-minutes 480 \
  --environment "type=LINUX_CONTAINER,image=aws/codebuild/amazonlinux-x86_64-standard:5.0,computeType=BUILD_GENERAL1_LARGE,environmentVariables=[{name=HF_REPO,value=${HF_REPO}},{name=BUCKET,value=${BUCKET}},{name=PREFIX,value=${WEIGHTS_PREFIX}}]"

aws codebuild start-build --project-name deepseek-v41-flash-weights
```

Result: the uncompressed checkpoint under `s3://${BUCKET}/models/deepseek-v41-flash/`. Use this prefix as `weights_s3_uri` in the notebook. SageMaker mounts it read-only at `/opt/ml/model` on the endpoint.

> **Optional HF token.** The weights are public, so no token is required. To raise HuggingFace rate limits, store a token in AWS Secrets Manager and add it to the project as a `SECRETS_MANAGER` environment variable named `HF_TOKEN` (and grant the role `secretsmanager:GetSecretValue` on that secret) rather than passing it in plain text.

## Deploy

Open [`DeepSeek-V4.1.ipynb`](DeepSeek-V4.1.ipynb) in SageMaker Studio (or any Jupyter environment with AWS credentials), set `weights_s3_uri` from the setup above, and run the cells:

| Section | Description |
| :--- | :--- |
| One-time setup | Verify the weights staged to S3 (staging itself is the CLI above) |
| Container | Point at the AWS vLLM DLC and configure the serving environment |
| Deployment | Create the SageMaker Model (with the S3 weights), Endpoint Configuration, and Endpoint |
| Text Inference | Basic text generation |
| Reasoning | Inference with thinking/reasoning enabled and configurable effort |
| Streaming | Streaming response with tokens-per-second metrics |
| Cleanup | Delete the endpoint and associated resources |

The endpoint's `inference_image` is the DLC URI `763104351884.dkr.ecr.<region>.amazonaws.com/vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker`. Cold start (image pull + weight load + engine warmup) can take up to ~60 minutes; the endpoint config sets a 3600s startup health-check timeout.

## Cost

Real-time endpoints bill for the instance for as long as the endpoint exists, regardless of traffic. Run the **Cleanup** cell when you are done to stop charges. There is no scale-to-zero for classic real-time endpoints. The staged weights in S3 persist and incur storage cost until you delete them.
