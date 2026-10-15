# Decision-model serving image on the PyTorch 2.14 DLC

A small Amazon SageMaker AI hosting image for the decision-model samples in this repository:
[Strands Decider 2B](../../01-models/StrandsAgents/strands-decider-2B/) and
[Decision 2.0 Sol 2B](../../01-models/vllm-sr/Decision-2.0-Sol-2B/) on GPU, and the CPU-hosted
[Decision 2.0 Kai 0.6B](../../01-models/vllm-sr/Decision-2.0-Kai-0.6B-cpu/) and
[Vela 2.0 0.3B](../../01-models/vllm-sr/Vela-2.0-0.3B-cpu/). It's built on the maintained
[AWS PyTorch 2.14 SageMaker DLC](https://github.com/aws/deep-learning-containers/blob/main/docs/pytorch/index.md)
(`pytorch:2.14-cu133-amzn2023-sagemaker` by default, or the CPU DLC; both supported until 28 September 2027).

## Why an image

The AWS PyTorch *inference* DLC line ends at 2.6, which reached end of support on 30 June 2026.
The newer PyTorch DLC is built for training: it has no inference toolkit and no `serve` entrypoint.
This image adds only the hosting contract that the samples need:

- `serve` answers `GET /ping` and `POST /invocations` on port 8080 with one process.
- At startup it installs the sample's `code/requirements.txt`, if present. pip honours `--no-index`
  lines, so network-isolated deployments work.
- It imports the sample's `code/inference.py` (`SAGEMAKER_PROGRAM` in `SAGEMAKER_SUBMIT_DIRECTORY`),
  loads the model once with `model_fn("/opt/ml/model")`, and passes each request to
  `transform_fn(model, body, content_type, accept)`.
- `GenericInferenceToolkitError` keeps its HTTP status, so a 422 from an adapter still reaches the
  caller as a `ModelError` with `OriginalStatusCode` 422. An error message that is itself a JSON
  object or array is returned as `application/json`, so an adapter that reports a structured error
  (Sol and Kai for `max_length_exceeded`, Vela for `too_long`) keeps its body and its media type.

The samples' adapters run unchanged.

## Per-request timeout

`SAGEMAKER_MODEL_SERVER_TIMEOUT` (a number of seconds) bounds how long a caller waits for one
request: when it expires the server answers `504` with a JSON body. Unset, or `0`, waits as long as
the caller does, and a value that is not a number fails at startup rather than silently. The Strands
Decider and Sol notebooks set 60 seconds, which is how long
[SageMaker AI allows a model container to respond](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html),
and they set their client socket timeout to the 70 seconds that page recommends alongside it.

A running Python thread cannot be cancelled, so this bounds the answer, not the work. The call keeps
running and, because the samples' adapters serialize on a lock, it holds that lock until it returns;
requests that arrive meanwhile time out too, until it finishes. Work that timed out still runs to
completion afterwards, in arrival order, with nobody waiting for it. So the timeout stops an
unbounded queue of callers, not an unbounded queue of inference: bound the work as well, by limiting
questions per request.

## GPU driver

PyTorch 2.14 in this image is built for CUDA 13, which ships with NVIDIA driver 580. Create the endpoint with
`InferenceAmiVersion` set to `al2023-ami-sagemaker-inference-gpu-4-1` (driver 580, CUDA 13.0). It is compatible with `ml.g4dn`, `ml.g5`,
`ml.g6` and `ml.g6e`, and the sample notebooks set it for you.

The default AMIs for `ml.g5` and `ml.g4dn` have driver 470 (CUDA 11.4). On both `ml.g5.xlarge` and
`ml.g4dn.xlarge` the default AMI failed with `CannotStartContainerError`; with the driver 580 AMI the same image
started and answered (A10G and T4). The default `ml.g6` AMI worked through the DLC's CUDA forward compatibility,
but set the AMI explicitly anyway.

## CPU variant

The same Dockerfile builds a CPU image from the PyTorch 2.14 **CPU** SageMaker DLC
(`pytorch:2.14-cpu-amzn2023-sagemaker`, also supported until 28 September 2027), which the
[Decision 2.0 Kai 0.6B](../../01-models/vllm-sr/Decision-2.0-Kai-0.6B-cpu/) and
[Vela 2.0 0.3B](../../01-models/vllm-sr/Vela-2.0-0.3B-cpu/) CPU samples use on `ml.c7i` instances. Pass the base through the `BASE_IMAGE` build argument; `build_image.py`
exposes it and tags the variant separately in the shared repository:

```bash
python build_image.py --region us-east-1 --bucket <artifact-bucket> \
  --codebuild-role-arn arn:aws:iam::<account>:role/<codebuild-role> \
  --base-image public.ecr.aws/deep-learning-containers/pytorch:2.14-cpu-amzn2023-sagemaker \
  --tag-prefix pt214-cpu
```

The CPU DLC has the same `/usr/local/bin/entrypoint.sh`, which only applies the CUDA compatibility
check when a CUDA runtime is present, so `serve` starts unchanged. CPU endpoints do not need
`InferenceAmiVersion`. The CPU image is smaller than the CUDA one because the base carries no
CUDA libraries, and the shared Python pins are the same.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | `FROM` the PyTorch 2.14 SageMaker DLC (`BASE_IMAGE` build argument; CUDA by default); adds the shared libraries and `serve` |
| `requirements.txt` | Pinned Hugging Face libraries, `sagemaker-inference`, and the server; PyTorch comes from the base |
| `serve`, `serve.py` | The hosting entrypoint and server |
| `buildspec.yml` | AWS CodeBuild build and push; forwards `BASE_IMAGE` when set |
| `build_image.py` | Uploads the build context, creates or updates the CodeBuild project, waits, and returns the digest-pinned ECR image URI |

## Build

The sample notebooks build the image for you. To build it on its own:

```bash
python build_image.py --region us-east-1 --bucket <artifact-bucket> \
  --codebuild-role-arn arn:aws:iam::<account>:role/<codebuild-role>
```

- The CodeBuild service role needs read access to `s3://<artifact-bucket>/decision-model-serving/*`, CloudWatch Logs write access, `ecr:GetAuthorizationToken`, and push and describe access on the `decision-model-serving` ECR repository.
- The caller also needs permission to create that repository and the CodeBuild project.
- The build runs on `BUILD_GENERAL1_LARGE` in about 7 minutes.

## Notes

- The image is about 10 GB compressed, because the base DLC includes the PyTorch training stack (DeepSpeed, Transformer Engine, EFA). Each build pushes a new tag, and ECR bills for the storage; delete old images you no longer use.
- The base image tag is rebuilt for security patches, so two builds can differ. The notebooks use the digest-pinned URI that `build_image.py` returns.
- To remove everything the build created: `aws ecr delete-repository --repository-name decision-model-serving --force` and `aws codebuild delete-project --name decision-model-serving-build`.
- Like the samples, this image is an evaluation example. Review the dependencies and the image before production use.
