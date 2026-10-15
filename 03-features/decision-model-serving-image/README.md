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

The samples' adapters run unchanged. `transform_fn` is called with four arguments and no
TorchServe-style `context`, so an adapter that declares an optional `context=None` takes its
`context is None` branch here. Those adapters keep the parameter because the SageMaker inference
toolkit does pass a `context`, and they are meant to stay runnable under it.

## Per-request timeout

`SAGEMAKER_MODEL_SERVER_TIMEOUT` (a number of seconds) bounds how long a caller waits for one
request: when it expires the server answers `504` with a JSON body. Unset, or `0`, waits as long as
the caller does. Anything that is not a finite number of seconds, zero or greater, fails at startup
with the reason rather than silently: a negative value would answer `504` immediately, and `nan` or
`inf` would either fail every request or remove the bound.

The variable name is the inference toolkit's, for the same purpose, but the mechanism differs and
it is worth knowing which one you have. The toolkit passes it to TorchServe as
`default_response_timeout`, which declares the backend worker unresponsive and reboots it; that
does end the work, at the cost of reloading the model and dropping everything else in flight. This
server instead answers the caller and leaves the call running, as described below.

The notebooks set 55 seconds, inside the 60 seconds
[SageMaker AI allows a model container to respond](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html),
and they set their client socket timeout to the 70 seconds that page recommends alongside the
60-second limit. The 5 seconds of headroom are this sample's choice, not an AWS-published number:
the documentation gives the model "a maximum processing time of 60 seconds" and asks callers to
allow 70, but it does not say when the platform's own clock starts relative to the container
receiving the request. Since a container timeout of exactly 60 leaves nothing for the response to
travel in, a value below 60 is the safe reading. At 55 the `504` arrives as a `ModelError` with
`OriginalStatusCode` 504, the same way an adapter's 422 does; past the platform's deadline the
caller gets a SageMaker-authored timeout message instead, with no `OriginalStatusCode` to inspect.
Raising the limit is not an option: it is fixed for real-time endpoints.

Real-time endpoints are what that 60-second limit applies to. Asynchronous Inference allows
[up to one hour](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html) per request,
and Batch Transform has its own configurable per-invocation timeout, so this server assumes no
platform limit of its own: it is unbounded unless the variable is set, and the deployment chooses a
value to suit the endpoint type. AWS's own guidance on reusing a real-time container
asynchronously is to confirm the container's timeouts can handle the longer limits.

A running Python thread cannot be cancelled, so this bounds the answer, not the work. The call keeps
running and, because the samples' adapters serialize on a lock, it holds that lock until it returns;
requests that arrive meanwhile may time out as well, until it finishes. Work that timed out still
runs to completion afterwards with nobody waiting for it, and `threading.Lock` does not promise the
waiting threads any particular order. So the timeout stops an unbounded queue of callers, not an
unbounded queue of inference: bound the work as well, by limiting questions per request.

## GPU driver

PyTorch 2.14 in this image is built for CUDA 13, which ships with NVIDIA driver 580. Create the endpoint with
`InferenceAmiVersion` set to `al2023-ami-sagemaker-inference-gpu-4-1` (driver 580, CUDA 13.0). It is compatible with `ml.g4dn`, `ml.g5`,
`ml.g6` and `ml.g6e`, and the sample notebooks set it for you.

The default AMIs tested for `ml.g5` and `ml.g4dn` had driver 470 (CUDA 11.4). On both `ml.g5.xlarge`
and `ml.g4dn.xlarge` that default failed with `CannotStartContainerError`; with the driver 580 AMI the
same image started and answered (A10G and T4). The `ml.g6` default tested worked through the DLC's
CUDA forward compatibility, but set the AMI explicitly anyway: defaults change.

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
