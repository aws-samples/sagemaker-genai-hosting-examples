# Decision-model serving image on the PyTorch 2.14 DLC

A small Amazon SageMaker AI hosting image for the decision-model samples in this repository, such as
[Strands Decider 2B](../../01-models/StrandsAgents/strands-decider-2B/). It's built on the maintained
[AWS PyTorch 2.14 SageMaker DLC](https://github.com/aws/deep-learning-containers/blob/main/docs/pytorch/index.md)
(`pytorch:2.14-cu133-amzn2023-sagemaker`, supported until 28 September 2027).

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
  caller as a `ModelError` with `OriginalStatusCode` 422.

The samples' adapters run unchanged. The DLC entrypoint still applies CUDA forward compatibility, so
the image runs on the default SageMaker GPU AMI.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | `FROM` the PyTorch 2.14 SageMaker DLC; adds the shared libraries and `serve` |
| `requirements.txt` | Pinned Hugging Face libraries, `sagemaker-inference`, and the server; PyTorch comes from the base |
| `serve`, `serve.py` | The hosting entrypoint and server |
| `buildspec.yml` | AWS CodeBuild build and push |
| `build_image.py` | Uploads the build context, creates or updates the CodeBuild project, waits, and returns the digest-pinned ECR image URI |

## Build

The sample notebooks build the image for you. To build it on its own:

```bash
python build_image.py --region us-east-1 --bucket <artifact-bucket> \
  --codebuild-role-arn arn:aws:iam::<account>:role/<codebuild-role>
```

- The CodeBuild service role needs read access to `s3://<artifact-bucket>/decision-model-serving/*`, CloudWatch Logs write access, `ecr:GetAuthorizationToken`, and push and describe access on the `decision-model-serving` ECR repository.
- The caller also needs permission to create that repository and the CodeBuild project.
- The build runs on `BUILD_GENERAL1_LARGE` in about 5 minutes.

## Notes

- The image is about 10 GB compressed, because the base DLC includes the PyTorch training stack (DeepSpeed, Transformer Engine, EFA). Endpoint startup still beats installing PyTorch at startup, which the previous samples did.
- Like the samples, this image is an evaluation example. Review the dependencies and the image before production use.
