# Deploy Reflection Beam 1.0 NVFP4 on Amazon SageMaker AI

This sample deploys Reflection Beam 1.0 501B-A23B NVFP4 to an Amazon SageMaker
AI real-time endpoint, invokes it, runs a bounded smoke test, and cleans up the
endpoint.

The sample uses the
`huggingface-llm-beam-1-0-501b-a23b-nvfp4` model ID with
`SageMakerPublicHub`.

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

## Choose the model ID

The sample is configured for the NVFP4 checkpoint. To use another Beam
checkpoint, replace `JUMPSTART_MODEL_ID` in `jumpstart_real_time.py` with one of
these published IDs and select a supported instance type:

| Checkpoint | JumpStart model ID |
|---|---|
| 501B-A23B | `huggingface-llm-beam-1-0-501b-a23b` |
| 501B-A23B FP8 | `huggingface-llm-beam-1-0-501b-a23b-fp8` |
| 501B-A23B NVFP4 | `huggingface-llm-beam-1-0-501b-a23b-nvfp4` |

Keep credentials, account IDs, role ARNs, and endpoint names out of the source
file.

## Run the lifecycle CLI

1. **Run the deploy, smoke-test, and cleanup lifecycle.**

   ```bash
   python jumpstart_real_time.py \
     --endpoint-name reflection-beam-smoke
   ```

2. **Set a different positive startup timeout only when the released model requires it.**

   ```bash
   python jumpstart_real_time.py \
     --endpoint-name reflection-beam-smoke \
     --timeout-seconds 1200
   ```

The default `--timeout-seconds` value is `900`. The sample uses it for model
download, container health checks, and the endpoint waiter. It rejects zero,
negative, Boolean, or missing timeout values when you call the Python function
directly.

The CLI deploys an endpoint, runs `smoke_test()`, prints the returned text, and
then exits. It always attempts cleanup before printing the result. If cleanup
cannot complete, it reports the endpoint name and exits with an error so you can
inspect the endpoint in SageMaker. Use the Python functions directly when you
need to keep an endpoint running.

## Use the Python functions

| Function | Behavior |
|---|---|
| `deploy_endpoint(sagemaker_session=None, endpoint_name=None, timeout_seconds=900)` | Validates the public model ID and positive timeout, creates a `JumpStartModel`, deploys it synchronously, and returns its predictor. |
| `invoke_endpoint(predictor, prompt, max_tokens=64, timeout_seconds=120)` | Sends one chat request with bounded connection and read timeouts, then returns `choices[0].message.content` as text. |
| `smoke_test(predictor)` | Requests a short greeting with at most 64 output tokens and rejects empty output. |
| `cleanup_endpoint(predictor)` | Requests endpoint deletion, waits for deletion to complete, then deletes the endpoint configuration and model. Passing `None` is safe. |
| `run_lifecycle(sagemaker_session=None, endpoint_name=None, timeout_seconds=900)` | Deploys, smoke tests, and calls cleanup from `finally` on success or smoke-test failure. |

Use `try` and `finally` when you call the functions independently:

```python
from jumpstart_real_time import (
    cleanup_endpoint,
    deploy_endpoint,
    invoke_endpoint,
)

predictor = deploy_endpoint(
    endpoint_name="reflection-beam-development",
    timeout_seconds=900,
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
from removing resources that an endpoint still uses. If deployment fails before
returning a predictor, the sample uses the endpoint name chosen before
deployment to inspect and remove partial resources. Cleanup failures are
reported instead of suppressed.

## Run the tests

1. **Run the standard-library unit tests from this directory.**

   ```bash
   python -m unittest discover -s tests -v
   ```

The tests stub the SageMaker SDK and do not require AWS credentials or a live
endpoint. They verify the configured model ID, credential hygiene,
bounded deployment timeouts, request and response validation, smoke-test
behavior, and cleanup on success and failure.
