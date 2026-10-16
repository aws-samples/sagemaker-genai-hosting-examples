# Deploy Reflection Beam 1.0 with Amazon SageMaker JumpStart

This sample deploys the BF16, FP8, or NVFP4 variant of Reflection Beam 1.0
501B-A23B to an Amazon SageMaker AI real-time endpoint, invokes it, runs a
bounded smoke test, and cleans up the endpoint.

The sample uses the released model IDs from `SageMakerPublicHub`. It defaults to
NVFP4; use `--variant fp8` or `--variant bf16` to select another variant.

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
