# Deploy PolyAI Dialog-RSN-1 on Amazon SageMaker AI

This directory contains examples for deploying the PolyAI Dialog-RSN-1 model
package from
[AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-l4qyhvfxnkq7e)
to an Amazon SageMaker AI real-time endpoint and talking to it over a
bidirectional HTTP/2 stream.

## Contents

| File | Description |
| --- | --- |
| [`Dialog-RSN-1.ipynb`](Dialog-RSN-1.ipynb) | Notebook example that deploys the model package to a SageMaker real-time endpoint, asks it one question over HTTP/2, and deletes the created resources. |
| [`sample.py`](sample.py) | Python script that runs the same flow as the notebook end to end. |

## Prerequisites

- Python 3.12 or later. The HTTP/2 client requires it.
- A subscription to the
  [Dialog-RSN-1 listing on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-l4qyhvfxnkq7e).
  Subscribing is free until an endpoint runs. The examples will not work
  without this subscription.
- An existing SageMaker AI execution role.
- Service quota for one `ml.p4d.24xlarge` (or `ml.p5.48xlarge`) endpoint
  instance in your Region.
- An AWS identity with permissions to create, describe, invoke, and delete
  SageMaker endpoints, endpoint configurations, and models.

Install the dependencies:

```bash
pip install boto3 "aws-sdk-sagemaker-runtime-http2[awscrt]"
```

## Configuration

Before running either example, update the configuration values at the top of
`sample.py` or in the notebook's Configuration cell:

| Setting | Default | What to change |
| --- | --- | --- |
| `REGION` | `us-west-2` | The Region you want to deploy in. |
| `EXECUTION_ROLE_ARN` | `arn:aws:iam::ACCOUNT_ID:role/SAGEMAKER_EXECUTION_ROLE` | Your SageMaker AI execution role ARN. |
| `MODEL_PACKAGE_ARN` | `arn:aws:sagemaker:REGION:594846645681:model-package/dialog-rsn-1-...` | The full model package ARN for your Region, copied from the Marketplace listing after you subscribe. The ARN in the examples is a placeholder. |
| `INSTANCE_TYPE` | `ml.p4d.24xlarge` | Optionally `ml.p5.48xlarge`. |
| `INSTRUCTIONS` | Restaurant assistant prompt | The system instructions sent to the model. |

Model package ARNs are Region-specific. Use the ARN the listing shows for your
chosen Region rather than editing the Region segment of the placeholder.

## SageMaker Endpoint Example

The notebook and script use `boto3` for the SageMaker AI control plane and the
`aws-sdk-sagemaker-runtime-http2` async client for inference to:

1. Create a model from the Marketplace model package with network isolation
   enabled, which Marketplace packages require.
2. Create an endpoint configuration with one `ml.p4d.24xlarge` instance, the
   `al2023-ami-sagemaker-inference-gpu-4-1` inference AMI, and 3,600-second
   model download and container startup health check timeouts.
3. Create the endpoint and poll every 30 seconds until it is `InService`.
   20-40 minutes is normal for a `p4d` pull of a large model. If the endpoint
   fails, the error includes its `FailureReason`.
4. Open a bidirectional HTTP/2 stream with `invoke_endpoint_with_bidirectional_stream`
   and ask one question.
5. Delete the endpoint, endpoint configuration, and model.

The endpoint exchanges JSON events in both directions over a single HTTP/2
connection:

1. `session.update` configures the session with text output and the
   `INSTRUCTIONS` prompt.
2. `conversation.item.create` sends the user's question.
3. `response.create` triggers the response.
4. The client reads events until `response.done`, keeping the text from
   `response.output_text.done`.

The client uses `AWSCRTHTTPClient` as its transport because the default
transport cannot do duplex streaming.

## Running the Example

To run the notebook, open [`Dialog-RSN-1.ipynb`](Dialog-RSN-1.ipynb) and run
the cells in order. The notebook deploys the endpoint, waits for it, and asks
`What time do you close?`. If waiting or invoking fails, it deletes the
resources it created.

To run the script end to end:

```bash
python sample.py
```

The script deploys an endpoint named `dialog-rsn-1-<YYYYMMDD-HHMM>` on the
configured instance type, waits for it, prints the agent's reply, and then
deletes the endpoint, endpoint configuration, and model, even if a step fails.

## Cleanup

The endpoint is billed while it exists. `sample.py` deletes its resources
automatically when it finishes. In the notebook, run the Cleanup cell when you
are done to delete the endpoint, endpoint configuration, and model.
