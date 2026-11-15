"""Talk to PolyAI Dialog-RSN-1 on Amazon SageMaker, from pure Python. No Node required.

Setup (Python 3.12 or later -- the HTTP/2 client requires it):

    pip install boto3 "aws-sdk-sagemaker-runtime-http2[awscrt]"

Subscribe to the listing before you run this (one click, free until an endpoint runs):
    https://aws.amazon.com/marketplace/pp/prodview-l4qyhvfxnkq7e

The below code will not work without a subscription to this listing.
"""

import asyncio
import json
import time

import boto3
from aws_sdk_sagemaker_runtime_http2.client import (
    AsyncSageMakerRuntimeHTTP2Client,
    AsyncSageMakerRuntimeHTTP2Config,
)
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventModelStreamError,
    ResponseStreamEventPayloadPart,
)
from smithy_http.aio.crt import AWSCRTHTTPClient

REGION = "us-west-2"  # change region value

EXECUTION_ROLE_ARN = "arn:aws:iam::ACCOUNT_ID:role/SAGEMAKER_EXECUTION_ROLE"

MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:REGION:594846645681:model-package/dialog-rsn-1-..."
)

INSTANCE_TYPE = "ml.p4d.24xlarge"  # Or ml.p5.48xlarge

INSTRUCTIONS = (
    "You are the assistant for Maison Lumiere, a restaurant. Answer in one short sentence, "
    "from these facts only: we are open until 11 pm every day, and the nearest parking is "
    "the public car park on Bridge Street, two minutes' walk away."
)

sagemaker = boto3.client("sagemaker", region_name=REGION)


def deploy(name: str) -> None:
    """Create the model, endpoint config and endpoint. Marketplace packages must run
    network-isolated. Returns once creation is accepted; the endpoint is not ready yet."""
    sagemaker.create_model(
        ModelName=name,
        ExecutionRoleArn=EXECUTION_ROLE_ARN,
        PrimaryContainer={"ModelPackageName": MODEL_PACKAGE_ARN},
        EnableNetworkIsolation=True,
    )
    sagemaker.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": name,
                "InitialInstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "InferenceAmiVersion": "al2023-ami-sagemaker-inference-gpu-4-1",
                "ModelDataDownloadTimeoutInSeconds": 3600,
                "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
            }
        ],
    )
    sagemaker.create_endpoint(EndpointName=name, EndpointConfigName=name)


def wait_until_ready(name: str) -> None:
    """Poll until InService. 20-40 minutes is normal for a p4d pull of a large model.
    Raises with the real FailureReason instead of a bare timeout."""
    while True:
        status = sagemaker.describe_endpoint(EndpointName=name)
        print(f"  {time.strftime('%H:%M:%S')}  {status['EndpointStatus']}")
        if status["EndpointStatus"] == "InService":
            return
        if status["EndpointStatus"] in ("Failed", "RollingBack", "OutOfService"):
            raise RuntimeError(status.get("FailureReason", status["EndpointStatus"]))
        time.sleep(30)


def delete(name: str) -> None:
    """Remove everything deploy() created. Safe to call twice."""
    for call in (
        lambda: sagemaker.delete_endpoint(EndpointName=name),
        lambda: sagemaker.delete_endpoint_config(EndpointConfigName=name),
        lambda: sagemaker.delete_model(ModelName=name),
    ):
        try:
            call()
        except sagemaker.exceptions.ClientError:
            pass


async def ask(name: str, question: str) -> str:
    """Open one conversation, ask one question, return the reply.

    The endpoint speaks JSON events both ways over a single HTTP/2 connection:
    session.update to configure, conversation.item.create for the question,
    response.create to trigger, then read events until response.done.
    AWSCRTHTTPClient is required -- the default transport cannot do duplex streaming.
    """
    config = await AsyncSageMakerRuntimeHTTP2Config.resolve(
        region=REGION, transport=AWSCRTHTTPClient()
    )
    client = AsyncSageMakerRuntimeHTTP2Client(config=config)
    stream = await client.invoke_endpoint_with_bidirectional_stream(
        input=InvokeEndpointWithBidirectionalStreamInput(endpoint_name=name)
    )

    async def send(event: dict) -> None:
        await stream.input_stream.send(
            RequestStreamEventPayloadPart(
                RequestPayloadPart(
                    bytes_=json.dumps(event).encode(),
                    data_type="UTF8",
                    completion_state="COMPLETE",
                )
            )
        )

    async with stream:
        await send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "dialog-rsn-1",
                    "output_modalities": ["text"],
                    "instructions": INSTRUCTIONS,
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": 16_000}}
                    },
                },
            }
        )
        await send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": question}],
                },
            }
        )
        await send({"type": "response.create"})

        _initial, events = await stream.await_output()
        reply = ""
        async for event in events:
            match event:
                case ResponseStreamEventPayloadPart():
                    message = json.loads(event.value.bytes_)
                    if message["type"] == "error":
                        raise RuntimeError(message["error"])
                    if message["type"] == "response.output_text.done":
                        reply = message.get("text", "")
                    elif message["type"] == "response.done":
                        return reply
                case (
                    ResponseStreamEventModelStreamError()
                    | ResponseStreamEventInternalStreamFailure()
                ):
                    raise RuntimeError(event.value.message)
        return reply


def main() -> None:
    name = f"dialog-rsn-1-{time.strftime('%Y%m%d-%H%M')}"
    print(f"deploying {name} on {INSTANCE_TYPE}")
    deploy(name)
    try:
        wait_until_ready(name)
        print("agent:", asyncio.run(ask(name, "What time do you close?")))
    finally:
        print(f"deleting {name}")
        delete(name)


if __name__ == "__main__":
    main()
