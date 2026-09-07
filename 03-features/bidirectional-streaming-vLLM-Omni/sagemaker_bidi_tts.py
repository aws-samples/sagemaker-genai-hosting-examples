"""Shared SageMaker bidirectional streaming helpers for vLLM-Omni TTS."""

import asyncio
import base64
import json
import logging
import os

import boto3
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import (
    Config,
    HTTPAuthSchemeResolver,
)
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventModelStreamError,
    ResponseStreamEventPayloadPart,
    ResponseStreamEventUnknown,
)
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
from smithy_aws_core.identity import EnvironmentCredentialsResolver

SPEECH_STREAM_PATH = "v1/audio/speech/stream"
DEFAULT_SAMPLE_RATE = 24_000
logger = logging.getLogger(__name__)


def make_bidi_client(region):
    """Create the SageMaker Runtime HTTP/2 client."""
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials found.")

    frozen = credentials.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token

    config = Config(
        endpoint_uri=(f"https://runtime.sagemaker.{region}.amazonaws.com:8443"),
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
    )
    return SageMakerRuntimeHTTP2Client(config=config)


async def send_json(stream, message):
    """Send one complete UTF-8 JSON message."""
    payload = RequestPayloadPart(
        bytes_=json.dumps(message).encode("utf-8"),
        data_type="UTF8",
        completion_state="COMPLETE",
    )
    await stream.input_stream.send(RequestStreamEventPayloadPart(value=payload))


async def stream_pcm_events(
    endpoint_name,
    text,
    region="us-east-1",
    voice="Vivian",
    language="English",
    deadline_seconds=180,
):
    """Yield lifecycle events and PCM chunks from one TTS request."""
    client = make_bidi_client(region)
    request = InvokeEndpointWithBidirectionalStreamInput(
        endpoint_name=endpoint_name,
        model_invocation_path=SPEECH_STREAM_PATH,
    )
    stream = await client.invoke_endpoint_with_bidirectional_stream(input=request)
    loop = asyncio.get_running_loop()
    sample_rate = DEFAULT_SAMPLE_RATE

    try:
        _, receiver = await stream.await_output()
        await send_json(
            stream,
            {
                "type": "session.config",
                "voice": voice,
                "language": language,
                "response_format": "pcm",
                "stream_audio": True,
            },
        )
        await send_json(stream, {"type": "input.text", "text": text})
        await send_json(stream, {"type": "input.done"})

        started_at = loop.time()
        while loop.time() - started_at < deadline_seconds:
            remaining = deadline_seconds - (loop.time() - started_at)
            try:
                event = await asyncio.wait_for(
                    receiver.receive(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError("Timed out waiting for streamed audio.") from error

            if event is None:
                raise RuntimeError("The response stream ended unexpectedly.")
            if isinstance(event, ResponseStreamEventModelStreamError):
                raise RuntimeError(  # noqa: TRY004
                    f"Model stream error: {event.value.error_code}: "
                    f"{event.value.message}"
                )
            if isinstance(event, ResponseStreamEventInternalStreamFailure):
                raise RuntimeError(  # noqa: TRY004
                    f"Internal stream failure: {event.value.message}"
                )
            if isinstance(event, ResponseStreamEventUnknown):
                raise RuntimeError(  # noqa: TRY004
                    f"Unknown response event: {event.tag}"
                )
            if not isinstance(event, ResponseStreamEventPayloadPart):
                raise RuntimeError(  # noqa: TRY004
                    f"Unexpected response event: {type(event).__name__}"
                )

            payload = event.value
            raw = payload.bytes_ or b""
            if (payload.data_type or "").upper() != "UTF8":
                if raw:
                    yield {
                        "type": "audio.chunk",
                        "audio": raw,
                        "sample_rate": sample_rate,
                    }
                continue

            message = json.loads(raw.decode("utf-8"))
            event_type = message.get("type")
            if event_type == "audio.start":
                sample_rate = int(message.get("sample_rate") or DEFAULT_SAMPLE_RATE)
                yield message
            elif event_type == "audio.chunk":
                audio = base64.b64decode(message.get("audio_b64") or "")
                if audio:
                    yield {
                        **message,
                        "audio": audio,
                        "sample_rate": int(message.get("sample_rate") or sample_rate),
                    }
            elif event_type == "audio.done":
                yield message
            elif event_type == "session.done":
                yield message
                return
            elif event_type == "error":
                raise RuntimeError(
                    message.get("message") or "The model returned an error."
                )
    finally:
        try:
            await send_json(stream, {"type": "session.close"})
        except Exception:
            logger.debug("Failed to send session.close.", exc_info=True)
        try:
            await stream.input_stream.close()
        except Exception:
            logger.debug("Failed to close the input stream.", exc_info=True)


async def collect_pcm(endpoint_name, text, **kwargs):
    """Collect one streamed response for repeatable validation."""
    chunks = []
    sample_rate = DEFAULT_SAMPLE_RATE
    saw_start = False
    saw_done = False

    async for event in stream_pcm_events(
        endpoint_name,
        text,
        **kwargs,
    ):
        event_type = event.get("type")
        if event_type == "audio.start":
            saw_start = True
            sample_rate = int(event.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        elif event_type == "audio.chunk":
            chunks.append(event["audio"])
            sample_rate = int(event.get("sample_rate") or sample_rate)
        elif event_type == "audio.done":
            saw_done = True

    return {
        "audio": b"".join(chunks),
        "sample_rate": sample_rate,
        "saw_start": saw_start,
        "saw_done": saw_done,
    }
