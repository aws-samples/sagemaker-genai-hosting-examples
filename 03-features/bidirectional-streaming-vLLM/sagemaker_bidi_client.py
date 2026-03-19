#!/usr/bin/env python3
"""
SageMaker Bidirectional Streaming Client — vLLM Realtime API.
Adapted from vLLM's [openai_realtime_client.py](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client) for SageMaker HTTP/2.
sends UTF8 text frames via DataType header.
"""
import argparse
import asyncio
import base64
import boto3
import json
import logging
import os
import sys

import librosa
import numpy as np
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart,
    ResponseStreamEventPayloadPart,
    ResponseStreamEventModelStreamError,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventUnknown,
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
AUDIO_CHUNK_SIZE = 4096


def audio_to_pcm16_bytes(audio_path: str) -> bytes:
    """Load audio file and convert to PCM16 @ 16kHz."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    return (audio * 32767).astype(np.int16).tobytes()


class SageMakerRealtimeClient:
    """
    vLLM Realtime API client over SageMaker bidirectional streaming.
    """

    def __init__(self, endpoint_name, region):
        self.endpoint_name = endpoint_name
        self.region = region
        self.client = None
        self.stream = None
        self.recv_task = None
        self.is_active = False
        self.session_ready = asyncio.Event()
        self.transcription_done = asyncio.Event()

    def _initialize_client(self):
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                     "AWS_SESSION_TOKEN"]:
            os.environ.pop(key, None)

        session = boto3.Session()
        creds = session.get_credentials().get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
        if creds.token:
            os.environ["AWS_SESSION_TOKEN"] = creds.token
        logger.info(
            f"Credentials refreshed (key ...{creds.access_key[-4:]})"
        )

        bidi_endpoint = (
            f"https://runtime.sagemaker.{self.region}.amazonaws.com:8443"
        )
        config = Config(
            endpoint_uri=bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=(
                EnvironmentCredentialsResolver()
            ),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={
                "aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")
            },
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)

    async def _send(self, event_dict: dict):
        """Send a JSON message as a UTF8 text frame."""
        payload_bytes = json.dumps(event_dict).encode("utf-8")
        payload = RequestPayloadPart(
            bytes_=payload_bytes,
            data_type="UTF8",  # SageMaker creates Text WebSocket frame
        )
        event = RequestStreamEventPayloadPart(value=payload)
        await self.stream.input_stream.send(event)

    async def _receive_loop(self):
            """
            Receive server responses with proper event type handling.

            Response events can be:
            - ResponseStreamEventPayloadPart → model data
            - ResponseStreamEventModelStreamError → model error
            - ResponseStreamEventInternalStreamFailure → platform error
            - ResponseStreamEventUnknown → unknown event
            """
            try:
                output = await self.stream.await_output()
                output_stream = output[1]

                while True:
                    result = await output_stream.receive()

                    if result is None:
                        logger.info("Output stream ended")
                        break

                    # ── Model stream error ──
                    if isinstance(result, ResponseStreamEventModelStreamError):
                        err = result.value
                        logger.error(
                            f"Model stream error: "
                            f"[{err.error_code}] {err.message}"
                        )
                        self.transcription_done.set()
                        break

                    # ── Internal platform failure ──
                    if isinstance(result, ResponseStreamEventInternalStreamFailure):
                        err = result.value
                        logger.error(
                            f"Internal stream failure: {err.message}"
                        )
                        self.transcription_done.set()
                        break

                    # ── Unknown event type ──
                    if isinstance(result, ResponseStreamEventUnknown):
                        logger.warning(
                            f"Unknown stream event: {result.tag}"
                        )
                        continue

                    # ── Normal payload ──
                    if not isinstance(result, ResponseStreamEventPayloadPart):
                        logger.warning(
                            f"Unexpected event type: {type(result).__name__}"
                        )
                        continue

                    payload = result.value
                    if not (payload and payload.bytes_):
                        continue

                    raw = payload.bytes_.decode("utf-8")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON: {raw[:200]}")
                        continue

                    msg_type = data.get("type", "unknown")

                    if msg_type == "session.created":
                        logger.info(
                            f"✓ Session created: {data.get('id')}"
                        )
                        self.session_ready.set()

                    elif msg_type == "transcription.delta":
                        print(data.get("delta", ""), end="", flush=True)

                    elif msg_type == "transcription.done":
                        print(f"\n\nFinal: {data.get('text', '')}")
                        if data.get("usage"):
                            print(f"Usage: {data['usage']}")
                        self.transcription_done.set()
                        break

                    elif msg_type == "error":
                        logger.error(f"Server error: {data}")
                        self.transcription_done.set()
                        break

                    else:
                        logger.debug(f"[{msg_type}]")

            except Exception as e:
                logger.error(f"Receive error: {e}")
            finally:
                self.session_ready.set()
                self.transcription_done.set()

    async def connect(self):
        if not self.client:
            self._initialize_client()

        logger.info(f"Connecting to endpoint: {self.endpoint_name}")
        self.stream = (
            await self.client.invoke_endpoint_with_bidirectional_stream(
                InvokeEndpointWithBidirectionalStreamInput(
                    endpoint_name=self.endpoint_name,
                )
            )
        )
        self.is_active = True
        self.recv_task = asyncio.create_task(self._receive_loop())
        await asyncio.wait_for(self.session_ready.wait(), timeout=15.0)

    async def close(self):
        if not self.is_active:
            return
        self.is_active = False
        try:
            await self.stream.input_stream.close()
        except Exception:
            pass
        if self.recv_task and not self.recv_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self.recv_task), timeout=5.0
                )
            except asyncio.TimeoutError:
                pass
        logger.info("Connection closed")

    async def transcribe_audio(self, audio_path: str, model: str):
        """Transcribe audio file using vLLM Realtime API protocol."""
        await self._send({"type": "session.update", "model": model})
        await self._send({"type": "input_audio_buffer.commit"})

        logger.info(f"Loading audio: {audio_path}")
        pcm_audio = audio_to_pcm16_bytes(audio_path)
        total_chunks = (
            (len(pcm_audio) + AUDIO_CHUNK_SIZE - 1) // AUDIO_CHUNK_SIZE
        )

        # Real-time pacing for true bidirectional streaming
        bytes_per_sec = SAMPLE_RATE * 2  # 16-bit PCM
        chunk_duration = AUDIO_CHUNK_SIZE / bytes_per_sec

        logger.info(
            f"Streaming {total_chunks} chunks at real-time pace "
            f"({len(pcm_audio) / bytes_per_sec:.1f}s of audio)"
        )

        for idx, i in enumerate(
            range(0, len(pcm_audio), AUDIO_CHUNK_SIZE)
        ):
            chunk = pcm_audio[i : i + AUDIO_CHUNK_SIZE]
            await self._send({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            })

            if idx % 50 == 0 and idx > 0:
                elapsed = idx * chunk_duration
                logger.info(
                    f"  ▶ Sent {idx}/{total_chunks} "
                    f"({elapsed:.1f}s of audio)"
                )

            # Yield to event loop — allows receive loop
            # to process deltas while sending
            await asyncio.sleep(chunk_duration)

        await self._send({
            "type": "input_audio_buffer.commit",
            "final": True,
        })

        await asyncio.wait_for(
            self.transcription_done.wait(), timeout=60.0
        )


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker Realtime Transcription Client"
    )
    parser.add_argument("ENDPOINT_NAME", help="SageMaker endpoint name")
    parser.add_argument("AUDIO_FILE", help="Path to audio file")
    parser.add_argument(
        "--model",
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
    )
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Realtime API — SageMaker AI")
    print("=" * 60)
    print(f"Endpoint:  {args.ENDPOINT_NAME}")
    print(f"Model:     {args.model}")
    print(f"Audio:     {args.AUDIO_FILE}")
    print("=" * 60)

    async def run():
        client = SageMakerRealtimeClient(
            endpoint_name=args.ENDPOINT_NAME,
            region=args.region,
        )
        try:
            await client.connect()
            await client.transcribe_audio(args.AUDIO_FILE, args.model)
            await client.close()
            logger.info("Done")
            return 0
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await client.close()
            return 1

    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
