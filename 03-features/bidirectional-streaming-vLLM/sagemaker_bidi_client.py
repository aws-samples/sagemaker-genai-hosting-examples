#!/usr/bin/env python3
"""
SageMaker Bidirectional Streaming Client — vLLM Realtime API.
Adapted from vLLM's [openai_realtime_client.py](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client) for SageMaker HTTP/2.

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
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AWS_REGION = "us-east-1"
BIDI_ENDPOINT = f"https://runtime.sagemaker.{AWS_REGION}.amazonaws.com:8443"
AUDIO_CHUNK_SIZE = 4096
SAMPLE_RATE = 16_000


def audio_to_pcm16_bytes(audio_path: str) -> bytes:
    """Load audio file and convert to PCM16 @ 16kHz."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    return (audio * 32767).astype(np.int16).tobytes()


class SageMakerRealtimeClient:
    """
    Translates vLLM WebSocket Realtime API protocol
    over SageMaker HTTP/2 bidirectional streaming.
    """

    def __init__(self, endpoint_name, region=AWS_REGION):
        self.endpoint_name = endpoint_name
        self.region = region
        self.client = None
        self.stream = None
        self.recv_task = None
        self.is_active = False

        # Coordination events (Python-level, safe to timeout on)
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

        config = Config(
            endpoint_uri=BIDI_ENDPOINT,
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
        payload_bytes = json.dumps(event_dict).encode("utf-8")
        payload = RequestPayloadPart(bytes_=payload_bytes)
        event = RequestStreamEventPayloadPart(value=payload)
        await self.stream.input_stream.send(event)

    # ──────────────────────────────────────────────
    # Receive loop
    # ──────────────────────────────────────────────

    async def _receive_loop(self):
        """
        Process server responses.
        """
        try:
            output = await self.stream.await_output()
            output_stream = output[1]

            while True:
                result = await output_stream.receive()

                if result is None:
                    logger.info("Output stream ended")
                    break

                if not (result.value and result.value.bytes_):
                    continue

                raw = result.value.bytes_.decode("utf-8")
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
                    logger.error(
                        f"Server error: {data.get('error', data)}"
                    )
                    self.transcription_done.set()
                    break

                else:
                    logger.debug(f"[{msg_type}]")

        except Exception as e:
            logger.error(f"Receive error: {e}")
        finally:
            # Unblock waiters on early exit
            self.session_ready.set()
            self.transcription_done.set()

    # ──────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────

    async def connect(self):
        if not self.client:
            self._initialize_client()

        logger.info(f"Connecting to endpoint: {self.endpoint_name}")
        self.stream = (
            await self.client.invoke_endpoint_with_bidirectional_stream(
                InvokeEndpointWithBidirectionalStreamInput(
                    endpoint_name=self.endpoint_name
                )
            )
        )
        self.is_active = True

        # Start receiver — it handles session.created
        self.recv_task = asyncio.create_task(self._receive_loop())

        # Wait for session (timeout on Python Event, safe)
        await asyncio.wait_for(
            self.session_ready.wait(), timeout=15.0
        )

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

    # ──────────────────────────────────────────────
    # Transcription
    # ──────────────────────────────────────────────

    async def transcribe_audio(self, audio_path: str, model: str):
            # 1. Set model
            await self._send({"type": "session.update", "model": model})

            # 2. Signal ready
            await self._send({"type": "input_audio_buffer.commit"})

            # 3. Send audio chunks at real-time pace
            pcm_audio = audio_to_pcm16_bytes(audio_path)
            total_chunks = (
                (len(pcm_audio) + AUDIO_CHUNK_SIZE - 1) // AUDIO_CHUNK_SIZE
            )

            # Calculate real-time delay per chunk:
            # PCM16 @ 16kHz = 32,000 bytes/sec
            # 4096 bytes/chunk = 0.128 sec of audio per chunk
            bytes_per_sec = SAMPLE_RATE * 2  # 16-bit = 2 bytes per sample
            chunk_duration = AUDIO_CHUNK_SIZE / bytes_per_sec

            logger.info(
                f"Streaming {total_chunks} chunks at real-time pace "
                f"({chunk_duration:.3f}s/chunk, "
                f"{len(pcm_audio) / bytes_per_sec:.1f}s total audio)"
            )

            for idx, i in enumerate(
                range(0, len(pcm_audio), AUDIO_CHUNK_SIZE)
            ):
                chunk = pcm_audio[i : i + AUDIO_CHUNK_SIZE]
                await self._send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                })

                # Log progress periodically
                if idx % 50 == 0:
                    elapsed_audio = idx * chunk_duration
                    logger.info(
                        f"  ▶ Sent chunk {idx}/{total_chunks} "
                        f"({elapsed_audio:.1f}s of audio)"
                    )

                # Pace at real-time — this yields to the event loop,
                # allowing _receive_loop to process incoming deltas
                await asyncio.sleep(chunk_duration)

            # 4. Final commit
            await self._send({
                "type": "input_audio_buffer.commit",
                "final": True,
            })
            logger.info("Audio fully sent. Waiting for final transcription...")

            # 5. Wait for completion
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
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Realtime API — SageMaker Streaming Client")
    print("=" * 60)
    print(f"Endpoint:  {args.ENDPOINT_NAME}")
    print(f"Model:     {args.model}")
    print(f"Audio:     {args.AUDIO_FILE}")
    print("=" * 60)

    async def run():
        client = SageMakerRealtimeClient(
            endpoint_name=args.ENDPOINT_NAME
        )
        try:
            await client.connect()
            await client.transcribe_audio(args.AUDIO_FILE, args.model)
            await client.close()
            logger.info("Done")
            return 0
        except Exception as e:
            logger.error(f"Error: {e}")
            await client.close()
            return 1

    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
