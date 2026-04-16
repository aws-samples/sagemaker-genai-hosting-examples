#!/usr/bin/env python3
"""
SageMaker Bidirectional Streaming Client — vLLM-Omni TTS.

Sends text to a Voxtral TTS endpoint via SageMaker's HTTP/2 bidirectional
streaming transport and saves the generated audio to a file.

The bridge relays to vLLM-Omni's /v1/audio/speech/stream WebSocket,
which splits text at sentence boundaries and streams audio per sentence.

Protocol (relayed through the SageMaker bridge):
  Client -> Server:
    {"type": "session.config", "model": "...", "voice": "...", ...}
    {"type": "input.text", "text": "..."}
    {"type": "input.done"}

  Server -> Client:
    {"type": "audio.start", "sentence_index": N, ...}
    <binary frame: audio bytes>
    {"type": "audio.done", "sentence_index": N}
    {"type": "session.done", "total_sentences": N}

    python sagemaker_tts_bidi_client.py ENDPOINT_NAME "Hello world!" \\
        --voice casual_male --output output.wav
"""
import argparse
import asyncio
import boto3
import io
import json
import logging
import os
import sys

import numpy as np
import soundfile as sf
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


class SageMakerTTSClient:
    """
    vLLM-Omni TTS client over SageMaker bidirectional streaming.

    Uses the streaming TTS protocol where audio is delivered per-sentence
    as binary WebSocket frames.
    """

    def __init__(self, endpoint_name, region):
        self.endpoint_name = endpoint_name
        self.region = region
        self.client = None
        self.stream = None
        self.recv_task = None
        self.is_active = False

        # Receive state
        self.session_done = asyncio.Event()
        self.audio_chunks: list[bytes] = []
        self.sentence_count = 0
        self.error = None

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
            data_type="UTF8",
        )
        event = RequestStreamEventPayloadPart(value=payload)
        await self.stream.input_stream.send(event)

    async def _receive_loop(self):
        """
        Receive server responses.

        Handles both text frames (JSON control messages) and binary frames
        (raw audio data) from the vLLM streaming TTS endpoint.
        """
        try:
            output = await self.stream.await_output()
            output_stream = output[1]

            while True:
                result = await output_stream.receive()

                if result is None:
                    logger.info("Output stream ended")
                    break

                if isinstance(result, ResponseStreamEventModelStreamError):
                    err = result.value
                    self.error = f"[{err.error_code}] {err.message}"
                    logger.error(f"Model stream error: {self.error}")
                    break

                if isinstance(result, ResponseStreamEventInternalStreamFailure):
                    err = result.value
                    self.error = f"Internal failure: {err.message}"
                    logger.error(self.error)
                    break

                if isinstance(result, ResponseStreamEventUnknown):
                    logger.warning(f"Unknown stream event: {result.tag}")
                    continue

                if not isinstance(result, ResponseStreamEventPayloadPart):
                    continue

                payload = result.value
                if not (payload and payload.bytes_):
                    continue

                raw_bytes = payload.bytes_

                # Binary frames (audio data) won't have DataType=UTF8
                # Text frames (JSON control messages) have DataType=UTF8
                data_type = getattr(payload, "data_type", None)

                if data_type != "UTF8":
                    # Binary frame — raw audio bytes
                    self.audio_chunks.append(raw_bytes)
                    continue

                # Text frame — JSON control message
                raw = raw_bytes.decode("utf-8")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # Not JSON — treat as audio binary
                    self.audio_chunks.append(raw_bytes)
                    continue

                msg_type = data.get("type", "unknown")

                if msg_type == "audio.start":
                    idx = data.get("sentence_index", 0)
                    text = data.get("sentence_text", "")
                    fmt = data.get("format", "?")
                    logger.info(
                        f"  Sentence {idx} ({fmt}): "
                        f"{text[:60]}{'...' if len(text) > 60 else ''}"
                    )

                elif msg_type == "audio.done":
                    idx = data.get("sentence_index", 0)
                    total = data.get("total_bytes", 0)
                    logger.info(
                        f"  Sentence {idx} done: {total} bytes"
                    )

                elif msg_type == "session.done":
                    self.sentence_count = data.get(
                        "total_sentences", 0
                    )
                    logger.info(
                        f"Session done: {self.sentence_count} sentences"
                    )
                    break

                elif msg_type == "error":
                    self.error = data.get("message", str(data))
                    logger.error(f"Server error: {self.error}")
                    break

                else:
                    logger.debug(f"[{msg_type}] {data}")

        except Exception as e:
            if not self.error:
                self.error = str(e)
            logger.error(f"Receive error: {e}")
        finally:
            self.session_done.set()

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

    async def synthesize(
        self,
        text: str,
        model: str = "mistralai/Voxtral-4B-TTS-2603",
        voice: str = "casual_male",
        response_format: str = "wav",
    ) -> list[bytes]:
        """
        Synthesize speech from text using the streaming protocol.

        Sends session.config, then the full text as input.text,
        then input.done. Collects per-sentence audio binary frames.

        Returns list of audio byte chunks (one per sentence).
        Each chunk is a complete audio file in the requested format.
        """
        self.session_done.clear()
        self.audio_chunks.clear()
        self.error = None
        self.sentence_count = 0

        # 1. Send session config
        await self._send({
            "type": "session.config",
            "model": model,
            "voice": voice,
            "response_format": response_format,
        })

        # 2. Send text
        await self._send({
            "type": "input.text",
            "text": text,
        })

        # 3. Signal end of input
        await self._send({
            "type": "input.done",
        })

        logger.info("Waiting for audio...")
        await asyncio.wait_for(self.session_done.wait(), timeout=120.0)

        if self.error:
            raise RuntimeError(f"TTS failed: {self.error}")

        if not self.audio_chunks:
            raise RuntimeError("No audio data received")

        return self.audio_chunks


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker TTS Client (Bidirectional Streaming)"
    )
    parser.add_argument("ENDPOINT_NAME", help="SageMaker endpoint name")
    parser.add_argument("TEXT", help="Text to synthesize")
    parser.add_argument(
        "--model",
        default="mistralai/Voxtral-4B-TTS-2603",
    )
    parser.add_argument(
        "--voice",
        default="casual_male",
        help="Voice preset (e.g., casual_male, casual_female)",
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio output format",
    )
    parser.add_argument(
        "--output",
        default="tts_output.wav",
        help="Output audio file path",
    )
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM-Omni TTS — SageMaker AI")
    print("=" * 60)
    print(f"Endpoint:  {args.ENDPOINT_NAME}")
    print(f"Model:     {args.model}")
    print(f"Voice:     {args.voice}")
    print(f"Format:    {args.format}")
    print(f"Text:      {args.TEXT[:80]}{'...' if len(args.TEXT) > 80 else ''}")
    print(f"Output:    {args.output}")
    print("=" * 60)

    async def run():
        client = SageMakerTTSClient(
            endpoint_name=args.ENDPOINT_NAME,
            region=args.region,
        )
        try:
            await client.connect()
            audio_chunks = await client.synthesize(
                text=args.TEXT,
                model=args.model,
                voice=args.voice,
                response_format=args.format,
            )

            # Merge per-sentence audio chunks into one file.
            # Each chunk is a complete audio file (e.g., WAV with header).
            # Decode each, concatenate samples, write combined output.
            all_samples = []
            sample_rate = None
            for chunk in audio_chunks:
                try:
                    samples, sr = sf.read(
                        io.BytesIO(chunk), dtype="float32"
                    )
                    all_samples.append(samples)
                    sample_rate = sr
                except Exception as e:
                    logger.warning(f"Skipping chunk: {e}")

            if not all_samples or sample_rate is None:
                print("ERROR: Could not decode any audio chunks")
                await client.close()
                return 1

            combined = np.concatenate(all_samples)
            sf.write(args.output, combined, sample_rate)
            duration = len(combined) / sample_rate
            total_bytes = sum(len(c) for c in audio_chunks)
            print(
                f"\nAudio saved to: {args.output} "
                f"({total_bytes} bytes across "
                f"{client.sentence_count} sentences)"
            )
            print(f"Duration: {duration:.1f}s at {sample_rate} Hz")

            await client.close()
            return 0
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await client.close()
            return 1

    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
