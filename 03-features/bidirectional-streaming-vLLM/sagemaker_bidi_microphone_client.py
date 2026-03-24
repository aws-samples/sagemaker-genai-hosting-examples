#!/usr/bin/env python3
"""
Gradio demo for real-time speech transcription using vLLM Realtime API
via SageMaker Bidirectional Streaming.

sends UTF8 text frames via DataType header.

    python microphone_client.py \
        --endpoint-name YOUR_ENDPOINT_NAME \
        --region us-east-1

Requirements: gradio, numpy, boto3, aws-sdk-sagemaker-runtime-http2
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import queue
import threading

import boto3
import gradio as gr
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

# Global state
audio_queue: queue.Queue = queue.Queue()
transcription_text = ""
is_running = False
endpoint_name = ""
aws_region = ""
model_name = ""


# ──────────────────────────────────────────────
# SageMaker helpers
# ──────────────────────────────────────────────

def refresh_credentials():
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_SESSION_TOKEN"]:
        os.environ.pop(key, None)
    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    logger.info(f"Credentials refreshed (key ...{creds.access_key[-4:]})")


def create_client():
    refresh_credentials()
    config = Config(
        endpoint_uri=(
            f"https://runtime.sagemaker.{aws_region}.amazonaws.com:8443"
        ),
        region=aws_region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={
            "aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")
        },
    )
    return SageMakerRuntimeHTTP2Client(config=config)


async def send_event(stream, event_dict):
    """Send a JSON event as a UTF8 text frame."""
    payload_bytes = json.dumps(event_dict).encode("utf-8")
    payload = RequestPayloadPart(
        bytes_=payload_bytes,
        data_type="UTF8",  # SageMaker creates Text WebSocket frame
    )
    event = RequestStreamEventPayloadPart(value=payload)
    await stream.input_stream.send(event)


# ──────────────────────────────────────────────
# Streaming handler
# ──────────────────────────────────────────────

async def streaming_handler():
    """
    Connect to SageMaker bidi stream and bridge
    microphone audio to vLLM Realtime API.
    """
    global transcription_text, is_running

    client = create_client()

    logger.info(f"Connecting to endpoint: {endpoint_name}")
    stream = await client.invoke_endpoint_with_bidirectional_stream(
        InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=endpoint_name,
        )
    )

    output = await stream.await_output()
    output_stream = output[1]

    # Coordination events (Python-level, safe to timeout on)
    session_ready = asyncio.Event()
    transcription_complete = asyncio.Event()

    # ── Receive task ──
    async def receive_loop():
            global transcription_text
            try:
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
                        transcription_complete.set()
                        break

                    # ── Internal platform failure ──
                    if isinstance(result, ResponseStreamEventInternalStreamFailure):
                        err = result.value
                        logger.error(
                            f"Internal stream failure: {err.message}"
                        )
                        transcription_complete.set()
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
                            f"Unexpected event type: "
                            f"{type(result).__name__}"
                        )
                        continue

                    payload = result.value
                    if not (payload and payload.bytes_):
                        continue

                    raw = payload.bytes_.decode("utf-8")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "unknown")

                    if msg_type == "session.created":
                        logger.info(
                            f"✓ Session created: {data.get('id')}"
                        )
                        session_ready.set()

                    elif msg_type == "transcription.delta":
                        transcription_text += data.get("delta", "")

                    elif msg_type == "transcription.done":
                        logger.info("✓ Transcription complete")
                        if data.get("usage"):
                            logger.info(f"Usage: {data['usage']}")
                        transcription_complete.set()
                        break

                    elif msg_type == "error":
                        logger.error(f"Server error: {data}")
                        transcription_complete.set()
                        break

                    else:
                        logger.debug(f"[{msg_type}]")

            except Exception as e:
                logger.error(f"Receive error: {e}")
            finally:
                session_ready.set()
                transcription_complete.set()

    # Start receiver FIRST — it handles session.created
    recv_task = asyncio.create_task(receive_loop())

    # Wait for session (timeout on Python Event, safe)
    await asyncio.wait_for(session_ready.wait(), timeout=15.0)

    # Configure session
    await send_event(stream, {
        "type": "session.update",
        "model": model_name,
    })
    await send_event(stream, {
        "type": "input_audio_buffer.commit",
    })
    logger.info("✓ Ready — streaming microphone audio")

    # ── Send audio while recording ──
    while is_running:
        try:
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, lambda: audio_queue.get(timeout=0.1)
            )
            await send_event(stream, {
                "type": "input_audio_buffer.append",
                "audio": chunk,
            })
        except queue.Empty:
            continue

    # User clicked Stop — flush remaining audio
    logger.info("Sending final audio commit")
    await send_event(stream, {
        "type": "input_audio_buffer.commit",
        "final": True,
    })

    # Wait for transcription to finish
    try:
        await asyncio.wait_for(
            transcription_complete.wait(), timeout=30.0
        )
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for final transcription")

    # Cleanup — close input, let recv_task finish naturally
    try:
        await stream.input_stream.close()
    except Exception:
        pass

    if not recv_task.done():
        try:
            await asyncio.wait_for(
                asyncio.shield(recv_task), timeout=5.0
            )
        except asyncio.TimeoutError:
            pass

    logger.info("Stream closed")


# ──────────────────────────────────────────────
# Gradio callbacks
# ──────────────────────────────────────────────

def start_streaming_thread():
    global is_running
    is_running = True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(streaming_handler())
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)


def start_recording():
    global transcription_text
    transcription_text = ""

    # Drain leftover audio from previous session
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    thread = threading.Thread(target=start_streaming_thread, daemon=True)
    thread.start()
    return (
        gr.update(interactive=False),
        gr.update(interactive=True),
        "",
    )


def stop_recording():
    global is_running
    is_running = False
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        transcription_text,
    )


def process_audio(audio):
    """Process incoming microphone audio from Gradio."""
    global transcription_text

    if audio is None or not is_running:
        return transcription_text

    sample_rate, audio_data = audio

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to float32
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)

    # Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE:
        num_samples = int(len(audio_float) * SAMPLE_RATE / sample_rate)
        audio_float = np.interp(
            np.linspace(0, len(audio_float) - 1, num_samples),
            np.arange(len(audio_float)),
            audio_float,
        )

    # Convert to PCM16 and base64 encode
    pcm16 = (audio_float * 32767).astype(np.int16)
    b64_chunk = base64.b64encode(pcm16.tobytes()).decode("utf-8")
    audio_queue.put(b64_chunk)

    return transcription_text


# ──────────────────────────────────────────────
# Gradio interface
# ──────────────────────────────────────────────

with gr.Blocks(title="Real-time Speech Transcription (SageMaker)") as demo:
    gr.Markdown("# 🎙️ Real-time Speech Transcription")
    gr.Markdown(
        "Powered by **Voxtral Mini 4B** via "
        "**vLLM Realtime API** on **SageMaker AI**"
    )
    gr.Markdown(
        "Click **Start**, speak into your microphone, "
        "then click **Stop**."
    )

    with gr.Row():
        start_btn = gr.Button("▶ Start", variant="primary")
        stop_btn = gr.Button("⏹ Stop", variant="stop", interactive=False)

    audio_input = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Microphone",
    )
    transcription_output = gr.Textbox(
        label="Transcription",
        lines=8,
        placeholder="Transcription will appear here...",
    )

    start_btn.click(
        start_recording,
        outputs=[start_btn, stop_btn, transcription_output],
    )
    stop_btn.click(
        stop_recording,
        outputs=[start_btn, stop_btn, transcription_output],
    )
    audio_input.stream(
        process_audio,
        inputs=[audio_input],
        outputs=[transcription_output],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime Transcription — Gradio + SageMaker"
    )
    parser.add_argument(
        "--endpoint-name", type=str, required=True,
        help="SageMaker endpoint name",
    )
    parser.add_argument(
        "--model", type=str,
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
    )
    parser.add_argument(
        "--region", type=str, default="us-east-1",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create public Gradio link",
    )
    args = parser.parse_args()

    endpoint_name = args.endpoint_name
    aws_region = args.region
    model_name = args.model
    demo.launch(server_port=6006, share=True)
