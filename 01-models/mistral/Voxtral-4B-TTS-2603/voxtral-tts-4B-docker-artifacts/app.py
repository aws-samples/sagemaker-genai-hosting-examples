#!/usr/bin/env python3
"""
SageMaker <-> vLLM-Omni TTS Bridge.

Routes:
  /ping                              -> /health               (health check)
  /invocations                       -> /v1/audio/speech      (HTTP POST)
  /invocations-bidirectional-stream  -> /v1/audio/speech/stream (WebSocket relay)

The bidirectional-stream route is a direct WebSocket-to-WebSocket relay
between SageMaker and vLLM-Omni's streaming TTS endpoint. Text frames
and binary audio frames are forwarded in both directions.

vLLM-Omni streaming TTS protocol (/v1/audio/speech/stream):
  Client -> Server:
    {"type": "session.config", "model": "...", "voice": "...", ...}
    {"type": "input.text", "text": "..."}
    {"type": "input.done"}

  Server -> Client:
    {"type": "audio.start", "sentence_index": N, "sentence_text": "...", "format": "wav"}
    <binary frame: audio bytes>
    {"type": "audio.done", "sentence_index": N, "total_bytes": M}
    {"type": "session.done", "total_sentences": N}
    {"type": "error", "message": "..."}
"""

import json
import asyncio
import logging

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, Response
import uvicorn
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VLLM_BASE_URL = "http://localhost:8081"
VLLM_HEALTH_URL = f"{VLLM_BASE_URL}/health"
VLLM_TTS_URL = f"{VLLM_BASE_URL}/v1/audio/speech"
VLLM_TTS_STREAM_WS_URL = "ws://localhost:8081/v1/audio/speech/stream"

# Content-Type mapping for audio formats
AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "opus": "audio/opus",
}


# --------------------------------------------------
# SageMaker health check
# --------------------------------------------------

@app.get("/ping")
@app.post("/ping")
async def ping():
    """Proxy health check to vLLM."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(VLLM_HEALTH_URL, timeout=5.0)
            return JSONResponse(
                content={"status": "healthy"},
                status_code=resp.status_code,
            )
    except Exception:
        return JSONResponse(
            content={"status": "unhealthy"}, status_code=503
        )


# --------------------------------------------------
# Standard SageMaker invocation (HTTP POST)
# --------------------------------------------------

@app.post("/invocations")
async def invocations(request: Request):
    """
    Standard SageMaker invocation endpoint.

    Expects JSON body compatible with OpenAI TTS API:
    {
        "input": "Text to synthesize",
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "voice": "casual_male",
        "response_format": "wav"
    }

    Returns audio bytes with appropriate Content-Type.
    """
    try:
        body = await request.json()
        response_format = body.get("response_format", "wav")
        content_type = AUDIO_CONTENT_TYPES.get(response_format, "audio/wav")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                VLLM_TTS_URL,
                json=body,
                timeout=120.0,
            )
            resp.raise_for_status()

            return Response(
                content=resp.content,
                media_type=content_type,
                status_code=200,
            )

    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM TTS error: {e.response.status_code} {e.response.text}")
        return JSONResponse(
            content={"error": f"TTS generation failed: {e.response.text}"},
            status_code=e.response.status_code,
        )
    except Exception as e:
        logger.error(f"Invocation error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


# --------------------------------------------------
# Bidirectional streaming WebSocket bridge
# --------------------------------------------------

@app.websocket("/invocations-bidirectional-stream")
async def websocket_bridge(sm_ws: WebSocket):
    """
    WebSocket-to-WebSocket relay: SageMaker <-> vLLM /v1/audio/speech/stream.
    Frame handling:
      SM -> vLLM: text frames pass through; binary decoded to text (fallback)
      vLLM -> SM: text frames pass through, binary frames pass through
    """
    await sm_ws.accept()
    logger.info("SageMaker WebSocket connected")

    try:
        async with websockets.connect(VLLM_TTS_STREAM_WS_URL) as vllm_ws:
            logger.info("Connected to vLLM /v1/audio/speech/stream")

            async def sm_to_vllm():
                """Forward: SageMaker -> vLLM"""
                try:
                    while True:
                        message = await sm_ws.receive()

                        if message["type"] == "websocket.receive":
                            if "text" in message and message["text"]:
                                # UTF8 client -> native text frame
                                await vllm_ws.send(message["text"])
                            elif "bytes" in message and message["bytes"]:
                                # Non-UTF8 fallback -> decode to text
                                text = message["bytes"].decode("utf-8")
                                await vllm_ws.send(text)

                        elif message["type"] == "websocket.disconnect":
                            logger.info("SageMaker disconnected")
                            break

                except WebSocketDisconnect:
                    logger.info("SageMaker WebSocket disconnected")
                except Exception as e:
                    logger.error(f"SM -> vLLM error: {e}")

            async def vllm_to_sm():
                """Forward: vLLM -> SageMaker"""
                try:
                    async for msg in vllm_ws:
                        if isinstance(msg, str):
                            # Text -> text (JSON control messages)
                            await sm_ws.send_text(msg)
                        elif isinstance(msg, bytes):
                            # Binary -> binary (audio data)
                            await sm_ws.send_bytes(msg)

                except websockets.ConnectionClosed as e:
                    logger.info(f"vLLM WebSocket closed: {e.code}")
                except Exception as e:
                    logger.error(f"vLLM -> SM error: {e}")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(sm_to_vllm()),
                    asyncio.create_task(vllm_to_sm()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except websockets.InvalidStatusCode as e:
        logger.error(f"vLLM rejected WebSocket: {e.status_code}")
        await sm_ws.send_text(json.dumps({
            "type": "error",
            "message": f"vLLM connection failed: {e.status_code}",
        }))
    except Exception as e:
        logger.error(f"Bridge error: {e}", exc_info=True)
    finally:
        try:
            await sm_ws.close()
        except Exception:
            pass
        logger.info("Bridge connection cleaned up")


def main():
    logger.info("Starting SageMaker <-> vLLM TTS bridge on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
