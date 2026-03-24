#!/usr/bin/env python3
"""
SageMaker ↔ vLLM Realtime WebSocket Bridge.

Routes /invocations-bidirectional-stream → /v1/realtime.

With UTF8 DataType on the client, SageMaker sends native text frames
and the bridge passes them through directly. Falls back to bytes→text
decoding for non-UTF8 clients.

vLLM sends text frames back → bridge forwards as text → SageMaker
sets DataType=UTF8 on the response PayloadPart.
"""

import json
import asyncio
import logging

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VLLM_WS_URL = "ws://localhost:8081/v1/realtime"
VLLM_HEALTH_URL = "http://localhost:8081/health"


# ──────────────────────────────────────────────
# SageMaker health check
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# WebSocket bridge
# ──────────────────────────────────────────────

@app.websocket("/invocations-bidirectional-stream")
async def websocket_bridge(sm_ws: WebSocket):
    """
    Accept SageMaker's WebSocket on /invocations-bidirectional-stream,
    connect to vLLM's /v1/realtime, and bridge messages both directions.

    Frame handling:
      SM → vLLM: text frames pass through; binary decoded to text (fallback)
      vLLM → SM: text frames pass through (SageMaker marks as UTF8)
    """
    await sm_ws.accept()
    logger.info("SageMaker WebSocket connected")

    try:
        async with websockets.connect(VLLM_WS_URL) as vllm_ws:
            logger.info("Connected to vLLM /v1/realtime")

            async def sm_to_vllm():
                """Forward: SageMaker → vLLM"""
                try:
                    while True:
                        message = await sm_ws.receive()

                        if message["type"] == "websocket.receive":
                            if "text" in message and message["text"]:
                                # UTF8 client → native text frame
                                await vllm_ws.send(message["text"])
                            elif "bytes" in message and message["bytes"]:
                                # Non-UTF8 fallback → decode to text
                                text = message["bytes"].decode("utf-8")
                                await vllm_ws.send(text)

                        elif message["type"] == "websocket.disconnect":
                            logger.info("SageMaker disconnected")
                            break

                except WebSocketDisconnect:
                    logger.info("SageMaker WebSocket disconnected")
                except Exception as e:
                    logger.error(f"SM → vLLM error: {e}")

            async def vllm_to_sm():
                """Forward: vLLM → SageMaker"""
                try:
                    async for msg in vllm_ws:
                        if isinstance(msg, str):
                            # Text → text (SageMaker sets DataType=UTF8)
                            await sm_ws.send_text(msg)
                        elif isinstance(msg, bytes):
                            await sm_ws.send_bytes(msg)

                except websockets.ConnectionClosed as e:
                    logger.info(f"vLLM WebSocket closed: {e.code}")
                except Exception as e:
                    logger.error(f"vLLM → SM error: {e}")

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
            "error": f"vLLM connection failed: {e.status_code}",
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
    logger.info("Starting SageMaker ↔ vLLM bridge on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
