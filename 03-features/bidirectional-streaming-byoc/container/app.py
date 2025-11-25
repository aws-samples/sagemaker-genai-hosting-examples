#!/usr/bin/env python3
import sys
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Store active WebSocket connections
active_connections = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

@app.get("/ping")
@app.post("/ping")
async def ping():
    """Health check endpoint that responds with pong (supports both GET and POST)"""
    return JSONResponse(content={"response": "pong"})

@app.post("/invocations")
async def invocations(request: dict = None):
    """Handle invocation requests"""
    if request is None:
        request = {}
    
    # Echo back the request with a response wrapper
    response = {
        "status": "success",
        "message": "Invocation processed",
        "request_data": request
    }
    return JSONResponse(content=response)

@app.websocket("/invocations-bidirectional-stream")
async def websocket_invoke(websocket: WebSocket):
    """
    WebSocket endpoint with RFC 6455 ping/pong and fragmentation support
    
    Handles:
    - Text messages (JSON) - including fragmented frames
    - Binary messages - including fragmented frames
    - Ping frames (automatically responds with pong)
    - Pong frames (logs receipt)
    - Fragmented frames per RFC 6455 Section 5.4
    """
    await manager.connect(websocket)
    
    # Fragment reassembly buffers per RFC 6455 Section 5.4
    text_fragments = []
    binary_fragments = []
    
    try:
        while True:
            # Use receive() to handle all WebSocket frame types
            message = await websocket.receive()
            print(f"Received message: {message}")
            if message["type"] == "websocket.receive":
                if "text" in message:
                    # Handle text frames (including fragments)
                    text_data = message["text"]
                    more_body = message.get("more_body", False)
                    
                    if more_body:
                        # This is a fragment, accumulate it
                        text_fragments.append(text_data)
                        print(f"Received text fragment: {len(text_data)} chars (more coming)")
                    else:
                        # This is the final frame or a complete message
                        if text_fragments:
                            # Reassemble fragmented message
                            text_fragments.append(text_data)
                            complete_text = "".join(text_fragments)
                            text_fragments.clear()
                            print(f"Reassembled fragmented text message: {len(complete_text)} chars total")
                            await handle_text_message(websocket, complete_text)
                        else:
                            # Complete message in single frame
                            await handle_text_message(websocket, text_data)
                    
                elif "bytes" in message:
                    # Handle binary frames (including fragments)
                    binary_data = message["bytes"]
                    more_body = message.get("more_body", False)
                    
                    if more_body:
                        # This is a fragment, accumulate it
                        binary_fragments.append(binary_data)
                        print(f"Received binary fragment: {len(binary_data)} bytes (more coming)")
                    else:
                        # This is the final frame or a complete message
                        if binary_fragments:
                            # Reassemble fragmented message
                            binary_fragments.append(binary_data)
                            complete_binary = b"".join(binary_fragments)
                            binary_fragments.clear()
                            print(f"Reassembled fragmented binary message: {len(complete_binary)} bytes total")
                            await handle_binary_message(websocket, complete_binary)
                        else:
                            # Complete message in single frame
                            await handle_binary_message(websocket, binary_data)
                    
            elif message["type"] == "websocket.ping":
                # Handle ping frames - RFC 6455 Section 5.5.2
                ping_data = message.get("bytes", b"")
                print(f"Received PING frame with payload: {ping_data}")
                # FastAPI automatically sends pong response
                
            elif message["type"] == "websocket.pong":
                # Handle pong frames
                pong_data = message.get("bytes", b"")
                print(f"Received PONG frame with payload: {pong_data}")
                
            elif message["type"] == "websocket.close":
                # Handle close frames - RFC 6455 Section 5.5.1
                close_code = message.get("code", 1000)
                close_reason = message.get("reason", "")
                print(f"Received CLOSE frame - Code: {close_code}, Reason: '{close_reason}'")
                
                # Send close frame response if not already closing
                try:
                    await websocket.close(code=close_code, reason=close_reason)
                    print(f"Sent CLOSE frame response - Code: {close_code}")
                except Exception as e:
                    print(f"Error sending close frame: {e}")
                break
                
            elif message["type"] == "websocket.disconnect":
                print("Client initiated disconnect")
                break

            else:
                print(f"Received unknown message type: {message['type']}")
                break
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        print("Client connection cleaned up")

async def handle_binary_message(websocket: WebSocket, binary_data: bytes):
    """Handle incoming binary messages (complete or reassembled from fragments)"""
    print(f"Processing complete binary message: {len(binary_data)} bytes")
    
    try:
        # Echo back the binary data
        await websocket.send_bytes(binary_data)
    except Exception as e:
        print(f"Error handling binary message: {e}")

async def handle_text_message(websocket: WebSocket, data: str):
    """Handle incoming text messages"""
    try:        
        # Send response back to the same client
        await manager.send_personal_message(data, websocket)
    except Exception as e:
        print(f"Error handling text message: {e}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("Starting server on port 8080...")
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        print("Usage: python app.py serve")
        sys.exit(1)

if __name__ == "__main__":
    main()
