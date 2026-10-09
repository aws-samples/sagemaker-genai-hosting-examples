# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Minimal SageMaker AI hosting server for a sample's model_fn / transform_fn adapter.

Mirrors the parts of the SageMaker PyTorch inference toolkit the decision-model samples use:

1. Install code/requirements.txt, if present (pip honours --no-index lines, so the
   network-isolated notebooks keep working).
2. Import SAGEMAKER_PROGRAM (default inference.py) from SAGEMAKER_SUBMIT_DIRECTORY and load
   the model once with model_fn("/opt/ml/model").
3. Serve GET /ping and POST /invocations on port 8080 with one process. Requests go to
   transform_fn; GenericInferenceToolkitError keeps its HTTP status, so a 422 from the
   adapter still reaches the caller as a ModelError with OriginalStatusCode 422.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from sagemaker_inference.errors import GenericInferenceToolkitError
from starlette.concurrency import run_in_threadpool

MODEL_DIR = "/opt/ml/model"
CODE_DIR = Path(os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", f"{MODEL_DIR}/code"))
PROGRAM = os.environ.get("SAGEMAKER_PROGRAM", "inference.py")


def install_requirements() -> None:
    requirements = CODE_DIR / "requirements.txt"
    if requirements.is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)],
                       check=True)


def load_adapter():
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("inference", CODE_DIR / PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install_requirements()
adapter = load_adapter()
model = adapter.model_fn(MODEL_DIR)
app = FastAPI()


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")
    accept = request.headers.get("accept", "application/json")
    try:
        # transform_fn is synchronous; run it in the thread pool, as FastAPI does for sync routes.
        payload, out_type = await run_in_threadpool(adapter.transform_fn, model, body, content_type, accept)
    except GenericInferenceToolkitError as error:
        return Response(content=error.message, status_code=error.status_code, media_type="text/plain")
    return Response(content=payload, media_type=out_type)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
