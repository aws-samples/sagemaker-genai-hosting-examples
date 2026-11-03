"""
SageMaker serving shim for SGLang Diffusion serving MiniMax-H3.

SageMaker requires a container to answer GET /ping and POST /invocations on
port 8080. SGLang Diffusion exposes an asynchronous OpenAI-compatible video
API on its own port. This shim bridges the two:

    POST /invocations
      -> submit job to SGLang  POST /v1/videos
      -> poll                  GET  /v1/videos/{id}
      -> download MP4          GET  /v1/videos/{id}/content
      -> upload MP4 to S3, return a JSON pointer

Reference media (first/last frames, reference images, videos, audio) arrive as
s3:// URIs in the request. They are downloaded to a local scratch directory and
rewritten as file:// URIs, because SGLang resolves conditions[].uri inside the
server's own filesystem.

Environment:
  SGLANG_PORT              port SGLang listens on            (default 30010)
  MODEL_VARIANT            fl2va | ref2va                    (default fl2va)
  OUTPUT_BUCKET            default S3 bucket for outputs     (optional)
  OUTPUT_PREFIX            default S3 key prefix             (default minimax-h3/out)
  MEDIA_DIR                scratch dir for reference media   (default /data/minimax-h3)
  JOB_POLL_INTERVAL_S      seconds between status polls      (default 2)
  JOB_TIMEOUT_S            max wait for a single job         (default 3000)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid

import boto3
import requests
from flask import Flask, Response, jsonify, request

SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30010"))
SGLANG_BASE = f"http://127.0.0.1:{SGLANG_PORT}"
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "fl2va")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "minimax-h3/out")
MEDIA_DIR = os.environ.get("MEDIA_DIR", "/data/minimax-h3")
POLL_INTERVAL_S = float(os.environ.get("JOB_POLL_INTERVAL_S", "2"))
JOB_TIMEOUT_S = float(os.environ.get("JOB_TIMEOUT_S", "3000"))

app = Flask(__name__)
s3 = boto3.client("s3")

os.makedirs(MEDIA_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# S3 helpers
# --------------------------------------------------------------------------
def _split_s3(uri):
    parsed = urllib.parse.urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _localise_media(uri, scratch):
    """Download an s3:// reference asset and return a server-local file:// URI.

    http(s):// and file:// URIs are passed through untouched.
    """
    if not uri.startswith("s3://"):
        return uri
    bucket, key = _split_s3(uri)
    local = os.path.join(scratch, os.path.basename(key) or f"asset-{uuid.uuid4().hex}")
    s3.download_file(bucket, key, local)
    return f"file://{local}"


def _localise_conditions(conditions, scratch):
    out = []
    for cond in conditions or []:
        cond = dict(cond)
        if "uri" in cond:
            cond["uri"] = _localise_media(cond["uri"], scratch)
        out.append(cond)
    return out


# --------------------------------------------------------------------------
# SGLang job lifecycle
# --------------------------------------------------------------------------
def _submit(payload):
    res = requests.post(f"{SGLANG_BASE}/v1/videos", json=payload, timeout=120)
    res.raise_for_status()
    return res.json()["id"]


def _await_completion(video_id):
    deadline = time.time() + JOB_TIMEOUT_S
    while time.time() < deadline:
        res = requests.get(f"{SGLANG_BASE}/v1/videos/{video_id}", timeout=60)
        res.raise_for_status()
        body = res.json()
        status = body.get("status")
        if status == "completed":
            return body
        if status == "failed":
            raise RuntimeError(f"SGLang job {video_id} failed: {json.dumps(body)[:800]}")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"SGLang job {video_id} exceeded {JOB_TIMEOUT_S}s")


def _download_variant(video_id, variant, dest):
    url = f"{SGLANG_BASE}/v1/videos/{video_id}/content"
    params = {"variant": variant} if variant is not None else None
    with requests.get(url, params=params, stream=True, timeout=600) as res:
        res.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in res.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


# --------------------------------------------------------------------------
# SageMaker contract
# --------------------------------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    """Healthy only once SGLang has finished loading weights and warming up.

    H3 load plus warmup is roughly 2.5 minutes on a warm local cache and much
    longer on a cold one, so set ContainerStartupHealthCheckTimeoutInSeconds
    generously on the endpoint config.
    """
    try:
        res = requests.get(f"{SGLANG_BASE}/health", timeout=5)
        return Response(status=200 if res.status_code == 200 else 503)
    except requests.RequestException:
        return Response(status=503)


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        data = request.get_json(force=True)
    except Exception as exc:
        return jsonify({"error": f"invalid JSON body: {exc}"}), 400

    bucket = data.pop("bucket", OUTPUT_BUCKET)
    if not bucket:
        return jsonify({"error": "no output bucket: set 'bucket' or OUTPUT_BUCKET"}), 400
    key_prefix = data.pop("key_prefix", OUTPUT_PREFIX)
    file_stem = data.pop("file_name", f"h3-{uuid.uuid4().hex[:12]}").removesuffix(".mp4")

    scratch = tempfile.mkdtemp(dir=MEDIA_DIR)
    try:
        # Reference assets must be visible to the SGLang process.
        if "conditions" in data:
            data["conditions"] = _localise_conditions(data["conditions"], scratch)

        # The served checkpoint partition is fixed at launch. fl2va serves
        # t2va and fl2va; ref2va serves ref2va (including video-to-video).
        task = data.get("task", "t2va")
        expected = {"t2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}.get(task)
        if expected and expected != MODEL_VARIANT:
            return jsonify({
                "error": (
                    f"task '{task}' requires --model-variant '{expected}' but this "
                    f"endpoint serves '{MODEL_VARIANT}'. Deploy a second endpoint "
                    f"for the other partition."
                )
            }), 400

        started = time.perf_counter()
        video_id = _submit(data)
        job = _await_completion(video_id)
        elapsed = time.perf_counter() - started

        n_outputs = int(data.get("num_outputs_per_prompt", data.get("n", 1)))
        outputs = []
        for variant in range(n_outputs):
            suffix = "" if n_outputs == 1 else f"-{variant}"
            local = os.path.join(scratch, f"{file_stem}{suffix}.mp4")
            _download_variant(video_id, variant if n_outputs > 1 else None, local)
            key = f"{key_prefix}/{file_stem}{suffix}.mp4"
            s3.upload_file(local, bucket, key)
            outputs.append(f"s3://{bucket}/{key}")

        return jsonify({
            "outputs": outputs,
            "video_id": video_id,
            "model_variant": MODEL_VARIANT,
            "elapsed_seconds": round(elapsed, 2),
            "status": job.get("status"),
        })
    except Exception as exc:
        app.logger.exception("invocation failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _launch_sglang():
    """Start SGLang Diffusion as a child process.

    Flags come from SGLANG_ARGS so the notebook can set the topology per
    instance type without rebuilding the image.
    """
    args = os.environ.get("SGLANG_ARGS", "").split()
    cmd = ["sglang", "serve", "--port", str(SGLANG_PORT), "--host", "127.0.0.1"] + args
    app.logger.info("launching: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    def _watch():
        code = proc.wait()
        app.logger.error("sglang exited with %s; terminating container", code)
        os._exit(1)

    threading.Thread(target=_watch, daemon=True).start()
    return proc


if __name__ == "__main__":
    _launch_sglang()
    app.run(host="0.0.0.0", port=8080, threaded=True)
