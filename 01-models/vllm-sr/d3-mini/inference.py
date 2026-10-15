"""SageMaker adapter for the pinned, model-native Decision 3.0 (d3) runtime on one GPU."""

import json
import os
import resource
import sys
import threading
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# d3-mini by default. The notebook selects d3-nano or d3-flash through these variables;
# all three share the d3 runtime, the request shape, and this adapter.
MODEL_ID = os.environ.get("DECISION_MODEL_ID", "vllm-sr/d3-mini")
MODEL_REVISION = os.environ.get(
    "DECISION_MODEL_REVISION", "f15a639c070dccd1da611d83c0618c77be8c23d6"
)
# Questions per forward pass (the runtime's default is 8).
BATCH_SIZE = int(os.environ.get("D3_BATCH_SIZE", "8"))
# Compile and autotune the Gated DeltaNet kernels and warm the vision tower before serving.
WARMUP = os.environ.get("D3_WARMUP", "1").strip().lower() not in ("0", "false", "no", "off")
# A request may carry these fields, as in the package's own d3_server.py.
ACCEPTED_FIELDS = ("state", "questions", "images", "videos", "model")
# The GPU. "cpu" is accepted only to exercise this adapter where no GPU exists; it is
# slow, and the flash-linear-attention kernels must then be absent.
DEVICE = os.environ.get("D3_DEVICE", "cuda:0")


def model_fn(model_dir):
    import torch
    from transformers import AutoModel

    cuda = torch.device(DEVICE).type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("This deployment requires a CUDA GPU.")
    with torch.inference_mode():
        # The runtime fixes numerics: BF16 backbone loaded straight to the GPU, FP32
        # readout. It verifies the package against MODEL_MANIFEST.json ("fast").
        engine = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=MODEL_REVISION,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            verify="fast",
        )
    runtime = engine.runtime
    loaded_revision = getattr(engine.config, "_commit_hash", None)
    if loaded_revision is None and Path(runtime.root).parent.name == "snapshots":
        loaded_revision = Path(runtime.root).name
    if loaded_revision != MODEL_REVISION:
        raise RuntimeError(
            f"Loaded model revision {loaded_revision!r} does not match {MODEL_REVISION}."
        )
    warmup_seconds = None
    if WARMUP:
        with torch.inference_mode():
            warmup_seconds = round(runtime.warmup(), 2)
    print(
        json.dumps({
            "ready": runtime.model_name,
            "model_revision": loaded_revision,
            "device": str(runtime.device),
            "gpu": torch.cuda.get_device_name(runtime.device) if cuda else None,
            "kernels": runtime.kernels,
            "loaded_seconds": round(runtime.loaded_seconds, 2),
            "warmup_seconds": warmup_seconds,
        }),
        flush=True,
    )
    return {
        "model": engine, "runtime": runtime, "lock": threading.Lock(), "has_request": False,
        "cuda": cuda,
        "model_revision": loaded_revision, "warmup_seconds": warmup_seconds,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{loaded_revision}",
    }


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _diagnostics(worker, torch):
    runtime = worker["runtime"]
    parameters = list(runtime.backbone.parameters())
    dtypes = Counter()
    for parameter in parameters:
        dtypes[str(parameter.dtype)] += parameter.numel()
    dtypes[str(runtime.readout.dtype)] += runtime.readout.numel()
    manifest = runtime.manifest or {}
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cuda = worker["cuda"]
    return {
        "model": runtime.model_name,
        "model_id": MODEL_ID,
        "model_revision": worker["model_revision"],
        "requested_model_revision": MODEL_REVISION,
        "source_url": worker["source_url"],
        "source_identity": {
            "model_sha256": manifest.get("identity", {}).get("model_sha256"),
            "base_model": manifest.get("base_model"),
            "built_utc": manifest.get("built_utc"),
        },
        "device": str(runtime.device),
        "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
        "gpu": torch.cuda.get_device_name(runtime.device) if cuda else None,
        "loaded_parameters": runtime.parameter_count(),
        "manifest_parameters": manifest.get("parameters"),
        "parameter_dtypes": dict(dtypes),
        "max_input_tokens": runtime.max_length,
        "batch_size": runtime.batch_size,
        "attention_mode": runtime.attention_mode,
        # Which implementation Transformers bound for the Gated DeltaNet layers:
        # flash-linear-attention kernels or its PyTorch reference.
        "kernels": runtime.kernels,
        "fast_path": runtime.fast_report(),
        "images": runtime.image_contract(),
        "videos": runtime.video_contract(),
        "loaded_seconds": round(runtime.loaded_seconds, 2),
        "warmup_seconds": worker["warmup_seconds"],
        "versions": {
            package: _package_version(package)
            for package in (
                "torch", "torchvision", "transformers", "accelerate", "safetensors",
                "huggingface_hub", "numpy", "pillow", "opencv-python",
                "opencv-python-headless", "flash-linear-attention", "fla-core",
                "einops", "triton", "causal-conv1d", "sagemaker-inference",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(runtime.device) if cuda else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device) if cuda else None,
        "gpu_peak_scope": "last_request" if worker["has_request"] else "worker_startup",
        # Linux reports ru_maxrss in KiB; macOS reports bytes.
        "worker_peak_rss_bytes": highwater * (1 if sys.platform == "darwin" else 1024),
    }


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _unprocessable(detail):
    from sagemaker_inference.errors import GenericInferenceToolkitError

    return GenericInferenceToolkitError(
        422, json.dumps({"detail": detail}), "Unprocessable Entity"
    )


def _media(runtime, payload, name):
    """Decode a request's images or videos in the server's strict mode: base64 data URLs only.

    Local paths and http(s) URLs are refused, so the endpoint never reads its own file
    system or fetches from the network on a caller's behalf.
    """
    items = payload.get(name)
    if items is None:
        return None
    if not isinstance(items, list):
        raise _unprocessable(f"{name} takes a list of base64 data URLs")
    loader = runtime.load_images if name == "images" else runtime.load_videos
    try:
        return loader(items, strict=True) or None
    except ValueError as exc:
        raise _unprocessable(str(exc)) from exc


def transform_fn(model, request_body, content_type, accept, context=None):
    import torch
    from sagemaker_inference.errors import GenericInferenceToolkitError

    if (content_type or "").split(";")[0].strip().lower() != "application/json":
        raise GenericInferenceToolkitError(415, "ContentType must be application/json.")
    if accept and accept.split(";")[0].strip().lower() not in ("application/json", "*/*"):
        raise GenericInferenceToolkitError(406, "Accept must be application/json.")
    try:
        payload = json.loads(request_body, parse_constant=_reject_constant)
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
        # Also reject overflowed JSON numbers (e.g. 1e999), including in state.
        json.dumps(payload, allow_nan=False)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise GenericInferenceToolkitError(400, str(exc)) from exc

    diagnostics = payload.get("operation") == "diagnostics"
    if not diagnostics:
        unknown = sorted(set(payload) - set(ACCEPTED_FIELDS))
        if unknown:
            raise _unprocessable(
                f"{', '.join(map(repr, unknown))} not understood; "
                f"a request may carry {', '.join(ACCEPTED_FIELDS)}"
            )
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions or any(not key for key in questions):
            raise _unprocessable("questions must be a nonempty object of question IDs.")

    runtime, cuda = model["runtime"], model["cuda"]
    # Serialize every request, diagnostics included: allocator peaks, the runtime's
    # media handling, and the single GPU belong to one inference at a time.
    with model["lock"], torch.inference_mode():
        if diagnostics:
            response = _diagnostics(model, torch)
        else:
            images = _media(runtime, payload, "images")
            videos = _media(runtime, payload, "videos")
            if cuda:
                torch.cuda.synchronize(runtime.device)
                torch.cuda.reset_peak_memory_stats(runtime.device)
            model["has_request"] = True
            started = time.perf_counter()
            try:
                # As the package server does: malformed questions are refused before
                # anything runs; every other answer field and error remains native.
                prepared = runtime.prepare(payload.get("state"), questions, images, videos)
                invalid = {
                    key: error["message"]
                    for key, error in prepared.errors.items()
                    if error.get("error") == "invalid_question"
                }
                if invalid:
                    raise _unprocessable({"invalid_questions": invalid})
                probabilities, tokens = runtime.run(prepared)
                response = runtime.respond(prepared, probabilities, tokens)
            except ValueError as exc:
                raise _unprocessable(str(exc)) from exc
            runtime.synchronize()
            response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            if cuda:
                response["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(runtime.device)
        body = json.dumps(response, allow_nan=False)
    return body, "application/json"
