"""SageMaker adapter for the pinned, model-native Vela 2.0 0.3B runtime on CPU."""

import json
import os
import platform
import resource
import sys
import threading
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

MODEL_ID = "vllm-sr/Vela-2.0-0.3B"
MODEL_REVISION = "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"


def _thread_count():
    """Torch intra-op threads: VELA_THREADS, else every vCPU of the instance."""
    requested = os.environ.get("VELA_THREADS")
    threads = int(requested) if requested else (os.cpu_count() or 1)
    if threads < 1:
        raise ValueError("VELA_THREADS must be a positive integer.")
    return threads


def model_fn(model_dir):
    import torch
    from transformers import AutoModel

    torch.set_num_threads(_thread_count())
    with torch.inference_mode():
        # Vela 2.0 loads through Transformers' own weight loader in FP32; the model card
        # does not support BF16/FP16 Torch loading. Do not pass a dtype.
        model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, revision=MODEL_REVISION,
        )
    model.eval()
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision != MODEL_REVISION:
        raise RuntimeError(
            f"Loaded model revision {loaded_revision!r} does not match {MODEL_REVISION}."
        )
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise RuntimeError("This deployment expects the CPU runtime path.")
    # Build the published engine (tokenizer, calibration, windowing, span decoding) now,
    # so the first request does not pay for it.
    engine = model.vela2_engine
    print(
        json.dumps({
            "ready": MODEL_ID,
            "model_revision": loaded_revision,
            "device": str(device),
            "max_input_tokens": engine.max_len,
            "torch_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
        }),
        flush=True,
    )
    return {
        "model": model, "engine": engine, "lock": threading.Lock(),
        "model_revision": loaded_revision,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{loaded_revision}",
    }


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _processor_name():
    """The host CPU model from /proc/cpuinfo on Linux; the machine type elsewhere."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _diagnostics(worker, torch):
    model, engine = worker["model"], worker["engine"]
    parameters = list(model.parameters())
    dtypes = Counter()
    for parameter in parameters:
        dtypes[str(parameter.dtype)] += parameter.numel()
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "model": MODEL_ID,
        "model_revision": worker["model_revision"],
        "requested_model_revision": MODEL_REVISION,
        "source_url": worker["source_url"],
        "model_type": model.config.model_type,
        "backbone": {
            "model_type": model.encoder.config.model_type,
            "num_hidden_layers": model.encoder.config.num_hidden_layers,
            "hidden_size": model.encoder.config.hidden_size,
            "attn_implementation": model.encoder.config._attn_implementation,
        },
        "device": str(next(model.parameters()).device),
        "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
        "processor": _processor_name(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("VELA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "loaded_parameters": sum(parameter.numel() for parameter in parameters),
        "parameter_dtypes": dict(dtypes),
        "max_input_tokens": engine.max_len,
        "window_overlap": engine.overlap,
        "forward_token_budget": engine.budget,
        "forward_batch_size": engine.bs,
        "question_types": ["noul", "choice", "score", "span", "set"],
        "calibration_temperatures": engine.cal["temperature"],
        "calibration_thresholds": engine.cal["thresholds"],
        # Cumulative since the worker started: how often a long part was windowed.
        "engine_stats": dict(engine.stats),
        "versions": {
            package: _package_version(package)
            for package in (
                "torch", "transformers", "tokenizers", "safetensors",
                "huggingface_hub", "numpy", "sagemaker-inference",
            )
        },
        "cuda_available": torch.cuda.is_available(),
        # Linux reports ru_maxrss in KiB; macOS reports bytes.
        "worker_peak_rss_bytes": highwater * (1 if sys.platform == "darwin" else 1024),
    }


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


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
        if not isinstance(payload.get("state"), (str, dict, list)):
            raise GenericInferenceToolkitError(422, "state must be text, an object, or an array.")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions or any(not key for key in questions):
            raise GenericInferenceToolkitError(422, "questions must be a nonempty object of question IDs.")

    # Serialize every request: the single worker's CPU threads and the engine's
    # tokenization cache belong to one inference at a time.
    with model["lock"], torch.inference_mode():
        if diagnostics:
            response = _diagnostics(model, torch)
        else:
            started = time.perf_counter()
            try:
                # Question validation, windowing, calibration, span decoding, and all
                # answer fields remain native. Other runtime exceptions propagate.
                response = model["model"].system_one(
                    state=payload["state"], questions=questions,
                )
            except ValueError as exc:
                detail = getattr(exc, "detail", None)
                if not callable(detail):
                    raise
                # The published SystemOne server maps these the same way: 413 when the
                # questions alone exceed the token window, 422 for an invalid request.
                status = 413 if getattr(exc, "type", None) == "too_long" else 422
                phrase = "Payload Too Large" if status == 413 else "Unprocessable Entity"
                raise GenericInferenceToolkitError(
                    status, json.dumps({"detail": [detail()]}), phrase,
                ) from exc
            response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        body = json.dumps(response, allow_nan=False)
    return body, "application/json"
