"""SageMaker adapter for the pinned, model-native Decision 2.0 runtime on CPU."""

import json
import os
import platform
import resource
import sys
import threading
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Decision 2.0 Kai 0.6B by default. The notebook may select Decision 2.0 Eos 0.8B
# (the 16,384-token model of the same runtime family) through these variables.
MODEL_ID = os.environ.get("DECISION_MODEL_ID", "vllm-sr/Decision-2.0-Kai-0.6B")
MODEL_REVISION = os.environ.get(
    "DECISION_MODEL_REVISION", "cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b"
)


def _thread_count():
    """Torch intra-op threads: DECISION_THREADS, else every vCPU of the instance."""
    requested = os.environ.get("DECISION_THREADS")
    threads = int(requested) if requested else (os.cpu_count() or 1)
    if threads < 1:
        raise ValueError("DECISION_THREADS must be a positive integer.")
    return threads


def model_fn(model_dir):
    import torch
    from transformers import AutoModel

    threads = _thread_count()
    with torch.inference_mode():
        # The native runtime fixes CPU numerics at FP32 and rejects dtype overrides;
        # `threads` is its intra-op thread setting (torch.set_num_threads).
        engine = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=MODEL_REVISION,
            device="cpu",
            threads=threads,
        )
    loaded_revision = getattr(engine.config, "_commit_hash", None)
    if loaded_revision is None:
        source = getattr(engine, "_source", None)
        if source is not None and Path(source).parent.name == "snapshots":
            loaded_revision = Path(source).name
    if loaded_revision != MODEL_REVISION:
        raise RuntimeError(
            f"Loaded model revision {loaded_revision!r} does not match {MODEL_REVISION}."
        )
    if str(engine.runtime.backend.device) != "cpu":
        raise RuntimeError("This deployment expects the CPU runtime path.")
    print(
        json.dumps({
            "ready": engine.model_name,
            "model_revision": loaded_revision,
            "device": str(engine.runtime.backend.device),
            "torch_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
        }),
        flush=True,
    )
    return {
        "model": engine, "lock": threading.Lock(),
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
    engine = worker["model"]
    backend = engine.runtime.backend
    manifest = engine.manifest
    parameters = list(backend.model.parameters())
    dtypes = Counter()
    for parameter in parameters:
        dtypes[str(parameter.dtype)] += parameter.numel()
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "model": engine.model_name,
        "model_revision": worker["model_revision"],
        "requested_model_revision": MODEL_REVISION,
        "source_url": worker["source_url"],
        "source_identity": {
            "model_sha256": manifest["identity"]["model_sha256"],
            "profile": manifest.get("profile"),
            "runtime_source": manifest.get("runtime", {}).get("runtime_source"),
            "remote_code_source": manifest.get("remote_code", {}).get("automap_source"),
        },
        "device": str(backend.device),
        "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
        "processor": _processor_name(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("DECISION_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cpu_reference_layers": getattr(engine, "cpu_reference_layers", 0),
        "loaded_parameters": backend.parameter_count(),
        "parameter_dtypes": dict(dtypes),
        "head_parameter_dtypes": sorted({
            str(parameter.dtype) for parameter in backend.model.head.parameters()
        }),
        "max_input_tokens": engine.max_input_tokens,
        "backend_residency": getattr(backend, "residency", None),
        "backend_fast_path": getattr(backend, "fast", None) is not None,
        "versions": {
            package: _package_version(package)
            for package in (
                "torch", "transformers", "safetensors", "huggingface_hub",
                "accelerate", "numpy", "peft", "sagemaker-inference",
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

    # Serialize every request: the single worker's CPU threads and the runtime's
    # mutable caches belong to one inference at a time.
    with model["lock"], torch.inference_mode():
        if diagnostics:
            response = _diagnostics(model, torch)
        else:
            started = time.perf_counter()
            # Question validation and all answer fields/errors remain native.
            # Runtime exceptions propagate; they are not all input errors.
            response = model["model"].system_one(
                state=payload["state"], questions=questions,
            )
            response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        body = json.dumps(response, allow_nan=False)
        if not diagnostics and any(
            answer.get("error") == "max_length_exceeded"
            for answer in response["answers"].values()
        ):
            if context is None:
                raise GenericInferenceToolkitError(422, body, "Unprocessable Entity")
            context.set_response_status(code=422, phrase="Unprocessable Entity")
    return body, "application/json"
