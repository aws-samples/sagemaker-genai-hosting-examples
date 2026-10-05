"""SageMaker adapter for the pinned, model-native Decision 2.0 runtime."""

import json
import resource
import sys
import threading
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MODEL_ID = "vllm-sr/Decision-2.0-Sol-2B"
MODEL_REVISION = "64235bef55dad29387dd16da7c90e038bf2f0972"


def model_fn(model_dir):
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("This deployment requires a CUDA GPU.")
    torch.set_num_threads(2)
    with torch.inference_mode():
        # The native loader keeps the head in FP32 and selects BF16-exact
        # backbone Linear weights for residency. Do not override its dtype.
        engine = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=MODEL_REVISION,
            device="cuda:0",
            bf16_resident=True,
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
    print(
        json.dumps({
            "ready": engine.model_name,
            "model_revision": loaded_revision,
            "device": str(engine.runtime.backend.device),
            "gpu": torch.cuda.get_device_name(engine.runtime.backend.device),
        }),
        flush=True,
    )
    return {
        "model": engine, "lock": threading.Lock(), "has_request": False,
        "model_revision": loaded_revision,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{loaded_revision}",
    }


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


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
            "runtime_source": manifest.get("runtime", {}).get("runtime_source"),
            "remote_code_source": manifest.get("remote_code", {}).get("automap_source"),
        },
        "device": str(backend.device),
        "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
        "gpu": torch.cuda.get_device_name(backend.device),
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
                "accelerate", "numpy", "torchvision", "torchaudio",
                "peft", "flash-linear-attention", "causal-conv1d",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(backend.device),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(backend.device),
        "gpu_peak_scope": "last_request" if worker["has_request"] else "worker_startup",
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

    # Serialize diagnostics too: allocator peaks and mutable backend caches
    # belong to this worker and must not race another request.
    with model["lock"], torch.inference_mode():
        if diagnostics:
            response = _diagnostics(model, torch)
        else:
            device = model["model"].runtime.backend.device
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            model["has_request"] = True
            started = time.perf_counter()
            # Question validation and all answer fields/errors remain native.
            # Runtime exceptions propagate; they are not all input errors.
            response = model["model"].system_one(
                state=payload["state"], questions=questions,
            )
            torch.cuda.synchronize(device)
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
