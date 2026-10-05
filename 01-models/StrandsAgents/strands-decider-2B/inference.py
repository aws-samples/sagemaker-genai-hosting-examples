"""SageMaker adapter for the official Strands Decider engine."""

import json
import threading
import time
from importlib.metadata import distribution, version

MODEL_ID = "StrandsAgents/strands-decider-2B-hobson-v19"
MODEL_REVISION = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
CODE_REVISION = "75c9fd32e664954cdc18481434018aa507eee8fb"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
SOURCE_URL = f"https://github.com/strands-labs/strands-decider/archive/{CODE_REVISION}.tar.gz"


def runtime_source_url():
    metadata = distribution("strands-decider").read_text("direct_url.json")
    source = json.loads(metadata or "{}").get("url")
    if source != SOURCE_URL:
        raise RuntimeError("The installed Decider runtime does not match the pinned source URL.")
    return source


def model_fn(model_dir):
    import torch
    from huggingface_hub import snapshot_download
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderModel

    source_url = runtime_source_url()
    if not torch.cuda.is_available():
        raise RuntimeError("This deployment requires a CUDA GPU.")
    torch.set_num_threads(2)
    checkpoint = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "hobson_config.json", "head.safetensors", "lora/*",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "provenance.json", "MANIFEST.sha256", "LICENSE.md",
        ],
    )
    with torch.inference_mode():
        model = StrandsDeciderModel.load(checkpoint, device_map="cuda")
        if model.config.base_revision != BASE_REVISION:
            raise RuntimeError("Unexpected base-model revision.")
        engine = SystemOneEngine(
            model,
            EngineConfig(
                device="cuda", model_name=MODEL_ID, strict_window=True,
                use_prefix_cache=True, max_batch=4,
            ),
        )
    print(json.dumps({"ready": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    return {"engine": engine, "lock": threading.Lock(), "source_url": source_url}


def transform_fn(model, request_body, content_type, accept):
    import torch
    from sagemaker_inference.errors import GenericInferenceToolkitError
    from strands_decider.schema import SystemOneRequest

    if content_type.split(";")[0].strip() != "application/json":
        raise GenericInferenceToolkitError(415, "ContentType must be application/json.")
    if accept and accept not in ("application/json", "*/*"):
        raise GenericInferenceToolkitError(406, "Accept must be application/json.")
    try:
        payload = json.loads(request_body)
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
    except ValueError as exc:
        raise GenericInferenceToolkitError(400, str(exc)) from exc
    if payload.get("operation") == "diagnostics":
        engine = model["engine"]
        response = {
            "model": engine.cfg.model_name, "model_revision": MODEL_REVISION,
            "code_revision": CODE_REVISION,
            "source_url": model["source_url"],
            "base_revision": engine.model.config.base_revision,
            "device": engine.cfg.device, "gpu": torch.cuda.get_device_name(0),
            "max_length": engine.model.config.max_length,
            "strict_window": engine.cfg.strict_window, "max_batch": engine.cfg.max_batch,
            "versions": {
                package: version(package)
                for package in ["torch", "transformers", "peft", "huggingface_hub"]
            },
            "cuda_runtime": torch.version.cuda,
            "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
        }
    else:
        try:
            request = SystemOneRequest.model_validate(payload)
            started = time.perf_counter()
            # The official server does not establish concurrent-engine safety.
            # Keep the model worker serial and do not share mutable inference caches.
            with model["lock"], torch.inference_mode():
                response = model["engine"].evaluate(request).model_dump()
        except ValueError as exc:
            raise GenericInferenceToolkitError(422, str(exc)) from exc
        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return json.dumps(response), "application/json"
