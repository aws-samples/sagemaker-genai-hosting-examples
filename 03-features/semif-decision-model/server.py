"""SageMaker HTTP adapter for SemIf's native llama.cpp option scorer."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

MAX_BODY_BYTES = 1024 * 1024
_GGML_LIBRARY: ctypes.CDLL | None = None
LOGGER = logging.getLogger(__name__)


def load_ggml_backends(library_dir: Path) -> list[str]:
    """Load every backend plugin supplied by the selected llama.cpp DLC."""
    global _GGML_LIBRARY
    library_path = library_dir / "libggml.so"
    if not library_path.is_file():
        raise RuntimeError(f"libggml.so does not exist: {library_path}")
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    library.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    library.ggml_backend_load_all_from_path.restype = None
    library.ggml_backend_reg_count.argtypes = []
    library.ggml_backend_reg_count.restype = ctypes.c_size_t
    library.ggml_backend_reg_get.argtypes = [ctypes.c_size_t]
    library.ggml_backend_reg_get.restype = ctypes.c_void_p
    library.ggml_backend_reg_name.argtypes = [ctypes.c_void_p]
    library.ggml_backend_reg_name.restype = ctypes.c_char_p
    library.ggml_backend_load_all_from_path(str(library_dir).encode("utf-8"))
    count = int(library.ggml_backend_reg_count())
    if count < 1:
        raise RuntimeError(f"No llama.cpp backend loaded from {library_dir}")
    names = []
    for index in range(count):
        registration = library.ggml_backend_reg_get(index)
        raw_name = library.ggml_backend_reg_name(registration)
        names.append(raw_name.decode("utf-8") if raw_name else f"unknown-{index}")
    _GGML_LIBRARY = library
    return names


def require_gpu_backend(backend_names: list[str], n_gpu_layers: int) -> None:
    if n_gpu_layers == 0:
        return
    if not any("CUDA" in name.upper() for name in backend_names):
        raise RuntimeError(
            "SEMIF_N_GPU_LAYERS requests GPU offload, but no CUDA backend is loaded"
        )


def model_params_with_gpu_layers(
    original: Callable[[Any], Any],
    n_gpu_layers: int,
) -> Callable[[Any], Any]:
    def configured(library):
        params = original(library)
        params.n_gpu_layers = n_gpu_layers
        return params

    return configured


class Scorer(Protocol):
    metadata: dict[str, Any]

    def score(self, decision: dict[str, Any]) -> dict[str, Any]: ...

    def score_shared(
        self, decisions: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    def close(self) -> None: ...


class SemifScorer:
    """Own one SemIf context and serialize its stateful model operations."""

    def __init__(
        self,
        gguf_path: Path,
        tokenizer_source: str,
        tokenizer_revision: str,
        *,
        threads: int | None,
        n_gpu_layers: int,
        max_tokens: int,
    ):
        library_dir = Path(
            os.environ.get("SEMIF_LLAMA_CPP_LIB_DIR", "/opt/llama.cpp/bin")
        )
        backend_names = load_ggml_backends(library_dir)
        require_gpu_backend(backend_names, n_gpu_layers)
        from semif_phase1 import llamacpp_backend

        self._backend_module = llamacpp_backend
        original_model_params = llamacpp_backend._cpu_model_params
        llamacpp_backend._cpu_model_params = model_params_with_gpu_layers(
            original_model_params,
            n_gpu_layers,
        )
        try:
            self._model, self._tokenizer, self.metadata = llamacpp_backend.load_model(
                tokenizer_source,
                tokenizer_revision,
                gguf_path,
                threads=threads,
                context_tokens=max_tokens,
            )
        finally:
            llamacpp_backend._cpu_model_params = original_model_params
        self.metadata.update(
            {
                "n_gpu_layers": n_gpu_layers,
                "registered_backends": backend_names,
                "gpu_offload_requested": n_gpu_layers != 0,
            }
        )
        self._max_tokens = max_tokens
        self._lock = threading.Lock()

    def score(self, decision: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._backend_module.score(
                self._model,
                self._tokenizer,
                decision,
                self.metadata,
                self._max_tokens,
            )

    def score_shared(
        self, decisions: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with self._lock:
            return self._backend_module.score_shared(
                self._model,
                self._tokenizer,
                decisions,
                self.metadata,
                self._max_tokens,
            )

    def close(self) -> None:
        self._model.close()


def _positive_integer(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def gpu_layers_from_environment() -> int:
    raw = os.environ.get("SEMIF_N_GPU_LAYERS", "0")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(
            "SEMIF_N_GPU_LAYERS must be -1 or a non-negative integer"
        ) from error
    if value < -1:
        raise RuntimeError(
            "SEMIF_N_GPU_LAYERS must be -1 or a non-negative integer"
        )
    return value


def discover_gguf(model_dir: Path, explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise RuntimeError(f"SEMIF_GGUF_PATH does not exist: {path}")
        return path
    matches = sorted(model_dir.glob("*.gguf")) + sorted(model_dir.glob("*/*.gguf"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one GGUF under {model_dir}, found {len(matches)}; "
            "set SEMIF_GGUF_PATH explicitly"
        )
    return matches[0]


def load_scorer_from_environment() -> SemifScorer:
    model_dir = Path(os.environ.get("SEMIF_MODEL_DIR", "/opt/ml/model"))
    gguf_path = discover_gguf(model_dir, os.environ.get("SEMIF_GGUF_PATH"))
    tokenizer_source = os.environ.get(
        "SEMIF_TOKENIZER_SOURCE", str(model_dir / "tokenizer")
    )
    tokenizer_revision = os.environ.get("SEMIF_TOKENIZER_REVISION")
    if not tokenizer_revision:
        raise RuntimeError("SEMIF_TOKENIZER_REVISION is required")
    return SemifScorer(
        gguf_path,
        tokenizer_source,
        tokenizer_revision,
        threads=_positive_integer("SEMIF_THREADS"),
        n_gpu_layers=gpu_layers_from_environment(),
        max_tokens=_positive_integer("SEMIF_MAX_TOKENS", 4096) or 4096,
    )


def parse_request(
    payload: Any,
    *,
    max_shared_decisions: int = 64,
) -> tuple[str, dict[str, Any] | list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")  # noqa: TRY004
    unsupported = set(payload) - {"decision", "decisions"}
    if unsupported:
        raise ValueError(f"Unsupported request fields: {sorted(unsupported)}")
    supplied = [name for name in ("decision", "decisions") if name in payload]
    if len(supplied) != 1:
        raise ValueError(
            "Request must contain exactly one of decision or decisions"
        )
    if supplied[0] == "decision":
        decision = payload["decision"]
        if not isinstance(decision, dict):
            raise ValueError(
                "Request must contain one decision object"
            )
        return "single", decision

    decisions = payload["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(
            "Request must contain a nonempty decisions list"
        )
    if len(decisions) > max_shared_decisions:
        raise ValueError(
            f"decisions supports at most {max_shared_decisions} items"
        )
    if any(not isinstance(decision, dict) for decision in decisions):
        raise ValueError("decisions must contain only decision objects")
    return "shared", decisions


def make_handler(
    scorer: Scorer,
    *,
    max_in_flight: int = 1,
    max_shared_decisions: int = 64,
):
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be a positive integer")
    if max_shared_decisions < 1:
        raise ValueError("max_shared_decisions must be a positive integer")
    admission = threading.BoundedSemaphore(max_in_flight)

    class Handler(BaseHTTPRequestHandler):
        server_version = "SemIfSageMaker/0.1"

        def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/ping":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/invocations":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "Content-Type must be application/json"},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    raise ValueError(
                        f"Content-Length must be between 1 and {MAX_BODY_BYTES} bytes"
                    )
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if not admission.acquire(blocking=False):
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": "model is busy; retry the request",
                        "retryable": True,
                    },
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
                mode, work = parse_request(
                    payload,
                    max_shared_decisions=max_shared_decisions,
                )
                if mode == "single":
                    result = scorer.score(work)
                    response = {"result": result}
                else:
                    results, timing = scorer.score_shared(work)
                    response = {"results": results, "timing": timing}
            except (ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception:
                LOGGER.exception("SemIf scoring failed")
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "scoring failed"},
                )
                return
            finally:
                admission.release()
            self._json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return Handler


def main() -> None:
    scorer = load_scorer_from_environment()
    port = _positive_integer("SEMIF_PORT", 8080) or 8080
    max_in_flight = _positive_integer("SEMIF_MAX_IN_FLIGHT", 1) or 1
    max_shared_decisions = (
        _positive_integer("SEMIF_MAX_SHARED_DECISIONS", 64) or 64
    )
    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        make_handler(
            scorer,
            max_in_flight=max_in_flight,
            max_shared_decisions=max_shared_decisions,
        ),
    )
    try:
        print(
            json.dumps(
                {
                    "event": "ready",
                    "port": port,
                    "max_in_flight": max_in_flight,
                    "max_shared_decisions": max_shared_decisions,
                    "model": scorer.metadata,
                },
                allow_nan=False,
            ),
            flush=True,
        )
        server.serve_forever()
    finally:
        server.server_close()
        scorer.close()


if __name__ == "__main__":
    main()
