import importlib
import json
import socket
import sys
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Event, Thread
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

SAMPLE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(SAMPLE_ROOT))

server = importlib.import_module("server")


def sample_decision() -> dict:
    return {
        "id": "route-1",
        "state": "The customer cannot sign in.",
        "question": "Which queue should handle this request?",
        "options": [
            {"id": "access", "description": "Account access support."},
            {"id": "billing", "description": "Billing support."},
        ],
    }


class FakeScorer:
    metadata: ClassVar[dict[str, str]] = {"backend": "fake"}

    def score(self, decision: dict) -> dict:
        return {
            "id": decision["id"],
            "option_ids": [option["id"] for option in decision["options"]],
            "probabilities": [0.8, 0.2],
        }

    def score_shared(self, decisions: list[dict]) -> tuple[list[dict], dict]:
        return (
            [self.score(decision) for decision in decisions],
            {
                "batch_size": len(decisions),
                "prefix_tokens": 12,
                "total_seconds": 0.2,
            },
        )

    def close(self) -> None:
        pass


@pytest.fixture
def endpoint():
    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(FakeScorer())
    )
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def request(
    endpoint,
    method: str,
    path: str,
    body=None,
    content_type: str = "application/json",
) -> tuple[int, dict]:
    connection = HTTPConnection(*endpoint)
    encoded = None if body is None else json.dumps(body)
    headers = {} if body is None else {"Content-Type": content_type}
    connection.request(method, path, encoded, headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    return response.status, payload


def test_ping_reports_ready(endpoint):
    assert request(endpoint, "GET", "/ping") == (
        HTTPStatus.OK,
        {"status": "ok"},
    )


def test_invocations_returns_typed_option_scores(endpoint):
    status, payload = request(
        endpoint,
        "POST",
        "/invocations",
        {"decision": sample_decision()},
    )

    assert status == HTTPStatus.OK
    assert payload == {
        "result": {
            "id": "route-1",
            "option_ids": ["access", "billing"],
            "probabilities": [0.8, 0.2],
        }
    }
    assert sum(payload["result"]["probabilities"]) == pytest.approx(1.0)


def test_invocations_scores_multiple_decisions_with_one_shared_state(endpoint):
    second = sample_decision()
    second["id"] = "priority-1"
    second["question"] = "How urgent is this request?"
    second["options"] = [
        {"id": "urgent", "description": "Handle immediately."},
        {"id": "normal", "description": "Use the standard queue."},
    ]

    status, payload = request(
        endpoint,
        "POST",
        "/invocations",
        {"decisions": [sample_decision(), second]},
    )

    assert status == HTTPStatus.OK
    assert [result["id"] for result in payload["results"]] == [
        "route-1",
        "priority-1",
    ]
    assert payload["timing"] == {
        "batch_size": 2,
        "prefix_tokens": 12,
        "total_seconds": 0.2,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({}, "exactly one"),
        ({"decision": []}, "one decision object"),
        ({"decisions": []}, "nonempty decisions list"),
        ({"decisions": [sample_decision(), []]}, "decision objects"),
        (
            {"decision": sample_decision(), "decisions": [sample_decision()]},
            "exactly one",
        ),
        (
            {"decision": sample_decision(), "mode": "shared"},
            "Unsupported request fields",
        ),
    ],
)
def test_invalid_request_shape_is_rejected(endpoint, payload, message):
    status, response = request(endpoint, "POST", "/invocations", payload)

    assert status == HTTPStatus.BAD_REQUEST
    assert message in response["error"]


def test_wrong_content_type_is_rejected(endpoint):
    status, payload = request(
        endpoint,
        "POST",
        "/invocations",
        {"decision": sample_decision()},
        "text/plain",
    )

    assert status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    assert "application/json" in payload["error"]


def test_unknown_routes_return_not_found(endpoint):
    assert request(endpoint, "GET", "/unknown") == (
        HTTPStatus.NOT_FOUND,
        {"error": "not found"},
    )
    assert request(endpoint, "POST", "/unknown", {}) == (
        HTTPStatus.NOT_FOUND,
        {"error": "not found"},
    )


def test_scoring_errors_do_not_expose_a_success_response():
    scorer = MagicMock()
    scorer.score.side_effect = RuntimeError("model context failed")
    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(scorer)
    )
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = request(
            httpd.server_address,
            "POST",
            "/invocations",
            {"decision": sample_decision()},
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    assert status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert payload == {"error": "scoring failed"}


def test_shared_scoring_validation_errors_return_bad_request():
    scorer = MagicMock()
    scorer.score_shared.side_effect = ValueError(
        "Shared scoring requires one nonempty exact state"
    )
    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(scorer)
    )
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = request(
            httpd.server_address,
            "POST",
            "/invocations",
            {"decisions": [sample_decision()]},
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    assert status == HTTPStatus.BAD_REQUEST
    assert payload == {
        "error": "Shared scoring requires one nonempty exact state"
    }


def test_busy_endpoint_rejects_excess_work_without_queueing():
    entered = Event()
    release = Event()

    class BlockingScorer(FakeScorer):
        def score(self, decision: dict) -> dict:
            entered.set()
            assert release.wait(timeout=5)
            return super().score(decision)

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        server.make_handler(BlockingScorer(), max_in_flight=1),
    )
    server_thread = Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    first_response = []
    first_thread = Thread(
        target=lambda: first_response.append(
            request(
                httpd.server_address,
                "POST",
                "/invocations",
                {"decision": sample_decision()},
            )
        )
    )
    first_thread.start()
    try:
        assert entered.wait(timeout=5)
        status, payload = request(
            httpd.server_address,
            "POST",
            "/invocations",
            {"decision": sample_decision()},
        )
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload == {
            "error": "model is busy; retry the request",
            "retryable": True,
        }
    finally:
        release.set()
        first_thread.join(timeout=5)
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()

    assert first_response[0][0] == HTTPStatus.OK


def test_busy_endpoint_rejects_request_before_reading_its_body():
    entered = Event()
    release = Event()

    class BlockingScorer(FakeScorer):
        def score(self, decision: dict) -> dict:
            entered.set()
            assert release.wait(timeout=5)
            return super().score(decision)

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        server.make_handler(BlockingScorer(), max_in_flight=1),
    )
    server_thread = Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    first_thread = Thread(
        target=lambda: request(
            httpd.server_address,
            "POST",
            "/invocations",
            {"decision": sample_decision()},
        )
    )
    first_thread.start()
    second_socket = None
    response_started = False
    encoded = json.dumps({"decision": sample_decision()}).encode("utf-8")
    try:
        assert entered.wait(timeout=5)
        second_socket = socket.create_connection(httpd.server_address, timeout=2)
        second_socket.settimeout(1)
        second_socket.sendall(
            (
                "POST /invocations HTTP/1.1\r\n"
                f"Host: {httpd.server_address[0]}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(encoded)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
        )

        response = HTTPResponse(second_socket)
        response.begin()
        response_started = True
        payload = json.loads(response.read())

        assert response.status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload == {
            "error": "model is busy; retry the request",
            "retryable": True,
        }
    finally:
        if second_socket is not None:
            if not response_started:
                try:
                    second_socket.sendall(encoded)
                except OSError:
                    pass
            second_socket.close()
        release.set()
        first_thread.join(timeout=5)
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()


def test_handler_rejects_invalid_admission_limit():
    with pytest.raises(ValueError, match="max_in_flight"):
        server.make_handler(FakeScorer(), max_in_flight=0)


def test_shared_batch_limit_is_enforced_before_scoring():
    scorer = MagicMock()
    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        server.make_handler(scorer, max_shared_decisions=1),
    )
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = request(
            httpd.server_address,
            "POST",
            "/invocations",
            {"decisions": [sample_decision(), sample_decision()]},
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    assert status == HTTPStatus.BAD_REQUEST
    assert payload == {"error": "decisions supports at most 1 items"}
    scorer.score_shared.assert_not_called()


def test_handler_rejects_invalid_shared_batch_limit():
    with pytest.raises(ValueError, match="max_shared_decisions"):
        server.make_handler(FakeScorer(), max_shared_decisions=0)


def test_discover_gguf_requires_one_unambiguous_file(tmp_path):
    with pytest.raises(RuntimeError, match="exactly one"):
        server.discover_gguf(tmp_path)

    first = tmp_path / "model.gguf"
    first.write_bytes(b"gguf")
    assert server.discover_gguf(tmp_path) == first

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.gguf").write_bytes(b"gguf")
    with pytest.raises(RuntimeError, match="found 2"):
        server.discover_gguf(tmp_path)


def test_explicit_gguf_must_exist(tmp_path):
    missing = tmp_path / "missing.gguf"

    with pytest.raises(RuntimeError, match="does not exist"):
        server.discover_gguf(tmp_path, str(missing))


def test_load_ggml_backends_requires_shared_library(tmp_path):
    with pytest.raises(RuntimeError, match="libggml.so does not exist"):
        server.load_ggml_backends(tmp_path)


def test_load_ggml_backends_registers_all_backends(tmp_path, monkeypatch):
    library_path = tmp_path / "libggml.so"
    library_path.touch()
    library = MagicMock()
    library.ggml_backend_reg_count.return_value = 2
    library.ggml_backend_reg_get.side_effect = [101, 202]
    library.ggml_backend_reg_name.side_effect = [b"CPU", b"CUDA"]
    cdll = MagicMock(return_value=library)
    monkeypatch.setattr(server.ctypes, "CDLL", cdll)

    names = server.load_ggml_backends(tmp_path)

    assert names == ["CPU", "CUDA"]
    cdll.assert_called_once_with(
        str(library_path),
        mode=server.ctypes.RTLD_GLOBAL,
    )
    library.ggml_backend_load_all_from_path.assert_called_once_with(
        str(tmp_path).encode("utf-8")
    )


def test_load_ggml_backends_rejects_empty_registry(tmp_path, monkeypatch):
    (tmp_path / "libggml.so").touch()
    library = MagicMock()
    library.ggml_backend_reg_count.return_value = 0
    monkeypatch.setattr(server.ctypes, "CDLL", MagicMock(return_value=library))

    with pytest.raises(RuntimeError, match="No llama.cpp backend loaded"):
        server.load_ggml_backends(tmp_path)


def test_gpu_offload_requires_cuda_backend():
    with pytest.raises(RuntimeError, match="CUDA"):
        server.require_gpu_backend(["CPU"], n_gpu_layers=-1)

    server.require_gpu_backend(["CPU", "CUDA"], n_gpu_layers=-1)
    server.require_gpu_backend(["CPU"], n_gpu_layers=0)


@pytest.mark.parametrize(("raw", "expected"), [(None, 0), ("0", 0), ("12", 12), ("-1", -1)])
def test_gpu_layer_environment_accepts_cpu_partial_and_full_offload(
    raw,
    expected,
    monkeypatch,
):
    if raw is None:
        monkeypatch.delenv("SEMIF_N_GPU_LAYERS", raising=False)
    else:
        monkeypatch.setenv("SEMIF_N_GPU_LAYERS", raw)

    assert server.gpu_layers_from_environment() == expected


@pytest.mark.parametrize("raw", ["-2", "all", "1.5"])
def test_gpu_layer_environment_rejects_invalid_values(raw, monkeypatch):
    monkeypatch.setenv("SEMIF_N_GPU_LAYERS", raw)

    with pytest.raises(RuntimeError, match="SEMIF_N_GPU_LAYERS"):
        server.gpu_layers_from_environment()


def test_model_parameter_override_preserves_cpu_initialization():
    original = MagicMock()
    params = MagicMock(n_gpu_layers=0)
    original.return_value = params

    override = server.model_params_with_gpu_layers(original, -1)
    returned = override("library")

    assert returned is params
    assert returned.n_gpu_layers == -1
    original.assert_called_once_with("library")
