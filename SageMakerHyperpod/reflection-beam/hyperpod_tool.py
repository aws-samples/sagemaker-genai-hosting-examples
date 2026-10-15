#!/usr/bin/env python3
"""Bounded lifecycle tooling for the Reflection Beam HyperPod sample."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


SAMPLE_ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
SAGEMAKER_ENDPOINT_RE = re.compile(
    r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$"
)
IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-fA-F]{64}$")
COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
PLACEHOLDER_PREFIXES = ("REPLACE_WITH_", "CHANGEME", "TODO")
FORBIDDEN_CONFIG_KEYS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "BEAM_API_KEY",
    "PASSWORD",
    "TOKEN",
}
SUCCESS_STATES = {"DeploymentComplete"}
FAILURE_STATES = {"DeploymentFailed", "DeletionFailed"}


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    manifest: str
    resource: str
    kind: str
    resource_name: str


RUNTIMES = {
    "jumpstart": RuntimeSpec(
        name="jumpstart",
        manifest="reflection-beam-jumpstart.yaml",
        resource="jumpstartmodels",
        kind="JumpStartModel",
        resource_name="reflection-beam-jumpstart",
    ),
    "sglang": RuntimeSpec(
        name="sglang",
        manifest="reflection-beam-sglang.yaml",
        resource="inferenceendpointconfigs",
        kind="InferenceEndpointConfig",
        resource_name="reflection-beam-sglang",
    ),
    "vllm": RuntimeSpec(
        name="vllm",
        manifest="reflection-beam-vllm.yaml",
        resource="inferenceendpointconfigs",
        kind="InferenceEndpointConfig",
        resource_name="reflection-beam-vllm",
    ),
}


class ToolError(RuntimeError):
    """Expected user-facing failure."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a non-executable KEY=VALUE file."""
    if not path.is_file():
        raise ToolError(f"configuration file not found: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise ToolError(f"{path}:{line_number}: use KEY=VALUE without 'export'")
        if "=" not in line:
            raise ToolError(f"{path}:{line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.fullmatch(key):
            raise ToolError(f"{path}:{line_number}: invalid key {key!r}")
        if key in FORBIDDEN_CONFIG_KEYS or key.endswith(("_PASSWORD", "_TOKEN")):
            raise ToolError(
                f"{path}:{line_number}: do not store credentials in deployment config"
            )

        raw_value = raw_value.strip()
        if not raw_value:
            value = ""
        else:
            try:
                parts = shlex.split(raw_value, comments=False, posix=True)
            except ValueError as error:
                raise ToolError(f"{path}:{line_number}: {error}") from error
            if len(parts) != 1:
                raise ToolError(
                    f"{path}:{line_number}: quote values containing whitespace"
                )
            value = parts[0]
        if "\n" in value or "\r" in value:
            raise ToolError(f"{path}:{line_number}: multiline values are not supported")
        values[key] = value
    return values


def load_config(paths: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_path in paths:
        values.update(parse_env_file(Path(raw_path).expanduser().resolve()))
    values.update(os.environ)
    return values


def is_placeholder(value: str) -> bool:
    upper = value.strip().upper()
    return not upper or upper.startswith(PLACEHOLDER_PREFIXES)


def yaml_double_quoted(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def validate_dns_label(name: str, value: str) -> None:
    if len(value) > 63 or not DNS_LABEL_RE.fullmatch(value):
        raise ToolError(
            f"{name} must be a lowercase Kubernetes DNS label of at most 63 characters"
        )


def validate_sagemaker_endpoint_name(value: str) -> None:
    if len(value) > 63 or not SAGEMAKER_ENDPOINT_RE.fullmatch(value):
        raise ToolError(
            "JUMPSTART_ENDPOINT_NAME must contain at most 63 letters, numbers, "
            "or internal hyphens"
        )


def validate_runtime_values(runtime: str, values: Mapping[str, str]) -> None:
    namespace = values["BEAM_NAMESPACE"]
    validate_dns_label("BEAM_NAMESPACE", namespace)

    if runtime == "jumpstart":
        validate_sagemaker_endpoint_name(values["JUMPSTART_ENDPOINT_NAME"])
        return

    prefix = runtime.upper()
    image = values[f"{prefix}_IMAGE"]
    if not IMAGE_DIGEST_RE.fullmatch(image):
        raise ToolError(
            f"{prefix}_IMAGE must be pinned as repository@sha256:<64-hex-digest>"
        )

    commit_sha = values[f"{prefix}_MODEL_COMMIT_SHA"]
    if not COMMIT_SHA_RE.fullmatch(commit_sha):
        raise ToolError(f"{prefix}_MODEL_COMMIT_SHA must be a full 40-character SHA")

    numeric_keys = [
        f"{prefix}_GPU_COUNT",
        f"{prefix}_TENSOR_PARALLEL_SIZE",
    ]
    if runtime == "sglang":
        numeric_keys.extend(
            [
                "SGLANG_PAGE_SIZE",
                "SGLANG_CONTEXT_LENGTH",
                "SGLANG_DFLASH_BLOCK_SIZE",
            ]
        )
    for key in numeric_keys:
        try:
            number = int(values[key])
        except ValueError as error:
            raise ToolError(f"{key} must be a positive integer") from error
        if number <= 0:
            raise ToolError(f"{key} must be a positive integer")

    if runtime == "sglang":
        try:
            memory_fraction = float(values["SGLANG_MEM_FRACTION_STATIC"])
        except ValueError as error:
            raise ToolError(
                "SGLANG_MEM_FRACTION_STATIC must be a number greater than 0 and at most 1"
            ) from error
        if not 0 < memory_fraction <= 1:
            raise ToolError(
                "SGLANG_MEM_FRACTION_STATIC must be a number greater than 0 and at most 1"
            )


def render_manifest(runtime: str, values: Mapping[str, str]) -> str:
    spec = RUNTIMES[runtime]
    template_path = SAMPLE_ROOT / spec.manifest
    template = template_path.read_text(encoding="utf-8")
    tokens = sorted(set(TOKEN_RE.findall(template)))

    missing = [
        token
        for token in tokens
        if token not in values or is_placeholder(values[token])
    ]
    if missing:
        names = ", ".join(missing)
        raise ToolError(
            f"missing configuration for {runtime}: {names}. "
            "Set real values in ignored local config files or the environment."
        )

    multiline = [
        token for token in tokens if "\n" in values[token] or "\r" in values[token]
    ]
    if multiline:
        raise ToolError(
            "configuration values must be single-line: " + ", ".join(multiline)
        )

    for match in TOKEN_RE.finditer(template):
        if match.start() == 0 or match.end() >= len(template):
            raise ToolError(f"{template_path.name}: template token must be quoted")
        if template[match.start() - 1] != '"' or template[match.end()] != '"':
            raise ToolError(
                f"{template_path.name}: ${{{match.group(1)}}} must occupy a "
                "double-quoted YAML scalar"
            )

    validate_runtime_values(runtime, values)
    rendered = TOKEN_RE.sub(
        lambda match: yaml_double_quoted(values[match.group(1)]),
        template,
    )
    leftovers = TOKEN_RE.findall(rendered)
    if leftovers or "REPLACE_WITH_" in rendered:
        raise ToolError(
            f"{template_path.name}: unresolved placeholders remain after rendering"
        )
    return rendered


def kubectl_prefix(args: argparse.Namespace) -> list[str]:
    command = [args.kubectl]
    if args.context:
        command.extend(["--context", args.context])
    if args.kubeconfig:
        command.extend(["--kubeconfig", args.kubeconfig])
    return command


def run_capture(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: int = 60,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as error:
        raise ToolError(f"command not found: {command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise ToolError(f"command timed out after {timeout}s: {command[0]}") from error

    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 2_000:
            detail = detail[-2_000:]
        raise ToolError(
            f"command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def get_resource(
    args: argparse.Namespace,
    spec: RuntimeSpec,
    namespace: str,
) -> dict[str, Any] | None:
    command = kubectl_prefix(args) + [
        "get",
        spec.resource,
        spec.resource_name,
        "--namespace",
        namespace,
        "--output",
        "json",
    ]
    result = run_capture(command, timeout=args.command_timeout_seconds, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "NotFound" in detail or "not found" in detail:
            return None
        raise ToolError(f"kubectl get failed: {detail or result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ToolError("kubectl returned invalid JSON") from error


def deployment_state(resource: Mapping[str, Any]) -> str:
    status = resource.get("status") or {}
    state = status.get("state")
    if state:
        return str(state)
    deployment = status.get("deploymentStatus") or {}
    return str(deployment.get("deploymentObjectOverallState") or "Unknown")


def endpoint_state(resource: Mapping[str, Any]) -> str:
    status = resource.get("status") or {}
    endpoints = status.get("endpoints") or {}
    sagemaker = endpoints.get("sagemaker") or {}
    return str(sagemaker.get("state") or "Unknown")


def resource_summary(
    spec: RuntimeSpec,
    namespace: str,
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    status = resource.get("status") or {}
    return {
        "runtime": spec.name,
        "resource": f"{spec.kind}/{spec.resource_name}",
        "namespace": namespace,
        "state": deployment_state(resource),
        "endpoint_state": endpoint_state(resource),
        "replicas": status.get("replicas"),
        "selector": status.get("selector"),
        "conditions": status.get("conditions") or [],
    }


def print_summary(summary: Mapping[str, Any], output: str = "text") -> None:
    if output == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"Resource:       {summary['resource']}")
    print(f"Namespace:      {summary['namespace']}")
    print(f"State:          {summary['state']}")
    print(f"Endpoint state: {summary['endpoint_state']}")
    print(f"Replicas:       {summary['replicas']}")
    selector = summary.get("selector")
    if selector:
        print(f"Selector:       {selector}")
    for condition in summary.get("conditions") or []:
        condition_type = condition.get("type", "Unknown")
        condition_status = condition.get("status", "Unknown")
        message = condition.get("message") or condition.get("reason") or ""
        print(f"Condition:      {condition_type}={condition_status} {message}".rstrip())


def wait_for_deployment(
    args: argparse.Namespace,
    spec: RuntimeSpec,
    namespace: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state = ""
    while True:
        resource = get_resource(args, spec, namespace)
        if resource is None:
            state = "NotFound"
        else:
            state = deployment_state(resource)
            if state in SUCCESS_STATES:
                return resource
            if state in FAILURE_STATES:
                raise ToolError(f"{spec.kind}/{spec.resource_name} entered {state}")
        if state != last_state:
            print(f"State: {state}", file=sys.stderr)
            last_state = state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolError(
                f"timed out after {timeout_seconds}s waiting for "
                f"{spec.kind}/{spec.resource_name}"
            )
        time.sleep(min(poll_seconds, remaining))


def wait_for_deletion(
    args: argparse.Namespace,
    spec: RuntimeSpec,
    namespace: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        resource = get_resource(args, spec, namespace)
        if resource is None:
            return
        state = deployment_state(resource)
        if state == "DeletionFailed":
            raise ToolError(f"{spec.kind}/{spec.resource_name} entered DeletionFailed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolError(
                f"timed out after {timeout_seconds}s deleting "
                f"{spec.kind}/{spec.resource_name}"
            )
        time.sleep(min(poll_seconds, remaining))


def show_diagnostics(
    args: argparse.Namespace,
    spec: RuntimeSpec,
    namespace: str,
) -> None:
    describe = run_capture(
        kubectl_prefix(args)
        + ["describe", spec.resource, spec.resource_name, "-n", namespace],
        timeout=args.command_timeout_seconds,
        check=False,
    )
    if describe.stdout:
        print("\nDeployment diagnostics:", file=sys.stderr)
        print(describe.stdout.rstrip(), file=sys.stderr)

    try:
        resource = get_resource(args, spec, namespace)
    except ToolError:
        return
    selector = ((resource or {}).get("status") or {}).get("selector")
    if not selector:
        return
    pods = run_capture(
        kubectl_prefix(args)
        + ["get", "pods", "-n", namespace, "-l", selector, "-o", "wide"],
        timeout=args.command_timeout_seconds,
        check=False,
    )
    if pods.stdout:
        print("\nPods:", file=sys.stderr)
        print(pods.stdout.rstrip(), file=sys.stderr)


def command_validate(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    rendered = render_manifest(args.runtime, values)
    if args.server_dry_run:
        result = run_capture(
            kubectl_prefix(args) + ["apply", "--dry-run=server", "--filename", "-"],
            input_text=rendered,
            timeout=args.command_timeout_seconds,
        )
        if result.stdout:
            print(result.stdout.rstrip())
    print(f"Valid: {RUNTIMES[args.runtime].manifest}")
    return 0


def command_deploy(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    spec = RUNTIMES[args.runtime]
    namespace = args.namespace or values.get("BEAM_NAMESPACE", "default")
    rendered = render_manifest(args.runtime, values)
    result = run_capture(
        kubectl_prefix(args) + ["apply", "--filename", "-"],
        input_text=rendered,
        timeout=args.command_timeout_seconds,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if args.no_wait:
        return 0
    try:
        resource = wait_for_deployment(
            args,
            spec,
            namespace,
            args.timeout_seconds,
            args.poll_seconds,
        )
    except ToolError:
        show_diagnostics(args, spec, namespace)
        raise
    print_summary(resource_summary(spec, namespace, resource), args.output)
    return 0


def command_status(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    spec = RUNTIMES[args.runtime]
    namespace = args.namespace or values.get("BEAM_NAMESPACE", "default")
    if args.wait:
        resource = wait_for_deployment(
            args,
            spec,
            namespace,
            args.timeout_seconds,
            args.poll_seconds,
        )
    else:
        resource = get_resource(args, spec, namespace)
        if resource is None:
            raise ToolError(
                f"{spec.kind}/{spec.resource_name} not found in namespace {namespace}"
            )
    print_summary(resource_summary(spec, namespace, resource), args.output)
    return 0


def validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("--base-url must be an http or https URL")
    if parsed.username or parsed.password:
        raise ToolError("do not place credentials in --base-url")
    return base_url.rstrip("/")


def invoke_custom(args: argparse.Namespace, base_url: str) -> int:
    command = [
        sys.executable,
        str(SAMPLE_ROOT / "beam_client.py"),
        "chat",
        "--base-url",
        validate_base_url(base_url),
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--timeout-seconds",
        str(args.request_timeout_seconds),
    ]
    result = run_capture(
        command,
        timeout=args.request_timeout_seconds + 15,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise ToolError(f"Beam invocation failed with exit code {result.returncode}")
    return 0


def invoke_jumpstart(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    request_file = Path(args.request_file).expanduser().resolve()
    if not request_file.is_file():
        raise ToolError(f"request file not found: {request_file}")
    endpoint_name = args.endpoint_name or values.get("JUMPSTART_ENDPOINT_NAME")
    if not endpoint_name or is_placeholder(endpoint_name):
        raise ToolError(
            "missing JumpStart endpoint name; set --endpoint-name or "
            "JUMPSTART_ENDPOINT_NAME"
        )
    validate_sagemaker_endpoint_name(endpoint_name)

    with tempfile.TemporaryDirectory(prefix="reflection-beam-response-") as temp_dir:
        response_file = Path(temp_dir) / "response.json"
        command = [args.aws_cli]
        if args.region:
            command.extend(["--region", args.region])
        command.extend(
            [
                "sagemaker-runtime",
                "invoke-endpoint",
                "--endpoint-name",
                endpoint_name,
                "--content-type",
                args.content_type,
                "--accept",
                args.accept,
                "--body",
                f"fileb://{request_file}",
                str(response_file),
            ]
        )
        result = run_capture(
            command,
            timeout=args.request_timeout_seconds,
        )
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if not response_file.is_file():
            raise ToolError("SageMaker Runtime did not produce a response body")
        print(response_file.read_text(encoding="utf-8"))
    return 0


def ready_pod_names(payload: Mapping[str, Any]) -> list[str]:
    ready: list[str] = []
    for pod in payload.get("items") or []:
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        if metadata.get("deletionTimestamp") or status.get("phase") != "Running":
            continue
        conditions = status.get("conditions") or []
        is_ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in conditions
        )
        name = metadata.get("name")
        if is_ready and name:
            ready.append(str(name))
    return sorted(ready)


def select_ready_pod(
    args: argparse.Namespace,
    namespace: str,
    selector: str,
) -> str:
    result = run_capture(
        kubectl_prefix(args)
        + [
            "get",
            "pods",
            "--namespace",
            namespace,
            "--selector",
            selector,
            "--output",
            "json",
        ],
        timeout=args.command_timeout_seconds,
    )
    try:
        names = ready_pod_names(json.loads(result.stdout))
    except json.JSONDecodeError as error:
        raise ToolError("kubectl returned invalid pod JSON") from error
    if not names:
        raise ToolError(f"no ready worker pod matched selector {selector!r}")
    return names[0]


def available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_port(
    process: subprocess.Popen[str],
    port: int,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise ToolError(
                "kubectl port-forward exited before it became ready"
                + (f": {output.strip()}" if output.strip() else "")
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise ToolError(f"kubectl port-forward was not ready after {timeout_seconds}s")


def check_health(base_url: str, timeout_seconds: int) -> None:
    parsed = urlsplit(validate_base_url(base_url))
    health_url = f"{parsed.scheme}://{parsed.netloc}/health"
    request = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if not 200 <= response.status < 300:
                raise ToolError(f"health check returned HTTP {response.status}")
    except urllib.error.URLError as error:
        raise ToolError(f"health check failed: {error.reason}") from error


def command_invoke(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    if args.runtime == "jumpstart":
        return invoke_jumpstart(args, values)
    return invoke_custom(args, args.base_url)


def command_smoke(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    spec = RUNTIMES[args.runtime]
    namespace = args.namespace or values.get("BEAM_NAMESPACE", "default")
    resource = get_resource(args, spec, namespace)
    if resource is None:
        raise ToolError(
            f"{spec.kind}/{spec.resource_name} not found in namespace {namespace}"
        )
    state = deployment_state(resource)
    if state not in SUCCESS_STATES:
        raise ToolError(
            f"{spec.kind}/{spec.resource_name} is not ready; current state is {state}"
        )

    if args.runtime == "jumpstart":
        return invoke_jumpstart(args, values)

    if args.base_url:
        base_url = validate_base_url(args.base_url)
        check_health(base_url, args.health_timeout_seconds)
        print("Health check: passed", file=sys.stderr)
        return invoke_custom(args, base_url)

    selector = ((resource.get("status") or {}).get("selector") or "").strip()
    if not selector:
        raise ToolError("deployment status does not contain a worker pod selector")
    pod_name = select_ready_pod(args, namespace, selector)
    local_port = available_local_port()
    command = kubectl_prefix(args) + [
        "port-forward",
        "--namespace",
        namespace,
        f"pod/{pod_name}",
        f"{local_port}:{args.remote_port}",
    ]
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as error:
        raise ToolError(f"command not found: {command[0]}") from error

    try:
        wait_for_port(process, local_port, args.port_forward_timeout_seconds)
        base_url = f"http://127.0.0.1:{local_port}/v1"
        check_health(base_url, args.health_timeout_seconds)
        print(f"Health check: passed via pod/{pod_name}", file=sys.stderr)
        return invoke_custom(args, base_url)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def command_cleanup(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    spec = RUNTIMES[args.runtime]
    namespace = args.namespace or values.get("BEAM_NAMESPACE", "default")
    if args.confirm_name != spec.resource_name:
        raise ToolError(
            f"cleanup requires --confirm-name {spec.resource_name} for this runtime"
        )
    resource = get_resource(args, spec, namespace)
    if resource is None:
        print(f"Already absent: {spec.kind}/{spec.resource_name}")
        return 0
    result = run_capture(
        kubectl_prefix(args)
        + [
            "delete",
            spec.resource,
            spec.resource_name,
            "--namespace",
            namespace,
            "--wait=false",
        ],
        timeout=args.command_timeout_seconds,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    wait_for_deletion(
        args,
        spec,
        namespace,
        args.timeout_seconds,
        args.poll_seconds,
    )
    print(f"Deleted: {spec.kind}/{spec.resource_name}")
    return 0


def add_runtime_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("runtime", choices=sorted(RUNTIMES))


def add_wait_arguments(parser: argparse.ArgumentParser, default_timeout: int) -> None:
    parser.add_argument(
        "--timeout-seconds",
        type=positive_int,
        default=default_timeout,
    )
    parser.add_argument("--poll-seconds", type=positive_int, default=10)


def add_invocation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BEAM_API_BASE"),
        help="OpenAI-compatible base URL for SGLang or vLLM.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("BEAM_MODEL", "reflection-beam")
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: reflection-beam-ready",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--request-timeout-seconds", type=positive_int, default=300)
    parser.add_argument(
        "--request-file",
        help="Release-compatible JSON request body for JumpStart.",
    )
    parser.add_argument("--endpoint-name")
    parser.add_argument("--region")
    parser.add_argument("--content-type", default="application/json")
    parser.add_argument("--accept", default="application/json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="Non-executable KEY=VALUE file. Repeat to merge files.",
    )
    parser.add_argument("--namespace")
    parser.add_argument("--context", help="Explicit kubectl context.")
    parser.add_argument("--kubeconfig")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument(
        "--command-timeout-seconds",
        type=positive_int,
        default=60,
    )

    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate and render one manifest.")
    add_runtime_argument(validate)
    validate.add_argument("--server-dry-run", action="store_true")

    deploy = commands.add_parser("deploy", help="Apply one manifest and wait.")
    add_runtime_argument(deploy)
    add_wait_arguments(deploy, 1_800)
    deploy.add_argument("--no-wait", action="store_true")
    deploy.add_argument("--output", choices=("text", "json"), default="text")

    status = commands.add_parser("status", help="Read bounded deployment status.")
    add_runtime_argument(status)
    add_wait_arguments(status, 1_800)
    status.add_argument("--wait", action="store_true")
    status.add_argument("--output", choices=("text", "json"), default="text")

    invoke = commands.add_parser("invoke", help="Invoke a configured endpoint.")
    add_runtime_argument(invoke)
    add_invocation_arguments(invoke)

    smoke = commands.add_parser(
        "smoke", help="Check health and run a short invocation."
    )
    add_runtime_argument(smoke)
    add_invocation_arguments(smoke)
    smoke.add_argument("--remote-port", type=positive_int, default=8_000)
    smoke.add_argument(
        "--port-forward-timeout-seconds",
        type=positive_int,
        default=30,
    )
    smoke.add_argument("--health-timeout-seconds", type=positive_int, default=15)

    cleanup = commands.add_parser("cleanup", help="Delete one exact sample resource.")
    add_runtime_argument(cleanup)
    add_wait_arguments(cleanup, 600)
    cleanup.add_argument("--confirm-name", required=True)

    return parser


def validate_invocation_args(args: argparse.Namespace) -> None:
    if args.command not in {"invoke", "smoke"}:
        return
    if args.runtime == "jumpstart":
        if not args.request_file:
            raise ToolError(
                "JumpStart invocation requires --request-file because the request "
                "schema is release-specific"
            )
    elif args.command == "invoke" and not args.base_url:
        raise ToolError("custom-runtime invocation requires --base-url")
    if not 0.0 <= args.temperature <= 2.0:
        raise ToolError("--temperature must be between 0 and 2")
    if not 0.0 <= args.top_p <= 1.0:
        raise ToolError("--top-p must be between 0 and 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        values = load_config(args.config)
        if args.namespace:
            values["BEAM_NAMESPACE"] = args.namespace
        validate_invocation_args(args)
        handlers = {
            "validate": command_validate,
            "deploy": command_deploy,
            "status": command_status,
            "invoke": command_invoke,
            "smoke": command_smoke,
            "cleanup": command_cleanup,
        }
        return handlers[args.command](args, values)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
