#!/usr/bin/env python3
"""Invoke Reflection Beam through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

try:
    from openai import OpenAI
except ImportError:  # Let validation and unit tests run before dependencies install.
    OpenAI = None  # type: ignore[assignment,misc]

DEFAULT_MODEL = "reflection-beam"
MAX_FILE_BYTES = 200_000
ALLOWED_SUFFIXES = {
    ".json",
    ".js",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".yaml",
    ".yml",
}


def validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--base-url must be an http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("Do not place credentials in --base-url")
    return base_url.rstrip("/")


def build_client(base_url: str, timeout_seconds: int) -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Run: "
            "python -m pip install -r requirements.txt"
        )
    validated_url = validate_base_url(base_url)
    api_key = os.environ.get("BEAM_API_KEY", "EMPTY")
    parsed = urlsplit(validated_url)
    if (
        api_key != "EMPTY"
        and parsed.scheme != "https"
        and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Authenticated endpoints must use HTTPS")
    return OpenAI(
        base_url=validated_url,
        api_key=api_key,
        timeout=float(timeout_seconds),
    )


def chat(args: argparse.Namespace) -> int:
    client = build_client(args.base_url, args.timeout_seconds)
    started = time.perf_counter()
    first_token_at: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        for choice in chunk.choices:
            text = choice.delta.content or ""
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                print(text, end="", flush=True)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    completed = time.perf_counter()
    telemetry = {
        "time_to_first_token_seconds": (
            round(first_token_at - started, 3) if first_token_at else None
        ),
        "end_to_end_seconds": round(completed - started, 3),
        "finish_reason": finish_reason,
        "usage": usage,
    }
    print("\n", file=sys.stdout)
    print(json.dumps(telemetry, indent=2), file=sys.stderr)
    return 0


def resolve_workspace_file(workspace_root: Path, relative_path: str) -> Path:
    workspace_root = workspace_root.resolve()
    if not workspace_root.is_dir():
        raise FileNotFoundError(f"Workspace root not found: {workspace_root}")
    candidate = (workspace_root / relative_path).resolve()
    if candidate != workspace_root and workspace_root not in candidate.parents:
        raise ValueError("Requested path escapes the workspace root")
    if not candidate.is_file():
        raise FileNotFoundError(relative_path)
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {candidate.suffix}")
    if candidate.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes")
    return candidate


def review(args: argparse.Namespace) -> int:
    client = build_client(args.base_url, args.timeout_seconds)
    workspace_root = Path(args.workspace_root).resolve()
    target = resolve_workspace_file(workspace_root, args.target)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_workspace_file",
                "description": "Read one text file inside the approved workspace root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "Repository-relative path to read.",
                        }
                    },
                    "required": ["relative_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Review code using only the supplied read-only tool. Return a minimal "
                "unified diff for human review. Never claim that you changed or "
                "applied the patch."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Review {target.relative_to(workspace_root)} for an off-by-one error. "
                "Use the file-reading tool before proposing a patch."
            ),
        },
    ]

    first_response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls or []
    if len(tool_calls) != 1:
        raise RuntimeError("The model must return exactly one file-reading tool call")

    messages.append(
        {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
        }
    )

    for tool_call in tool_calls:
        if tool_call.function.name != "read_workspace_file":
            raise ValueError(f"Unexpected tool: {tool_call.function.name}")
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as error:
            raise ValueError("Tool arguments were not valid JSON") from error
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"relative_path"}
            or not isinstance(arguments["relative_path"], str)
        ):
            raise ValueError("Tool arguments must contain only relative_path")
        requested_file = resolve_workspace_file(
            workspace_root,
            arguments["relative_path"],
        )
        if requested_file != target:
            raise ValueError(
                "The model requested a file other than the approved target"
            )
        result = {
            "path": str(requested_file.relative_to(workspace_root)),
            "content": requested_file.read_text(encoding="utf-8"),
        }
        print(
            "Tool call:",
            tool_call.function.name,
            json.dumps(arguments, separators=(",", ":")),
            file=sys.stderr,
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    started = time.perf_counter()
    first_token_at: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    stream = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        for choice in chunk.choices:
            text = choice.delta.content or ""
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                print(text, end="", flush=True)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    completed = time.perf_counter()
    telemetry = {
        "time_to_first_token_seconds": (
            round(first_token_at - started, 3) if first_token_at else None
        ),
        "end_to_end_seconds": round(completed - started, 3),
        "finish_reason": finish_reason,
        "usage": usage,
        "reasoning_field_present": bool(
            getattr(response_message, "reasoning_content", None)
        ),
        "patch_applied": False,
    }
    print("\n", file=sys.stdout)
    print(json.dumps(telemetry, indent=2), file=sys.stderr)
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke Reflection Beam through an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "mode",
        choices=("chat", "review"),
        help="Run a streaming chat or guarded code-review workflow.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BEAM_API_BASE", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BEAM_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--prompt", default="Introduce Reflection Beam in one sentence."
    )
    parser.add_argument("--workspace-root", default="example")
    parser.add_argument("--target", default="pagination.py")
    parser.add_argument("--max-tokens", type=positive_int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=positive_int, default=300)
    args = parser.parse_args(argv)
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "chat":
            return chat(args)
        return review(args)
    except (
        FileNotFoundError,
        OSError,
        UnicodeError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # OpenAI client exceptions vary by version.
        print(f"request failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
