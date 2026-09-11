#!/usr/bin/env python3
"""Client for the Qwen3.8 vLLM deployment (OpenAI-compatible API).

The model is served by the `qwen38` InferenceEndpointConfig on port 8000.
It is not exposed externally, so reach it via a port-forward first:

    kubectl port-forward pod/qwen38-7fbb468f4c-2cw44 8000:8000 &

Then:

    python qwen_client.py "Explain tensor parallelism in one paragraph."
    python qwen_client.py --stream "Write a haiku about GPUs."
    echo "What are you?" | python qwen_client.py -

Only depends on the standard library (urllib), so no pip install needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen3.8"


class QwenClient:
    """Minimal OpenAI-compatible chat client for the Qwen3.8 endpoint."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 api_key: str | None = None, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/v1/models", headers=self._headers(), method="GET"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float = 0.7, **kwargs) -> dict:
        """Non-streaming chat completion. Returns the parsed JSON response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data, headers=self._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def stream_chat(self, messages: list[dict], *, max_tokens: int = 1024,
                    temperature: float = 0.7, **kwargs):
        """Streaming chat completion. Yields (kind, text) chunks where kind is
        'reasoning' or 'content'."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data, headers=self._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                delta = json.loads(chunk)["choices"][0]["delta"]
                # vLLM emits reasoning separately when --reasoning-parser is set.
                if delta.get("reasoning_content"):
                    yield "reasoning", delta["reasoning_content"]
                if delta.get("content"):
                    yield "content", delta["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Invoke the Qwen3.8 vLLM endpoint.")
    parser.add_argument("prompt", help="Prompt text, or '-' to read from stdin.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Stream the response.")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Print the model's reasoning/chain-of-thought too.")
    args = parser.parse_args()

    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    client = QwenClient(base_url=args.base_url, model=args.model, api_key=args.api_key)

    try:
        if args.stream:
            in_reasoning = False
            for kind, text in client.stream_chat(
                messages, max_tokens=args.max_tokens, temperature=args.temperature
            ):
                if kind == "reasoning":
                    if not args.show_reasoning:
                        continue
                    if not in_reasoning:
                        sys.stderr.write("\n[reasoning]\n")
                        in_reasoning = True
                    sys.stderr.write(text)
                    sys.stderr.flush()
                else:
                    if in_reasoning:
                        sys.stdout.write("\n[answer]\n")
                        in_reasoning = False
                    sys.stdout.write(text)
                    sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            resp = client.chat(
                messages, max_tokens=args.max_tokens, temperature=args.temperature
            )
            msg = resp["choices"][0]["message"]
            if args.show_reasoning and msg.get("reasoning"):
                print("[reasoning]")
                print(msg["reasoning"])
                print("\n[answer]")
            print(msg["content"].strip())
            usage = resp.get("usage", {})
            if usage:
                sys.stderr.write(
                    f"\n[tokens] prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')} "
                    f"total={usage.get('total_tokens')}\n"
                )
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}\n")
        return 1
    except urllib.error.URLError as e:
        sys.stderr.write(
            f"Connection failed: {e.reason}\n"
            "Is the port-forward running? "
            "kubectl port-forward pod/qwen38-7fbb468f4c-2cw44 8000:8000 &\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
