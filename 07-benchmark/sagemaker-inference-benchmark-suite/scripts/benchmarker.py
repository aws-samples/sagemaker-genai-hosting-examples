"""
Benchmark engine for SageMaker inference endpoints.

Replaces benchmark_matrix.py with config-driven design.
Supports both non-streaming and streaming (TTFT) invocation modes.

Usage:
    python -m scripts.benchmarker recipes/qwen3-32b-g7e-eagle3.yaml --endpoint NAME
    python -m scripts.benchmarker recipes/recipe.yaml --endpoint NAME --ic IC_NAME --streaming
"""

import concurrent.futures
import csv
import json
import os
import statistics
import time
from datetime import datetime
from typing import Optional

import boto3
from botocore.config import Config

from scripts.config_loader import (
    BenchmarkConfig,
    InferenceParams,
    get_optimization_label,
    load_config,
    print_config_summary,
)
from scripts.prompts import USE_CASE_DESCRIPTIONS, USE_CASE_PROMPTS

DEFAULT_RESULTS_DIR = "results/matrix"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(config: BenchmarkConfig, endpoint_name: str,
                  ic_name: str = None, results_dir: str = None) -> list[dict]:
    """Run the full benchmark suite: all use cases x all concurrency levels.

    Returns a list of stats dicts, one per (use_case, concurrency) pair.
    Results are written incrementally to CSV files.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    region = config.deployment.endpoint.region
    client = boto3.client(
        "sagemaker-runtime", region_name=region,
        config=Config(read_timeout=300, retries={"max_attempts": 0}),
    )

    bp = config.benchmark
    params = bp.inference_params
    streaming = bp.streaming

    model_short = config.deployment.model.short_name or config.deployment.model.id.split("/")[-1][:15]
    opt_label = get_optimization_label(config)
    inst_short = config.deployment.instance.type.replace("ml.", "").replace(".", "").replace("xlarge", "xl")
    region_tag = region.replace("-", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    for use_case in bp.use_cases:
        if use_case not in USE_CASE_PROMPTS:
            print(f"WARNING: Unknown use case '{use_case}', skipping.")
            continue

        csv_file = os.path.join(
            results_dir,
            f"{model_short}_{opt_label}_{inst_short}_{use_case}_{region_tag}_{timestamp}.csv",
        )

        print(f"\n{'=' * 70}")
        print(f"Benchmark: {endpoint_name}" + (f" (IC: {ic_name})" if ic_name else ""))
        print(f"  Model: {config.deployment.model.id}")
        print(f"  Optimization: {opt_label} | Instance: {config.deployment.instance.type}")
        print(f"  Use case: {USE_CASE_DESCRIPTIONS.get(use_case, use_case)}")
        print(f"  Concurrency: {bp.concurrency_levels}")
        print(f"  Requests/level: {bp.requests_per_level} | Streaming: {streaming}")
        print(f"  Output: {csv_file}")
        print(f"{'=' * 70}\n")

        # Warmup (first use case only gets printed, but all get warmed)
        warmup(client, endpoint_name, use_case, params, ic_name,
               num_requests=bp.warmup_requests, streaming=streaming)

        header_written = False

        for conc in bp.concurrency_levels:
            try:
                stats = _run_concurrency_level(
                    config, client, endpoint_name, use_case, conc,
                    bp.requests_per_level, ic_name,
                )
                # Add metadata
                stats["model"] = model_short
                stats["optimization"] = opt_label
                stats["instance_type"] = config.deployment.instance.type
                stats["region"] = region
                stats["timestamp"] = timestamp
                all_results.append(stats)

                # Incremental CSV write
                if "error" not in stats or stats.get("successful", 0) > 0:
                    _write_csv_row(csv_file, stats, write_header=not header_written)
                    header_written = True
            except Exception as e:
                print(f"    ERROR at C={conc}: {e}")
                all_results.append({"concurrency": conc, "error": str(e)})

            if conc != bp.concurrency_levels[-1]:
                time.sleep(bp.pause_between_levels_sec)

        print(f"\nResults saved to {csv_file}")

    return all_results


def warmup(client, endpoint_name: str, use_case: str, params: InferenceParams,
           ic_name: str = None, num_requests: int = 3, streaming: bool = False):
    """Send warmup requests before measurement begins."""
    prompts = USE_CASE_PROMPTS[use_case]
    invoke_fn = _invoke_streaming if streaming else _invoke_non_streaming
    print(f"Warming up ({num_requests} requests)...")
    for i in range(num_requests):
        prompt_data = prompts[i % len(prompts)]
        invoke_fn(client, endpoint_name, prompt_data["messages"], params, ic_name)
    print("Warmup complete.\n")


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------

def _invoke_non_streaming(client, endpoint_name: str, messages: list,
                          params: InferenceParams, ic_name: str = None) -> dict:
    """Invoke endpoint synchronously (non-streaming). Returns result dict."""
    payload = {
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
    }
    if params.extra_payload:
        payload.update(params.extra_payload)

    invoke_kwargs = {
        "EndpointName": endpoint_name,
        "ContentType": "application/json",
        "Body": json.dumps(payload),
    }
    if ic_name:
        invoke_kwargs["InferenceComponentName"] = ic_name

    start = time.perf_counter()
    try:
        response = client.invoke_endpoint(**invoke_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_body = response["Body"].read()

        # Parse response
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as parse_err:
            return _error_result(
                elapsed_ms,
                f"ResponseParseError: {type(parse_err).__name__}",
                raw_body[:500].decode("utf-8", errors="replace") if isinstance(raw_body, bytes) else str(raw_body)[:500],
            )

        # Check for error in response body
        if "error" in body and "choices" not in body:
            err_msg = body.get("error", {})
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", str(err_msg))
            return _error_result(elapsed_ms, f"ModelError: {str(err_msg)[:200]}", json.dumps(body)[:500])

        usage = body.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        # Handle content: null from reasoning models (GPT-OSS)
        output_text = body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        return {
            "success": True,
            "latency_ms": elapsed_ms,
            "ttft_ms": None,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "tok_per_sec": output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 and output_tokens > 0 else 0,
            "output_text": output_text[:200],
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        status_code = ""
        if hasattr(e, "response"):
            status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", "")
        return _error_result(
            elapsed_ms,
            f"{type(e).__name__}: {str(e)[:200]}",
            f"HTTP {status_code}" if status_code else "",
        )


def _invoke_streaming(client, endpoint_name: str, messages: list,
                      params: InferenceParams, ic_name: str = None) -> dict:
    """Invoke endpoint with streaming for TTFT measurement. Returns result dict."""
    payload = {
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "stream": True,
    }
    if params.extra_payload:
        payload.update(params.extra_payload)

    invoke_kwargs = {
        "EndpointName": endpoint_name,
        "ContentType": "application/json",
        "Body": json.dumps(payload),
    }
    if ic_name:
        invoke_kwargs["InferenceComponentName"] = ic_name

    start = time.perf_counter()
    try:
        response = client.invoke_endpoint_with_response_stream(**invoke_kwargs)
        event_stream = response["Body"]

        first_token_time = None
        full_text = ""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        for event in event_stream:
            chunk = event.get("PayloadPart", {}).get("Bytes", b"")
            if not chunk:
                continue

            for line in chunk.decode("utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Handle both SSE format (data: {...}) and raw JSON ({...})
                if line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    json_str = line[6:]
                elif line.startswith("{"):
                    json_str = line
                else:
                    continue

                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                # Extract delta content
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text += content

                # Check for usage in final chunk
                usage = data.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)

        elapsed_ms = (time.perf_counter() - start) * 1000
        ttft_ms = (first_token_time - start) * 1000 if first_token_time else None

        # Estimate output tokens from text if usage not available
        if not output_tokens and full_text:
            output_tokens = max(1, len(full_text) // 4)  # rough estimate

        return {
            "success": True,
            "latency_ms": elapsed_ms,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tok_per_sec": output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 and output_tokens > 0 else 0,
            "output_text": (full_text or "")[:200],
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        status_code = ""
        if hasattr(e, "response"):
            status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", "")
        return _error_result(
            elapsed_ms,
            f"{type(e).__name__}: {str(e)[:200]}",
            f"HTTP {status_code}" if status_code else "",
        )


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _run_concurrency_level(config: BenchmarkConfig, client, endpoint_name: str,
                           use_case: str, concurrency: int, num_requests: int,
                           ic_name: str = None) -> dict:
    """Run benchmark at a single concurrency level. Returns stats dict."""
    prompts = USE_CASE_PROMPTS[use_case]
    params = config.benchmark.inference_params
    streaming = config.benchmark.streaming
    invoke_fn = _invoke_streaming if streaming else _invoke_non_streaming

    print(f"  C={concurrency}: sending {num_requests} requests ({use_case})...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            prompt_data = prompts[i % len(prompts)]
            futures.append(executor.submit(
                invoke_fn, client, endpoint_name,
                prompt_data["messages"], params, ic_name,
            ))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    # Log error breakdown
    if failed:
        error_counts = {}
        for f_result in failed:
            err = f_result.get("error", "unknown")
            err_key = err.split(":")[0] if ":" in err else err[:80]
            error_counts[err_key] = error_counts.get(err_key, 0) + 1
        print(f"    ERRORS ({len(failed)}/{len(results)} failed):")
        for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            sample = next((f_r for f_r in failed if f_r.get("error", "").startswith(err_type)), {})
            detail = sample.get("error_detail", "")[:100]
            print(f"      {count}x {err_type}")
            if detail:
                print(f"         detail: {detail}")

    if not successful:
        return {
            "endpoint": endpoint_name,
            "use_case": use_case,
            "concurrency": concurrency,
            "error": "All requests failed",
            "successful": 0,
            "failed": len(failed),
        }

    # Compute statistics
    latencies = [r["latency_ms"] for r in successful]
    tok_rates = [r["tok_per_sec"] for r in successful if r["tok_per_sec"] > 0]
    non_empty = [r for r in successful if len(r.get("output_text", "")) > 10]
    output_validation_rate = len(non_empty) / len(successful)

    stats = {
        "endpoint": endpoint_name,
        "use_case": use_case,
        "concurrency": concurrency,
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": len(failed),
        "output_validation_rate": round(output_validation_rate, 2),
        "latency_p50": round(statistics.median(latencies), 1),
        "latency_p90": round(sorted(latencies)[int(len(latencies) * 0.9)], 1),
        "latency_p99": round(sorted(latencies)[min(int(len(latencies) * 0.99), len(latencies) - 1)], 1),
        "latency_avg": round(statistics.mean(latencies), 1),
        "latency_min": round(min(latencies), 1),
        "latency_max": round(max(latencies), 1),
        "tok_per_sec_avg": round(statistics.mean(tok_rates), 1) if tok_rates else 0,
        "tok_per_sec_p50": round(statistics.median(tok_rates), 1) if tok_rates else 0,
        "avg_input_tokens": round(statistics.mean([r["input_tokens"] for r in successful]), 1),
        "avg_output_tokens": round(statistics.mean([r["output_tokens"] for r in successful]), 1),
    }

    # Aggregate throughput
    avg_lat_s = stats["latency_avg"] / 1000
    if avg_lat_s > 0:
        stats["rps"] = round(concurrency / avg_lat_s, 2)
        stats["aggregate_output_tok_sec"] = round(stats["rps"] * stats["avg_output_tokens"], 1)
    else:
        stats["rps"] = 0
        stats["aggregate_output_tok_sec"] = 0

    # TTFT stats (streaming only)
    if streaming:
        ttft_values = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
        if ttft_values:
            stats["ttft_p50"] = round(statistics.median(ttft_values), 1)
            stats["ttft_p90"] = round(sorted(ttft_values)[int(len(ttft_values) * 0.9)], 1)
            stats["ttft_avg"] = round(statistics.mean(ttft_values), 1)
        else:
            stats["ttft_p50"] = None
            stats["ttft_p90"] = None
            stats["ttft_avg"] = None

    # Print summary
    summary = (
        f"    p50={stats['latency_p50']:.0f}ms | tok/s={stats['tok_per_sec_avg']:.1f} | "
        f"rps={stats['rps']:.2f} | agg={stats['aggregate_output_tok_sec']:.0f} tok/s | "
        f"valid={stats['output_validation_rate']:.0%}"
    )
    if streaming and stats.get("ttft_p50") is not None:
        summary += f" | ttft_p50={stats['ttft_p50']:.0f}ms"
    print(summary)

    return stats


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv_row(filepath: str, stats: dict, write_header: bool = False):
    """Write a single stats row to CSV, optionally with header."""
    mode = "w" if write_header else "a"
    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(stats)


def _error_result(elapsed_ms: float, error: str, error_detail: str = "") -> dict:
    """Build a failure result dict."""
    return {
        "success": False,
        "latency_ms": elapsed_ms,
        "ttft_ms": None,
        "error": error,
        "error_detail": error_detail,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tok_per_sec": 0,
        "output_text": "",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SageMaker Inference Benchmarker")
    parser.add_argument("config", help="YAML config path")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name")
    parser.add_argument("--ic", default=None, help="Inference Component name")
    parser.add_argument("--use-case", default=None,
                        help="Override use cases (comma-separated, or 'all')")
    parser.add_argument("--concurrency", default=None,
                        help="Override concurrency levels (comma-separated)")
    parser.add_argument("--requests", type=int, default=None,
                        help="Override requests per level")
    parser.add_argument("--streaming", action="store_true", default=None,
                        help="Enable streaming (TTFT measurement)")
    parser.add_argument("--results-dir", default=None,
                        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.use_case:
        if args.use_case == "all":
            cfg.benchmark.use_cases = list(USE_CASE_PROMPTS.keys())
        else:
            cfg.benchmark.use_cases = [uc.strip() for uc in args.use_case.split(",")]
    if args.concurrency:
        cfg.benchmark.concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    if args.requests is not None:
        cfg.benchmark.requests_per_level = args.requests
    if args.streaming is not None:
        cfg.benchmark.streaming = args.streaming

    run_benchmark(cfg, args.endpoint, ic_name=args.ic, results_dir=args.results_dir)
