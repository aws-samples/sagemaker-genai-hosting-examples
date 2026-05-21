"""
KV Cache effectiveness benchmark — same prefix vs different prefix.

Adapted from daekeun-ml/aws-ai-infra-helper benchmark pattern.
Measures how prefix caching and intelligent routing improve performance
when requests share the same prefix vs completely different prefixes.

Usage:
    python -m scripts.benchmark_kvcache recipes/qwen3-32b-g7e-prefix-cache.yaml \
        --endpoint NAME --concurrent 20 --context-tokens 4000
"""

import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import boto3
from botocore.config import Config

from scripts.config_loader import (
    BenchmarkConfig,
    get_optimization_label,
    load_config,
)

# ~4K token shared context (Korean AI industry report, from daekeun's benchmark)
SHARED_CONTEXT = """
# 2024 Global AI Industry Report

## 1. Market Overview
The global AI market grew to $500 billion in 2024, recording a 35% annual growth rate.

### 1.1 Regional Analysis
North America holds 45% market share. The US leads AI R&D with big tech companies like OpenAI, Google, Microsoft, and Amazon. The Silicon Valley startup ecosystem is active, with AI startup investments exceeding $25 billion in H1 2024.

Europe holds 25% market share, with GDPR-based strong regulatory frameworks leading responsible AI development. The EU AI Act implementation has formalized risk-based AI regulation, becoming a global standard for AI governance.

Asia-Pacific holds 30% market share with the fastest growth. China focuses on domestic LLM development and AI chip manufacturing for technological independence. Korea integrates AI into semiconductors and consumer electronics through Samsung and LG.

### 1.2 Industry Applications
Finance uses AI for fraud detection, credit scoring, algorithmic trading, and customer service automation. Major financial institutions like JP Morgan and Goldman Sachs are expanding their AI research teams.

Healthcare is seeing AI-based diagnostic support systems deployed in clinical settings. AI accuracy in image diagnostics, pathology analysis, drug discovery, and patient monitoring is reaching human expert levels.

Manufacturing uses AI for quality control, predictive maintenance, supply chain optimization, and robotic automation to implement smart factories.

Retail has made AI essential for personalized recommendations, inventory management, demand forecasting, and customer service chatbots.

## 2. Technology Trends

### 2.1 Generative AI
2024 marks the year generative AI entered the practical deployment stage. LLM performance improved significantly with GPT-4, Claude 3, and Gemini Ultra, enabling text, image, video, audio, and code generation.

Companies are building specialized AI solutions by fine-tuning general LLMs with proprietary data or leveraging RAG techniques.

### 2.2 Multimodal AI
Multimodal AI that integrates text, image, audio, and video processing is gaining attention. GPT-4V and Gemini can understand and describe images, showing innovative results in real-time translation, content generation, and medical diagnostics.

### 2.3 Edge AI
The trend of running AI models on edge devices is spreading to reduce cloud dependency and enhance real-time processing and privacy.

## 3. Key Companies

OpenAI released GPT-4 Turbo and expanded ChatGPT Enterprise for the enterprise market. Google announced the Gemini model family. Microsoft expanded Copilot across all product lines. Amazon provides various foundation models through Bedrock and launched Q assistant. Meta contributed to AI democratization by open-sourcing LLaMA 3.

## 4. Future Outlook
AGI research will accelerate. AI agent practicalization will expand, automating complex tasks and increasing human-AI collaboration use cases. The convergence of AI with quantum computing, biotechnology, and robotics will create new industry paradigms.
""" * 2  # ~4K tokens


@dataclass
class KVCacheResult:
    test_type: str  # "same_prefix" or "different_prefix"
    concurrent_requests: int
    total_requests: int
    successful: int
    failed: int
    latency_p50: float
    latency_p90: float
    latency_p99: float
    latency_avg: float
    ttft_p50: Optional[float]
    ttft_p90: Optional[float]
    ttft_avg: Optional[float]
    throughput_tps: float
    total_tokens: int
    duration_sec: float


def run_kvcache_benchmark(
    config: BenchmarkConfig,
    endpoint_name: str,
    ic_name: str = None,
    concurrent_requests: int = 20,
    context_tokens: int = 4000,
) -> dict:
    """Run KV cache effectiveness benchmark: same prefix vs different prefix.

    Returns dict with 'same_prefix' and 'different_prefix' KVCacheResult,
    plus 'comparison' with improvement percentages.
    """
    region = config.deployment.endpoint.region
    client = boto3.client(
        "sagemaker-runtime", region_name=region,
        config=Config(read_timeout=300, retries={"max_attempts": 0}),
    )

    params = config.benchmark.inference_params

    print(f"\n{'=' * 70}")
    print(f"KV Cache Benchmark: {endpoint_name}")
    print(f"  Concurrent requests: {concurrent_requests}")
    print(f"  Context: ~{context_tokens} tokens (shared prefix)")
    print(f"  Optimization: {get_optimization_label(config)}")
    print(f"{'=' * 70}\n")

    # Warmup
    print("Warming up (3 requests)...")
    for i in range(3):
        _invoke_single(client, endpoint_name, SHARED_CONTEXT,
                       "Summarize the key findings.", params, ic_name)
    print("Warmup complete.\n")

    # Test 1: Same prefix (all requests share identical context → cache hits)
    print(f"Test 1: Same Prefix ({concurrent_requests} concurrent requests)...")
    same_results, same_duration = _run_concurrent(
        client, endpoint_name, concurrent_requests,
        use_same_context=True, params=params, ic_name=ic_name,
    )
    same = _analyze(same_results, same_duration, "same_prefix", concurrent_requests)

    time.sleep(3)

    # Test 2: Different prefix (each request has unique context → cache misses)
    print(f"\nTest 2: Different Prefix ({concurrent_requests} concurrent requests)...")
    diff_results, diff_duration = _run_concurrent(
        client, endpoint_name, concurrent_requests,
        use_same_context=False, params=params, ic_name=ic_name,
    )
    diff = _analyze(diff_results, diff_duration, "different_prefix", concurrent_requests)

    # Compare
    comparison = _compare(same, diff)
    _print_comparison(same, diff, comparison)

    return {"same_prefix": same, "different_prefix": diff, "comparison": comparison}


def _invoke_single(client, endpoint_name, context, question, params, ic_name=None):
    """Invoke endpoint with context + question."""
    payload = {
        "messages": [
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ],
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
        # Try streaming for TTFT
        payload["stream"] = True
        invoke_kwargs["Body"] = json.dumps(payload)
        response = client.invoke_endpoint_with_response_stream(**invoke_kwargs)

        ttft = None
        total_tokens = 0
        completion_tokens = 0

        for event in response["Body"]:
            chunk = event.get("PayloadPart", {}).get("Bytes", b"")
            if not chunk:
                continue
            if ttft is None:
                ttft = time.perf_counter() - start

            for line in chunk.decode("utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    data = json.loads(line[6:])
                    usage = data.get("usage")
                    if usage:
                        total_tokens = usage.get("total_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                except json.JSONDecodeError:
                    pass

        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "latency": elapsed,
            "ttft": ttft,
            "total_tokens": total_tokens,
            "completion_tokens": completion_tokens,
        }
    except Exception:
        # Fallback to non-streaming
        payload.pop("stream", None)
        invoke_kwargs["Body"] = json.dumps(payload)
        try:
            response = client.invoke_endpoint(**invoke_kwargs)
            elapsed = time.perf_counter() - start
            body = json.loads(response["Body"].read())
            usage = body.get("usage", {})
            return {
                "success": True,
                "latency": elapsed,
                "ttft": None,
                "total_tokens": usage.get("total_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            return {"success": False, "latency": elapsed, "error": str(e)}


# Questions for diverse prefix test
QUESTIONS = [
    "Summarize the market overview section.",
    "What are the key regional differences in AI adoption?",
    "Describe the healthcare AI applications mentioned.",
    "What is the outlook for Edge AI?",
    "Compare the strategies of OpenAI and Google.",
    "What regulatory frameworks are discussed?",
    "How is manufacturing using AI?",
    "What are the investment trends in AI startups?",
    "Describe the multimodal AI trends.",
    "What is the future outlook for AGI?",
    "How does the EU AI Act affect global regulation?",
    "What role does Meta play in AI democratization?",
    "Describe the retail AI applications.",
    "What are the key technology trends for 2025?",
    "How is China approaching AI development?",
    "What is the total global AI market size?",
    "Describe the finance sector AI applications.",
    "What is Amazon's AI strategy?",
    "How are Korean companies using AI?",
    "What are the ethical concerns in AI?",
]


def _run_concurrent(client, endpoint_name, num_requests, use_same_context, params, ic_name):
    """Run concurrent requests, return results and total duration."""
    results = []
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for i in range(num_requests):
            if use_same_context:
                context = SHARED_CONTEXT
                question = QUESTIONS[i % len(QUESTIONS)]
            else:
                # Each request gets unique context (no cache benefit)
                context = f"DOCUMENT_{i}: " + f"Unique content for request {i}. " * 200
                question = f"Summarize document {i}."

            futures.append(executor.submit(
                _invoke_single, client, endpoint_name, context, question, params, ic_name,
            ))

        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    duration = time.time() - start
    return results, duration


def _analyze(results, duration, test_type, concurrent_requests) -> KVCacheResult:
    """Analyze results into KVCacheResult."""
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if not successful:
        return KVCacheResult(
            test_type=test_type, concurrent_requests=concurrent_requests,
            total_requests=len(results), successful=0, failed=len(failed),
            latency_p50=0, latency_p90=0, latency_p99=0, latency_avg=0,
            ttft_p50=None, ttft_p90=None, ttft_avg=None,
            throughput_tps=0, total_tokens=0, duration_sec=duration,
        )

    latencies = sorted([r["latency"] for r in successful])
    total_tokens = sum(r.get("total_tokens", 0) for r in successful)

    ttft_values = sorted([r["ttft"] for r in successful if r.get("ttft") is not None])

    result = KVCacheResult(
        test_type=test_type,
        concurrent_requests=concurrent_requests,
        total_requests=len(results),
        successful=len(successful),
        failed=len(failed),
        latency_p50=latencies[len(latencies) // 2],
        latency_p90=latencies[int(len(latencies) * 0.9)],
        latency_p99=latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)],
        latency_avg=statistics.mean(latencies),
        ttft_p50=ttft_values[len(ttft_values) // 2] if ttft_values else None,
        ttft_p90=ttft_values[int(len(ttft_values) * 0.9)] if ttft_values else None,
        ttft_avg=statistics.mean(ttft_values) if ttft_values else None,
        throughput_tps=total_tokens / duration if duration > 0 else 0,
        total_tokens=total_tokens,
        duration_sec=duration,
    )

    print(f"  {test_type}: {len(successful)}/{len(results)} ok | "
          f"p50={result.latency_p50:.2f}s | p90={result.latency_p90:.2f}s | "
          f"TPS={result.throughput_tps:.1f}" +
          (f" | TTFT_p50={result.ttft_p50:.3f}s" if result.ttft_p50 else ""))

    return result


def _compare(same: KVCacheResult, diff: KVCacheResult) -> dict:
    """Compare same-prefix vs different-prefix results."""
    def improvement(same_val, diff_val):
        if diff_val and diff_val > 0:
            return ((diff_val - same_val) / diff_val) * 100
        return None

    comp = {
        "latency_p50_improvement": improvement(same.latency_p50, diff.latency_p50),
        "latency_p90_improvement": improvement(same.latency_p90, diff.latency_p90),
        "throughput_improvement": improvement(-same.throughput_tps, -diff.throughput_tps),
    }
    if same.ttft_p50 is not None and diff.ttft_p50 is not None:
        comp["ttft_p50_improvement"] = improvement(same.ttft_p50, diff.ttft_p50)
        comp["ttft_p90_improvement"] = improvement(same.ttft_p90, diff.ttft_p90)

    return comp


def _print_comparison(same: KVCacheResult, diff: KVCacheResult, comp: dict):
    """Print comparison table."""
    print(f"\n{'=' * 70}")
    print("KV Cache Effectiveness Analysis")
    print(f"{'=' * 70}\n")

    print(f"{'Metric':<25} {'Same Prefix':>15} {'Diff Prefix':>15} {'Improvement':>15}")
    print("-" * 70)

    print(f"{'Latency P50':<25} {same.latency_p50:>14.2f}s {diff.latency_p50:>14.2f}s "
          f"{comp.get('latency_p50_improvement', 0):>13.1f}%")
    print(f"{'Latency P90':<25} {same.latency_p90:>14.2f}s {diff.latency_p90:>14.2f}s "
          f"{comp.get('latency_p90_improvement', 0):>13.1f}%")
    print(f"{'Throughput (TPS)':<25} {same.throughput_tps:>14.1f} {diff.throughput_tps:>14.1f} "
          f"{comp.get('throughput_improvement', 0):>13.1f}%")

    if same.ttft_p50 is not None and diff.ttft_p50 is not None:
        print(f"{'TTFT P50':<25} {same.ttft_p50:>14.3f}s {diff.ttft_p50:>14.3f}s "
              f"{comp.get('ttft_p50_improvement', 0):>13.1f}%")
        print(f"{'TTFT P90':<25} {same.ttft_p90:>14.3f}s {diff.ttft_p90:>14.3f}s "
              f"{comp.get('ttft_p90_improvement', 0):>13.1f}%")

    print(f"\n{'Total Tokens':<25} {same.total_tokens:>15} {diff.total_tokens:>15}")
    print(f"{'Duration':<25} {same.duration_sec:>14.1f}s {diff.duration_sec:>14.1f}s")

    print(f"\nPrefix caching {'EFFECTIVE' if (comp.get('latency_p50_improvement', 0) or 0) > 5 else 'MINIMAL IMPACT'} "
          f"for this workload.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KV Cache Effectiveness Benchmark")
    parser.add_argument("config", help="YAML config path")
    parser.add_argument("--endpoint", required=True, help="Endpoint name")
    parser.add_argument("--ic", default=None, help="Inference Component name")
    parser.add_argument("--concurrent", type=int, default=20, help="Concurrent requests")
    parser.add_argument("--context-tokens", type=int, default=4000,
                        help="Approximate context length in tokens")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_kvcache_benchmark(cfg, args.endpoint, args.ic,
                          concurrent_requests=args.concurrent)
