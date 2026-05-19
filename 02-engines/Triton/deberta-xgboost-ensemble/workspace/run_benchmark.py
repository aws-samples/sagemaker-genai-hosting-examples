"""
Step 4: Benchmark Endpoint

Runs latency and throughput benchmarks against a deployed Triton endpoint.
Supports Inference Component routing via --inference-component-name.

This script performs client-side tokenization: for each input text, it constructs
N (premise, hypothesis) pairs, tokenizes them as a batch, and sends the
[N, max_seq_len] tensors to the Triton ensemble.

Inputs:
- Deployed SageMaker endpoint

Outputs:
- Latency statistics (min, mean, median, p90, p95, p99, max)
- Throughput metrics (requests per second)
- Sample predictions

Usage:
    python workspace/run_benchmark.py --endpoint-name <name> --inference-component-name <name> [--warmup 5] [--iterations 50] [--concurrency 1]
"""

import argparse
import json
import statistics
import time
import concurrent.futures

import boto3
import numpy as np
from transformers import AutoTokenizer

from config import (
    NLI_MODEL_ID,
    NLI_LABELS,
    N_LABELS,
    MAX_SEQ_LEN,
    HYPOTHESIS_TEMPLATE,
    BENCHMARK_TEXTS,
)

# Load tokenizer globally for reuse
_tokenizer = None


def get_tokenizer():
    """Get or create the NLI tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        print(f"Loading tokenizer from {NLI_MODEL_ID}...")
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
        print("✓ Tokenizer loaded")
    return _tokenizer


def tokenize_nli_pairs(text: str, labels: list[str] = NLI_LABELS, max_seq_len: int = MAX_SEQ_LEN):
    """
    Tokenize N (premise, hypothesis) pairs for NLI scoring.

    For a single input text and N candidate labels, constructs:
        premise_i   = text
        hypothesis_i = "This example is {label_i}."

    Returns:
        input_ids:      list of lists, shape [N, max_seq_len]
        attention_mask: list of lists, shape [N, max_seq_len]
    """
    tokenizer = get_tokenizer()
    premises = [text] * len(labels)
    hypotheses = [HYPOTHESIS_TEMPLATE.format(label) for label in labels]

    encoded = tokenizer(
        premises,
        hypotheses,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="np",
    )
    return encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()


def invoke_endpoint(
    runtime,
    endpoint_name: str,
    text: str,
    inference_component_name: str | None = None,
) -> tuple[int, float]:
    """
    Invoke the Triton endpoint with a single text input.

    Tokenizes the text against all N labels client-side, sends [N, max_seq_len]
    tensors to the ensemble. With max_batch_size=0, shapes match config exactly.

    Returns:
        tuple: (prediction, latency_ms)
    """
    input_ids, attention_mask = tokenize_nli_pairs(text)

    # With max_batch_size=0, shape matches config dims exactly (no batch dim)
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [N_LABELS, MAX_SEQ_LEN],
                "datatype": "INT64",
                "data": input_ids,
            },
            {
                "name": "attention_mask",
                "shape": [N_LABELS, MAX_SEQ_LEN],
                "datatype": "INT64",
                "data": attention_mask,
            },
        ]
    }

    invoke_kwargs = dict(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Body=json.dumps(payload),
    )
    if inference_component_name:
        invoke_kwargs["InferenceComponentName"] = inference_component_name

    t0 = time.perf_counter()
    response = runtime.invoke_endpoint(**invoke_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = json.loads(response["Body"].read().decode("utf-8"))
    prediction = result["outputs"][0]["data"][0]

    return int(prediction), latency_ms


def _worker(args):
    """Worker function for concurrent benchmarking."""
    runtime, endpoint_name, text, ic_name = args
    return invoke_endpoint(runtime, endpoint_name, text, ic_name)


def run_benchmark(
    endpoint_name: str,
    region: str,
    warmup: int = 5,
    iterations: int = 50,
    concurrency: int = 1,
    inference_component_name: str | None = None,
):
    """Run benchmark against the deployed endpoint."""

    runtime = boto3.client("sagemaker-runtime", region_name=region)
    texts = BENCHMARK_TEXTS

    def cycle_text(i):
        return texts[i % len(texts)]

    print("=" * 70)
    print("Benchmark Configuration")
    print("=" * 70)
    print(f"  Endpoint: {endpoint_name}")
    if inference_component_name:
        print(f"  Inference Component: {inference_component_name}")
    print(f"  Region: {region}")
    print(f"  NLI labels: {N_LABELS}")
    print(f"  Max sequence length: {MAX_SEQ_LEN}")
    print(f"  Warmup requests: {warmup}")
    print(f"  Benchmark requests: {iterations}")
    print(f"  Concurrency: {concurrency}")
    print()

    # Load tokenizer
    get_tokenizer()
    print()

    # Warmup phase
    print(f"Running {warmup} warmup request(s)...")
    for i in range(warmup):
        invoke_endpoint(runtime, endpoint_name, cycle_text(i), inference_component_name)
    print("✓ Warmup complete")
    print()

    # Benchmark phase
    print(f"Running {iterations} benchmark request(s)...")
    latencies = []
    errors = 0

    if concurrency == 1:
        for i in range(iterations):
            try:
                _, ms = invoke_endpoint(runtime, endpoint_name, cycle_text(i), inference_component_name)
                latencies.append(ms)
            except Exception as e:
                errors += 1
                print(f"  [ERROR] Request {i}: {e}")
    else:
        work = [
            (runtime, endpoint_name, cycle_text(i), inference_component_name)
            for i in range(iterations)
        ]
        wall_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, w): idx for idx, w in enumerate(work)}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    _, ms = fut.result()
                    latencies.append(ms)
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR]: {e}")

        wall_ms = (time.perf_counter() - wall_start) * 1000.0

    print("✓ Benchmark complete")
    print()

    # Compute statistics
    if not latencies:
        print("No successful requests - cannot compute stats")
        return

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(p):
        idx = max(0, int(np.ceil(p / 100.0 * n)) - 1)
        return latencies_sorted[idx]

    print("=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"  Successful requests: {n}")
    print(f"  Failed requests: {errors}")
    print()
    print("Latency Statistics:")
    print(f"  Min      : {min(latencies):.1f} ms")
    print(f"  Mean     : {statistics.mean(latencies):.1f} ms")
    print(f"  Median   : {statistics.median(latencies):.1f} ms")
    print(f"  p90      : {percentile(90):.1f} ms")
    print(f"  p95      : {percentile(95):.1f} ms")
    print(f"  p99      : {percentile(99):.1f} ms")
    print(f"  Max      : {max(latencies):.1f} ms")

    if n > 1:
        print(f"  StdDev   : {statistics.stdev(latencies):.1f} ms")

    if concurrency > 1:
        rps = n / (wall_ms / 1000.0)
        print()
        print("Throughput:")
        print(f"  Requests/sec: {rps:.1f} req/s  (concurrency={concurrency})")

    # Show sample predictions
    print()
    print("Sample Predictions:")
    for text in BENCHMARK_TEXTS[:4]:
        pred, ms = invoke_endpoint(runtime, endpoint_name, text, inference_component_name)
        print(f"  [{pred}]  ({ms:.0f} ms)  \"{text[:55]}\"")

    print()


def main():
    parser = argparse.ArgumentParser(description="Step 4: Benchmark Triton endpoint")
    parser.add_argument(
        "--endpoint-name",
        required=True,
        help="SageMaker endpoint name to benchmark",
    )
    parser.add_argument(
        "--inference-component-name",
        default=None,
        help="Inference component name (required for IC-based deployments)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (default: from boto3 session)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark requests (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent threads (default: 1)",
    )
    args = parser.parse_args()

    region = args.region
    if not region:
        session = boto3.Session()
        region = session.region_name

    run_benchmark(
        endpoint_name=args.endpoint_name,
        region=region,
        warmup=args.warmup,
        iterations=args.iterations,
        concurrency=args.concurrency,
        inference_component_name=args.inference_component_name,
    )

    print("=" * 70)
    print("✓ Step 4 Complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
