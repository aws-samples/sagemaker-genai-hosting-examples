"""
Report generator for SageMaker inference benchmark results.

Reads CSV results and generates markdown reports with cost analysis,
optimization comparisons, and TTFT analysis.

Usage:
    python -m scripts.reporter --results-dir results/matrix
    python -m scripts.reporter --results-dir results/matrix --compare recipes/a.yaml recipes/b.yaml
"""

import csv
import glob
import os
import statistics
from datetime import datetime
from typing import Optional


DEFAULT_RESULTS_DIR = "results/matrix"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(results_dir: str = None, cost_per_hour: dict = None) -> str:
    """Generate a markdown benchmark report from CSV results.

    Args:
        results_dir: Directory containing CSV result files.
        cost_per_hour: Optional dict mapping instance_type -> $/hr for cost calc.
                       If not provided, reads from CSV metadata if available.

    Returns:
        Markdown report string.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    rows = _load_all_csv(results_dir)
    if not rows:
        return "No benchmark results found."

    lines = []
    lines.append("# SageMaker Inference Benchmark Report")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data points**: {len(rows)}")
    lines.append("")

    # Section 1: Peak throughput (C=32 or highest concurrency)
    lines.extend(_section_peak_throughput(rows, cost_per_hour))

    # Section 2: Single request performance (C=1)
    lines.extend(_section_single_request(rows))

    # Section 3: Optimization speedup comparison
    lines.extend(_section_optimization_speedup(rows))

    # Section 4: TTFT analysis (if streaming data available)
    ttft_lines = _section_ttft(rows)
    if ttft_lines:
        lines.extend(ttft_lines)

    # Section 5: Latency scaling
    lines.extend(_section_latency_scaling(rows))

    report = "\n".join(lines)

    # Write to file
    report_file = os.path.join(results_dir, "BENCHMARK_REPORT.md")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")

    return report


def calculate_cost(agg_tok_sec: float, cost_per_hour: float) -> Optional[float]:
    """Calculate cost per million output tokens.

    Formula: $/M tokens = ($/hr) / (agg_tok_sec * 3600) * 1,000,000
    """
    if agg_tok_sec <= 0 or cost_per_hour <= 0:
        return None
    return (cost_per_hour / (agg_tok_sec * 3600)) * 1_000_000


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_peak_throughput(rows: list[dict], cost_per_hour: dict = None) -> list[str]:
    lines = []
    lines.append("## Peak Throughput (Highest Concurrency)")
    lines.append("")

    # Find highest concurrency rows per endpoint+use_case
    peak_rows = _get_peak_concurrency_rows(rows)
    if not peak_rows:
        lines.append("No peak throughput data available.")
        return lines

    lines.append("| Model | Instance | Optimization | Use Case | tok/s | RPS | Agg tok/s | p50 (ms) | $/M tokens |")
    lines.append("|-------|----------|-------------|----------|-------|-----|-----------|----------|-----------|")

    for row in sorted(peak_rows, key=lambda x: _safe_float(x.get("aggregate_output_tok_sec", 0)), reverse=True):
        model = row.get("model", "")
        inst = _short_instance(row.get("instance_type", ""))
        opt = row.get("optimization", "")
        uc = row.get("use_case", "")
        tok_s = row.get("tok_per_sec_avg", "")
        rps = row.get("rps", "")
        agg = row.get("aggregate_output_tok_sec", "")
        p50 = row.get("latency_p50", "")

        # Cost calculation
        cost_str = ""
        agg_f = _safe_float(agg)
        inst_type = row.get("instance_type", "")
        if cost_per_hour and inst_type in cost_per_hour and agg_f > 0:
            cost = calculate_cost(agg_f, cost_per_hour[inst_type])
            cost_str = f"${cost:.2f}" if cost else ""

        lines.append(f"| {model} | {inst} | {opt} | {uc} | {tok_s} | {rps} | {agg} | {p50} | {cost_str} |")

    lines.append("")
    return lines


def _section_single_request(rows: list[dict]) -> list[str]:
    lines = []
    lines.append("## Single Request Performance (C=1)")
    lines.append("")

    c1_rows = [r for r in rows if str(r.get("concurrency")) == "1"]
    if not c1_rows:
        lines.append("No C=1 data available.")
        lines.append("")
        return lines

    lines.append("| Model | Instance | Optimization | Use Case | tok/s | Latency p50 (ms) | Avg Input | Avg Output |")
    lines.append("|-------|----------|-------------|----------|-------|-----------------|-----------|------------|")

    for row in sorted(c1_rows, key=lambda x: (x.get("model", ""), x.get("optimization", ""), x.get("use_case", ""))):
        lines.append(
            f"| {row.get('model', '')} | {_short_instance(row.get('instance_type', ''))} | "
            f"{row.get('optimization', '')} | {row.get('use_case', '')} | "
            f"{row.get('tok_per_sec_avg', '')} | {row.get('latency_p50', '')} | "
            f"{row.get('avg_input_tokens', '')} | {row.get('avg_output_tokens', '')} |"
        )

    lines.append("")
    return lines


def _section_optimization_speedup(rows: list[dict]) -> list[str]:
    lines = []
    lines.append("## Optimization Speedup (vs Vanilla)")
    lines.append("")

    peak_rows = _get_peak_concurrency_rows(rows)

    # Build vanilla baselines
    vanilla = {}
    for row in peak_rows:
        if row.get("optimization") == "vanilla":
            key = (row.get("model", ""), row.get("instance_type", ""), row.get("use_case", ""))
            vanilla[key] = _safe_float(row.get("tok_per_sec_avg", 0))

    if not vanilla:
        lines.append("No vanilla baseline data for comparison.")
        lines.append("")
        return lines

    lines.append("| Model | Instance | Use Case | Vanilla tok/s | Optimization | Optimized tok/s | Speedup |")
    lines.append("|-------|----------|----------|--------------|-------------|----------------|---------|")

    for row in sorted(peak_rows, key=lambda x: (x.get("model", ""), x.get("use_case", ""), x.get("optimization", ""))):
        if row.get("optimization") == "vanilla":
            continue
        key = (row.get("model", ""), row.get("instance_type", ""), row.get("use_case", ""))
        baseline = vanilla.get(key, 0)
        optimized = _safe_float(row.get("tok_per_sec_avg", 0))
        speedup = f"{optimized / baseline:.2f}x" if baseline > 0 else "N/A"
        lines.append(
            f"| {row.get('model', '')} | {_short_instance(row.get('instance_type', ''))} | "
            f"{row.get('use_case', '')} | {baseline:.1f} | "
            f"{row.get('optimization', '')} | {optimized:.1f} | {speedup} |"
        )

    lines.append("")
    return lines


def _section_ttft(rows: list[dict]) -> list[str]:
    """Generate TTFT analysis section. Returns empty list if no TTFT data."""
    ttft_rows = [r for r in rows if r.get("ttft_p50") and r["ttft_p50"] not in ("", "None")]
    if not ttft_rows:
        return []

    lines = []
    lines.append("## Time to First Token (TTFT)")
    lines.append("")
    lines.append("| Model | Instance | Optimization | Use Case | C | TTFT p50 (ms) | TTFT p90 (ms) | TTFT avg (ms) |")
    lines.append("|-------|----------|-------------|----------|---|--------------|--------------|--------------|")

    for row in sorted(ttft_rows, key=lambda x: (x.get("model", ""), str(x.get("concurrency", "")))):
        lines.append(
            f"| {row.get('model', '')} | {_short_instance(row.get('instance_type', ''))} | "
            f"{row.get('optimization', '')} | {row.get('use_case', '')} | "
            f"{row.get('concurrency', '')} | {row.get('ttft_p50', '')} | "
            f"{row.get('ttft_p90', '')} | {row.get('ttft_avg', '')} |"
        )

    lines.append("")
    return lines


def _section_latency_scaling(rows: list[dict]) -> list[str]:
    lines = []
    lines.append("## Latency Scaling Under Load")
    lines.append("")

    # Group by (model, instance, optimization, use_case), show C=1 vs peak
    groups = {}
    for row in rows:
        key = (row.get("model", ""), row.get("instance_type", ""),
               row.get("optimization", ""), row.get("use_case", ""))
        conc = int(row.get("concurrency", 0))
        groups.setdefault(key, {})[conc] = row

    lines.append("| Model | Instance | Optimization | Use Case | C=1 p50 | Peak C p50 | Degradation |")
    lines.append("|-------|----------|-------------|----------|---------|-----------|------------|")

    for key, conc_map in sorted(groups.items()):
        model, inst, opt, uc = key
        c1 = conc_map.get(1)
        peak_c = max(conc_map.keys())
        peak = conc_map.get(peak_c)
        if not c1 or not peak or peak_c <= 1:
            continue

        c1_p50 = _safe_float(c1.get("latency_p50", 0))
        peak_p50 = _safe_float(peak.get("latency_p50", 0))
        if c1_p50 > 0:
            degradation = f"+{((peak_p50 - c1_p50) / c1_p50 * 100):.0f}%"
        else:
            degradation = "N/A"

        lines.append(
            f"| {model} | {_short_instance(inst)} | {opt} | {uc} | "
            f"{c1_p50:.0f}ms | {peak_p50:.0f}ms (C={peak_c}) | {degradation} |"
        )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_csv(results_dir: str) -> list[dict]:
    """Load all CSV files from results directory."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        return []

    all_rows = []
    for f in csv_files:
        try:
            with open(f) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    all_rows.append(row)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
    return all_rows


def _get_peak_concurrency_rows(rows: list[dict]) -> list[dict]:
    """Get the highest concurrency row for each (endpoint, use_case) combo."""
    groups = {}
    for row in rows:
        key = (row.get("endpoint", ""), row.get("use_case", ""))
        conc = int(row.get("concurrency", 0))
        if key not in groups or conc > int(groups[key].get("concurrency", 0)):
            groups[key] = row
    return list(groups.values())


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _short_instance(inst_type: str) -> str:
    return inst_type.replace("ml.", "").replace("xlarge", "xl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Report Generator")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory containing CSV results")
    parser.add_argument("--cost", nargs="*",
                        help="Instance costs as type=price pairs (e.g., ml.g7e.2xlarge=4.20)")
    args = parser.parse_args()

    cost_map = None
    if args.cost:
        cost_map = {}
        for pair in args.cost:
            inst, price = pair.split("=")
            cost_map[inst] = float(price)

    report = generate_report(args.results_dir, cost_map)
    print(report)
