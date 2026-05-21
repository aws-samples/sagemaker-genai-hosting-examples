"""Data loading utilities for TUI screens."""

import csv
import glob
import os

import yaml


def load_recipes(recipes_dir: str = "recipes") -> list[dict]:
    """Load all YAML recipes, extract metadata."""
    recipes = []
    pattern = os.path.join(recipes_dir, "*.yaml")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
            if not raw or not isinstance(raw, dict):
                continue
            dep = raw.get("deployment", {})
            recipes.append({
                "file": os.path.basename(path),
                "path": path,
                "name": raw.get("name", os.path.basename(path).replace(".yaml", "")),
                "model": dep.get("model", {}).get("id", "?"),
                "instance": dep.get("instance", {}).get("type", "?"),
                "optimization": _get_optimization(dep),
                "container": dep.get("container", {}).get("type", "?"),
                "pipeline": ", ".join(raw.get("pipeline", [])),
                "cost": raw.get("cost", {}).get("instance_cost_per_hour"),
            })
        except Exception:
            continue
    return recipes


def load_results(results_dir: str = "results/matrix") -> list[dict]:
    """Load all CSV result files, return list of row dicts."""
    rows = []
    pattern = os.path.join(results_dir, "*.csv")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for key in ["concurrency", "successful", "failed", "total_requests"]:
                        if key in row and row[key]:
                            try:
                                row[key] = int(row[key])
                            except ValueError:
                                pass
                    for key in ["latency_p50", "latency_p90", "latency_p99", "latency_avg",
                                "tok_per_sec_avg", "rps", "aggregate_output_tok_sec",
                                "avg_output_tokens", "avg_input_tokens", "output_validation_rate",
                                "ttft_p50", "ttft_p90", "ttft_avg"]:
                        if key in row and row[key]:
                            try:
                                row[key] = float(row[key])
                            except ValueError:
                                pass
                    row["_source_file"] = os.path.basename(path)
                    rows.append(row)
        except Exception:
            continue
    return rows


def group_results(rows: list[dict]) -> dict[str, list[dict]]:
    """Group results by (model, optimization, instance_type, use_case)."""
    groups = {}
    for row in rows:
        key = (
            row.get("model", ""),
            row.get("optimization", ""),
            row.get("instance_type", ""),
            row.get("use_case", ""),
        )
        groups.setdefault(key, []).append(row)
    # Sort each group by concurrency
    for key in groups:
        groups[key].sort(key=lambda r: r.get("concurrency", 0))
    return groups


def sparkline(values: list[float]) -> str:
    """Generate ASCII sparkline using block characters."""
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    mn, mx = min(values), max(values)
    if mn == mx:
        return blocks[4] * len(values)
    return "".join(blocks[1 + int((v - mn) / (mx - mn) * 7)] for v in values)


def _get_optimization(deployment: dict) -> str:
    """Derive optimization label from deployment config."""
    spec = deployment.get("speculative_decoding", {})
    if spec.get("enabled"):
        method = spec.get("method", "eagle3")
        return method

    vllm = deployment.get("vllm", {})
    if vllm.get("swap_space"):
        return "lmcache"

    prefix = deployment.get("prefix_caching", {})
    if prefix.get("enabled"):
        return "prefix_cache"

    return "vanilla"
