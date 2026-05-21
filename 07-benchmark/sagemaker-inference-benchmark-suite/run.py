#!/usr/bin/env python3
"""
SageMaker Inference Benchmark Suite.

Declarative:  run.py -f recipe.yaml              YAML drives pipeline & config
Imperative:   run.py --deploy --model M ...       CLI drives pipeline & config
Query:        run.py --status                     No config needed

YAML recipes are self-contained: model, instance, container, optimization,
benchmark params, AND pipeline steps are all declared in the file.
CLI flags are for ad-hoc execution and parameter overrides only.

Usage:
    run.py -f recipes/qwen3-32b-g7e-eagle3.yaml
    run.py -f recipes/recipe.yaml --dry-run
    run.py -f recipes/recipe.yaml --only benchmark --endpoint NAME
    run.py -f recipes/recipe.yaml --skip cleanup
    run.py --model Qwen/Qwen3-32B --instance ml.g7e.2xlarge --deploy
    run.py --status [--region us-west-2]
    run.py --report [--cost ml.g7e.2xlarge=4.20]
    run.py --cleanup --endpoint NAME [--region us-west-2]
"""

import argparse
import sys

VALID_STEPS = {"deploy", "benchmark", "kvcache", "report", "cleanup"}


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker Inference Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recipe mode (declarative — YAML drives pipeline):
  run.py -f recipes/qwen3-32b-g7e-eagle3.yaml
  run.py -f recipes/recipe.yaml --dry-run
  run.py -f recipes/recipe.yaml --only benchmark --endpoint NAME
  run.py -f recipes/recipe.yaml --skip cleanup
  run.py -f recipes/recipe.yaml --image-uri URI     # param override

Ad-hoc mode (imperative — CLI drives pipeline):
  run.py --model Qwen/Qwen3-32B --instance ml.g7e.2xlarge --deploy
  run.py --model Qwen/Qwen3-32B --instance ml.g7e.2xlarge --deploy --benchmark --cleanup

Standalone (no config needed):
  run.py --status --region us-west-2
  run.py --report --cost ml.g7e.2xlarge=4.20
  run.py --cleanup --endpoint NAME --region us-west-2
        """,
    )

    # ---- Recipe mode ----
    parser.add_argument("-f", "--file", default=None, help="YAML recipe file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config only (no AWS calls)")
    parser.add_argument("--only", default=None,
                        help="Run specific steps only (comma-sep: deploy,benchmark,...)")
    parser.add_argument("--skip", default=None,
                        help="Skip steps from pipeline (comma-sep)")

    # ---- Ad-hoc actions (without -f) ----
    parser.add_argument("--deploy", action="store_true", help="Deploy endpoint")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--kvcache", action="store_true", help="Run KV cache test")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete endpoint (ad-hoc or standalone)")

    # ---- Standalone ----
    parser.add_argument("--status", action="store_true", help="List active endpoints")
    parser.add_argument("--report", action="store_true",
                        help="Generate report from CSV results")

    # ---- Model & instance (ad-hoc or override) ----
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID (e.g., Qwen/Qwen3-32B)")
    parser.add_argument("--instance", default=None,
                        help="SageMaker instance type (e.g., ml.g7e.2xlarge)")

    # ---- Container (override) ----
    parser.add_argument("--image-uri", default=None,
                        help="Container image URI (overrides recipe)")
    parser.add_argument("--container-type", default=None,
                        choices=["vllm-dlc", "djl-lmi", "byoc"],
                        help="Container type")
    parser.add_argument("--container-version", default=None,
                        help="Container version (e.g., 0.18.0)")
    parser.add_argument("--cuda", default=None, help="CUDA version (e.g., cu129)")
    parser.add_argument("--public-ecr", action="store_true",
                        help="Use public.ecr.aws format (vLLM 0.18+)")

    # ---- vLLM params (override) ----
    parser.add_argument("--tp", type=int, default=None,
                        help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max sequence length")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Max concurrent sequences")
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--dtype", default=None,
                        help="Model dtype (bfloat16, float16, auto)")

    # ---- Endpoint ----
    parser.add_argument("--endpoint", default=None,
                        help="Endpoint name (for benchmark/cleanup)")
    parser.add_argument("--ic", default=None,
                        help="Inference Component name")
    parser.add_argument("--role", default=None, help="IAM role ARN")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--pattern", default=None,
                        choices=["standard", "inference_component"],
                        help="Deployment pattern")

    # ---- Benchmark overrides ----
    parser.add_argument("--use-case", default=None,
                        help="Use cases (comma-sep or 'all')")
    parser.add_argument("--concurrency", default=None,
                        help="Concurrency levels (comma-sep)")
    parser.add_argument("--requests", type=int, default=None,
                        help="Requests per level")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable streaming (TTFT)")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max output tokens")

    # ---- KV cache ----
    parser.add_argument("--concurrent", type=int, default=20,
                        help="Concurrent requests for kvcache test")

    # ---- Report ----
    parser.add_argument("--cost", nargs="+",
                        help="Instance costs (e.g., ml.g7e.2xlarge=4.20)")

    # ---- TUI ----
    parser.add_argument("--tui", action="store_true",
                        help="Launch interactive TUI dashboard")

    # ---- Misc ----
    parser.add_argument("--prefix", default=None,
                        help="Endpoint prefix filter for status")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip smoke test after deploy")

    args = parser.parse_args()

    # ================================================================
    # TUI mode
    # ================================================================

    if args.tui:
        from tui.app import BenchmarkApp
        app = BenchmarkApp()
        app.run()
        return 0

    # ================================================================
    # Standalone mode (no config needed)
    # ================================================================

    if args.status:
        from scripts.deployer import list_endpoints
        regions = [args.region] if args.region != "all" else ["us-west-2", "us-east-1"]
        for r in regions:
            list_endpoints(r, args.prefix)
        return 0

    if args.report and not args.file and not args.model:
        from scripts.reporter import generate_report
        report = generate_report(args.results_dir or "results/matrix",
                                 _parse_cost(args.cost))
        print(report)
        return 0

    if args.cleanup and not args.file and not args.model and not args.deploy:
        if not args.endpoint:
            print("Error: --endpoint required for standalone cleanup.")
            return 1
        from scripts.deployer import cleanup
        cleanup(args.endpoint, args.region, args.pattern or "standard")
        return 0

    # ================================================================
    # Recipe mode (-f): YAML drives pipeline
    # ================================================================

    if args.file:
        config = _build_config(args)
        if config is None:
            return 1

        if args.dry_run:
            _do_validate(config, args.file)
            return 0

        steps = _resolve_pipeline(config, args)
        for s in steps:
            if s not in VALID_STEPS:
                print(f"Error: unknown pipeline step '{s}'. "
                      f"Valid: {', '.join(sorted(VALID_STEPS))}")
                return 1

        return _run_pipeline(config, steps, args)

    # ================================================================
    # Ad-hoc mode (no -f, explicit action flags)
    # ================================================================

    has_action = any([args.deploy, args.benchmark, args.kvcache])
    if not has_action:
        parser.print_help()
        return 1

    config = _build_config(args)
    if config is None:
        print("Error: --model and --instance required for ad-hoc mode.")
        return 1

    steps = []
    if args.deploy:
        steps.append("deploy")
    if args.benchmark:
        steps.append("benchmark")
    if args.kvcache:
        steps.append("kvcache")
    if args.report:
        steps.append("report")
    if args.cleanup:
        steps.append("cleanup")

    return _run_pipeline(config, steps, args)


# ---------------------------------------------------------------------------
# Pipeline resolution
# ---------------------------------------------------------------------------

def _resolve_pipeline(config, args):
    """Determine pipeline steps for recipe mode.

    Priority: --only > explicit action flags > YAML pipeline field.
    """
    # --only overrides everything
    if args.only:
        return [s.strip() for s in args.only.split(",")]

    # Explicit action flags override YAML pipeline (backward compat)
    has_action = any([args.deploy, args.benchmark, args.kvcache,
                      args.cleanup, args.report])
    if has_action:
        steps = []
        if args.deploy:
            steps.append("deploy")
        if args.benchmark:
            steps.append("benchmark")
        if args.kvcache:
            steps.append("kvcache")
        if args.report:
            steps.append("report")
        if args.cleanup:
            steps.append("cleanup")
    else:
        # Use pipeline from YAML (the intended way)
        steps = list(config.pipeline)

    # Apply --skip
    if args.skip:
        skip = set(s.strip() for s in args.skip.split(","))
        steps = [s for s in steps if s not in skip]

    return steps


def _run_pipeline(config, steps, args):
    """Execute pipeline steps in order."""
    endpoint_name = args.endpoint or config.endpoint
    ic_name = args.ic

    for step in steps:
        if step == "deploy":
            result = _do_deploy(config, args.no_test)
            if not result.success:
                return 1
            endpoint_name = result.endpoint_name
            ic_name = result.ic_name

        elif step == "benchmark":
            if not endpoint_name:
                print("Error: endpoint required for benchmark. "
                      "Use --endpoint or include deploy in pipeline.")
                return 1
            _apply_benchmark_overrides(config, args)
            _do_benchmark(config, endpoint_name, ic_name, args)

        elif step == "kvcache":
            if not endpoint_name:
                print("Error: endpoint required for kvcache test.")
                return 1
            _do_kvcache(config, endpoint_name, ic_name, args.concurrent)

        elif step == "report":
            from scripts.reporter import generate_report
            report = generate_report(args.results_dir or "results/matrix",
                                     _parse_cost(args.cost))
            print(report)

        elif step == "cleanup":
            if endpoint_name:
                from scripts.deployer import cleanup
                print(f"\nCleaning up: {endpoint_name}")
                cleanup(endpoint_name, config.deployment.endpoint.region,
                        config.deployment.endpoint.pattern,
                        platform=config.deployment.platform,
                        hyperpod_config=config.deployment.hyperpod)
            else:
                print("Warning: no endpoint to clean up (skipping).")

    return 0


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config(args):
    """Build BenchmarkConfig from recipe file, inline args, or both."""
    from scripts.config_loader import BenchmarkConfig, load_config

    if args.file:
        config = load_config(args.file)
    elif args.model and args.instance:
        config = BenchmarkConfig()
        config.deployment.model.id = args.model
        config.deployment.model.short_name = args.model.split("/")[-1].lower().replace("-", "")[:20]
        config.deployment.instance.type = args.instance
        config.name = f"{args.model} on {args.instance}"
    elif args.model or args.instance:
        print("Error: both --model and --instance are required for ad-hoc mode.")
        return None
    else:
        return None

    # Apply parameter overrides (not actions, just config params)
    _apply_param_overrides(config, args)
    return config


def _apply_param_overrides(config, args):
    """Apply CLI parameter overrides to config."""
    d = config.deployment
    if args.model:
        d.model.id = args.model
        d.model.short_name = args.model.split("/")[-1].lower().replace("-", "")[:20]
    if args.instance:
        d.instance.type = args.instance
    if args.image_uri:
        d.container.image_uri = args.image_uri
    if args.container_type:
        d.container.type = args.container_type
    if args.container_version:
        d.container.version = args.container_version
    if args.cuda:
        d.container.cuda = args.cuda
    if args.public_ecr:
        d.container.public_ecr = True
    if args.tp is not None:
        d.vllm.tensor_parallel_size = args.tp
    if args.max_model_len is not None:
        d.vllm.max_model_len = args.max_model_len
    if args.max_num_seqs is not None:
        d.vllm.max_num_seqs = args.max_num_seqs
    if args.gpu_mem is not None:
        d.vllm.gpu_memory_utilization = args.gpu_mem
    if args.dtype:
        d.vllm.dtype = args.dtype
    if args.role:
        d.endpoint.role_arn = args.role
    if args.region:
        d.endpoint.region = args.region
    if args.pattern:
        d.endpoint.pattern = args.pattern
    if args.max_tokens is not None:
        config.benchmark.inference_params.max_tokens = args.max_tokens


def _apply_benchmark_overrides(config, args):
    """Apply CLI benchmark overrides to config."""
    from scripts.prompts import USE_CASE_PROMPTS
    if args.use_case:
        if args.use_case == "all":
            config.benchmark.use_cases = list(USE_CASE_PROMPTS.keys())
        else:
            config.benchmark.use_cases = [uc.strip() for uc in args.use_case.split(",")]
    if args.concurrency:
        config.benchmark.concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    if args.requests is not None:
        config.benchmark.requests_per_level = args.requests
    if args.streaming:
        config.benchmark.streaming = True


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _do_validate(config, source):
    from scripts.config_loader import (
        validate_config, build_container_uri, build_env_vars,
        build_endpoint_name, print_config_summary,
    )
    print(f"Config: {source}")
    print(f"Name: {config.name}")
    print(f"Pipeline: {config.pipeline}")
    print(f"Platform: {config.deployment.platform}")
    print_config_summary(config)

    warnings = validate_config(config)
    for w in warnings:
        print(f"  WARNING: {w}")

    if config.deployment.platform == "hyperpod":
        hp = config.deployment.hyperpod
        print(f"\nHyperPod cluster: {hp.cluster_name}")
        print(f"  Namespace: {hp.namespace}")
        print(f"  Replicas: {hp.replicas}")
        print(f"  Model source: {hp.model_source.type} ({hp.model_source.s3_bucket or hp.model_source.fsx_file_system_id})")
        print(f"  KV Cache: L1={hp.kv_cache.l1_cache}, L2={hp.kv_cache.l2_cache} ({hp.kv_cache.l2_backend})")
        print(f"  Routing: {hp.routing.strategy} (enabled={hp.routing.enabled})")
        print(f"  Worker: {hp.worker.gpu_count} GPU, {hp.worker.cpu_request} CPU, {hp.worker.memory_request} mem")
    else:
        uri = build_container_uri(config)
        env = build_env_vars(config)
        name = build_endpoint_name(config)
        print(f"\nContainer URI: {uri}")
        print(f"Endpoint name: {name}")
        print(f"Env vars ({len(env)}):")
        for k, v in sorted(env.items()):
            display_v = v if len(str(v)) < 80 else str(v)[:77] + "..."
            print(f"  {k}={display_v}")

    print(f"\nValidation passed.")


def _do_deploy(config, no_test=False):
    from scripts.deployer import deploy, smoke_test
    result = deploy(config)
    if result.success and not no_test:
        smoke_test(config, result.endpoint_name, result.ic_name)
    if result.success:
        print(f"\nEndpoint ready: {result.endpoint_name}")
        if result.ic_name:
            print(f"IC: {result.ic_name}")
    return result


def _do_benchmark(config, endpoint_name, ic_name, args):
    from scripts.benchmarker import run_benchmark
    results = run_benchmark(config, endpoint_name, ic_name=ic_name,
                            results_dir=args.results_dir)
    ok = sum(1 for r in results if "error" not in r)
    fail = sum(1 for r in results if "error" in r)
    print(f"\nBenchmark complete: {ok} successful, {fail} failed")
    return results


def _do_kvcache(config, endpoint_name, ic_name, concurrent):
    from scripts.benchmark_kvcache import run_kvcache_benchmark
    run_kvcache_benchmark(config, endpoint_name, ic_name=ic_name,
                          concurrent_requests=concurrent)


def _parse_cost(cost_args):
    if not cost_args:
        return None
    cost_map = {}
    for pair in cost_args:
        inst, price = pair.split("=")
        cost_map[inst] = float(price)
    return cost_map


if __name__ == "__main__":
    sys.exit(main())
