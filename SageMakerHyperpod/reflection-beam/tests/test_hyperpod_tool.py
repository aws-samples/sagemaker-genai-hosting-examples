from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hyperpod_tool


PUBLIC_DIGEST = "ghcr.io/example/runtime@sha256:" + ("a" * 64)
FULL_COMMIT = "b" * 40


def configured_values() -> dict[str, str]:
    return {
        "BEAM_NAMESPACE": "default",
        "JUMPSTART_ENDPOINT_NAME": "reflection-beam",
        "JUMPSTART_INSTANCE_TYPE": "ml.example.1xlarge",
        "JUMPSTART_MODEL_ID": "public-beam-jumpstart",
        "JUMPSTART_MODEL_VERSION": "1.0.0",
        "SGLANG_INSTANCE_TYPE": "ml.example.8xlarge",
        "SGLANG_GPU_COUNT": "8",
        "SGLANG_TENSOR_PARALLEL_SIZE": "8",
        "SGLANG_ATTENTION_BACKEND": "trtllm_mha",
        "SGLANG_PAGE_SIZE": "64",
        "SGLANG_CONTEXT_LENGTH": "1048576",
        "SGLANG_MEM_FRACTION_STATIC": "0.85",
        "SGLANG_DFLASH_BLOCK_SIZE": "8",
        "SGLANG_MODEL_ID": "reflection/example-beam",
        "SGLANG_MODEL_COMMIT_SHA": FULL_COMMIT,
        "SGLANG_DRAFT_MODEL_ID": "reflection/example-beam-dflash2",
        "SGLANG_IMAGE": PUBLIC_DIGEST,
        "SGLANG_REASONING_PARSER": "documented-parser",
        "SGLANG_TOOL_CALL_PARSER": "documented-tool-parser",
        "VLLM_INSTANCE_TYPE": "ml.example.8xlarge",
        "VLLM_GPU_COUNT": "8",
        "VLLM_TENSOR_PARALLEL_SIZE": "8",
        "VLLM_MODEL_ID": "reflection/example-beam",
        "VLLM_MODEL_COMMIT_SHA": FULL_COMMIT,
        "VLLM_IMAGE": PUBLIC_DIGEST,
        "VLLM_REASONING_PARSER": "documented-parser",
        "VLLM_TOOL_CALL_PARSER": "documented-tool-parser",
    }


class ConfigTests(unittest.TestCase):
    def test_example_values_fail_with_all_missing_fields(self) -> None:
        values = {}
        values.update(
            hyperpod_tool.parse_env_file(
                hyperpod_tool.SAMPLE_ROOT / "config/deployment.env.example"
            )
        )
        values.update(
            hyperpod_tool.parse_env_file(
                hyperpod_tool.SAMPLE_ROOT / "config/release.env.example"
            )
        )
        with self.assertRaisesRegex(
            hyperpod_tool.ToolError,
            "missing configuration for sglang",
        ):
            hyperpod_tool.render_manifest("sglang", values)

    def test_release_config_separates_sglang_target_and_drafter(self) -> None:
        values = hyperpod_tool.parse_env_file(
            hyperpod_tool.SAMPLE_ROOT / "config/release.env.example"
        )
        self.assertEqual(
            values["SGLANG_MODEL_ID"],
            "reflection/Beam-1.0-501B-A23B-NVFP4",
        )
        self.assertEqual(
            values["SGLANG_DRAFT_MODEL_ID"],
            "reflection/Beam-1.0-501B-A23B-DFlash2",
        )
        self.assertNotEqual(
            values["SGLANG_MODEL_ID"],
            values["SGLANG_DRAFT_MODEL_ID"],
        )
        self.assertEqual(
            values["SGLANG_MODEL_COMMIT_SHA"],
            "1bee7db4502ea1b7d0bd8574855e55334d87554f",
        )
        self.assertEqual(
            values["VLLM_MODEL_ID"],
            "reflection/Beam-1.0-501B-A23B-NVFP4",
        )
        self.assertFalse(hyperpod_tool.is_placeholder(values["VLLM_IMAGE"]))

    def test_environment_overrides_file_without_executing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.env"
            path.write_text("BEAM_NAMESPACE=from-file\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"BEAM_NAMESPACE": "from-environment"},
                clear=False,
            ):
                values = hyperpod_tool.load_config([str(path)])
        self.assertEqual(values["BEAM_NAMESPACE"], "from-environment")

    def test_rejects_credentials_and_export_syntax(self) -> None:
        for content in (
            "AWS_ACCESS_KEY_ID=example\n",
            "export BEAM_NAMESPACE=default\n",
            "SERVICE_TOKEN=example\n",
        ):
            with self.subTest(content=content):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "config.env"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(hyperpod_tool.ToolError):
                        hyperpod_tool.parse_env_file(path)

    def test_all_manifests_render_without_placeholders(self) -> None:
        values = configured_values()
        for runtime in hyperpod_tool.RUNTIMES:
            with self.subTest(runtime=runtime):
                rendered = hyperpod_tool.render_manifest(runtime, values)
                self.assertNotIn("${", rendered)
                self.assertNotIn("REPLACE_WITH_", rendered)
                self.assertIn('namespace: "default"', rendered)

    def test_custom_runtime_manifests_do_not_enable_expert_parallelism(self) -> None:
        values = configured_values()
        for runtime in ("sglang", "vllm"):
            with self.subTest(runtime=runtime):
                rendered = hyperpod_tool.render_manifest(runtime, values)
                self.assertNotIn("--ep-size", rendered)
                self.assertNotIn("--enable-expert-parallel", rendered)

    def test_rejects_unpinned_image_and_short_commit(self) -> None:
        values = configured_values()
        values["VLLM_IMAGE"] = "ghcr.io/example/runtime:latest"
        with self.assertRaisesRegex(hyperpod_tool.ToolError, "pinned"):
            hyperpod_tool.render_manifest("vllm", values)

        values = configured_values()
        values["SGLANG_MODEL_COMMIT_SHA"] = "abc123"
        with self.assertRaisesRegex(hyperpod_tool.ToolError, "40-character"):
            hyperpod_tool.render_manifest("sglang", values)

    def test_rejects_invalid_sglang_memory_fraction(self) -> None:
        for value in ("0", "1.1", "not-a-number"):
            values = configured_values()
            values["SGLANG_MEM_FRACTION_STATIC"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    hyperpod_tool.ToolError,
                    "SGLANG_MEM_FRACTION_STATIC",
                ):
                    hyperpod_tool.render_manifest("sglang", values)

    def test_rejects_overlong_endpoint_and_multiline_environment_value(self) -> None:
        values = configured_values()
        values["JUMPSTART_ENDPOINT_NAME"] = "a" + ("-" * 100) + "b"
        with self.assertRaisesRegex(hyperpod_tool.ToolError, "at most 63"):
            hyperpod_tool.render_manifest("jumpstart", values)

        values = configured_values()
        values["SGLANG_MODEL_ID"] = "reflection/beam\nunexpected: value"
        with self.assertRaisesRegex(hyperpod_tool.ToolError, "single-line"):
            hyperpod_tool.render_manifest("sglang", values)

    def test_namespace_override_is_used_in_applied_manifest(self) -> None:
        values = configured_values()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.env"
            config.write_text(
                "\n".join(f"{key}={value}" for key, value in values.items()),
                encoding="utf-8",
            )
            calls: list[str] = []

            def fake_run(
                command: list[str],
                *,
                input_text: str | None = None,
                timeout: int = 60,
                check: bool = True,
            ) -> subprocess.CompletedProcess[str]:
                del command, timeout, check
                calls.append(input_text or "")
                return subprocess.CompletedProcess([], 0, "applied\n", "")

            with mock.patch.object(hyperpod_tool, "run_capture", side_effect=fake_run):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = hyperpod_tool.main(
                        [
                            "--config",
                            str(config),
                            "--namespace",
                            "team-a",
                            "deploy",
                            "jumpstart",
                            "--no-wait",
                        ]
                    )
        self.assertEqual(code, 0)
        self.assertIn('namespace: "team-a"', calls[0])


class StatusTests(unittest.TestCase):
    def test_reads_root_and_legacy_deployment_states(self) -> None:
        self.assertEqual(
            hyperpod_tool.deployment_state({"status": {"state": "DeploymentComplete"}}),
            "DeploymentComplete",
        )
        self.assertEqual(
            hyperpod_tool.deployment_state(
                {
                    "status": {
                        "deploymentStatus": {
                            "deploymentObjectOverallState": "DeploymentInProgress"
                        }
                    }
                }
            ),
            "DeploymentInProgress",
        )

    def test_selects_only_running_ready_pods_in_name_order(self) -> None:
        payload = {
            "items": [
                {
                    "metadata": {"name": "worker-b"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                },
                {
                    "metadata": {"name": "worker-a"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                },
                {
                    "metadata": {"name": "worker-c"},
                    "status": {
                        "phase": "Pending",
                        "conditions": [{"type": "Ready", "status": "False"}],
                    },
                },
            ]
        }
        self.assertEqual(
            hyperpod_tool.ready_pod_names(payload),
            ["worker-a", "worker-b"],
        )


class SafetyTests(unittest.TestCase):
    def test_sample_contains_no_committed_sensitive_or_staged_values(self) -> None:
        forbidden = {
            "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            "account ID": re.compile(r"\b\d{12}\b"),
            "staged S3 URI": re.compile("s3" + "://", re.IGNORECASE),
        }
        for path in hyperpod_tool.SAMPLE_ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in forbidden.items():
                with self.subTest(path=path.name, label=label):
                    self.assertIsNone(pattern.search(content))

    def test_release_values_are_not_embedded_in_manifests(self) -> None:
        release_keys = {
            "MODEL_ID",
            "MODEL_COMMIT_SHA",
            "IMAGE",
            "REASONING_PARSER",
            "TOOL_CALL_PARSER",
        }
        for runtime, spec in hyperpod_tool.RUNTIMES.items():
            content = (hyperpod_tool.SAMPLE_ROOT / spec.manifest).read_text(
                encoding="utf-8"
            )
            if runtime == "jumpstart":
                self.assertIn("${JUMPSTART_MODEL_ID}", content)
                continue
            prefix = runtime.upper()
            for suffix in release_keys:
                self.assertIn(f"${{{prefix}_{suffix}}}", content)
            if runtime == "sglang":
                self.assertIn("${SGLANG_DRAFT_MODEL_ID}", content)


class MainTests(unittest.TestCase):
    def test_custom_invocation_accepts_recommended_sampling_settings(self) -> None:
        parser = hyperpod_tool.build_parser()
        args = parser.parse_args(
            [
                "invoke",
                "vllm",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--temperature",
                "1.0",
                "--top-p",
                "0.7",
            ]
        )
        hyperpod_tool.validate_invocation_args(args)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.top_p, 0.7)

    def test_custom_invocation_rejects_invalid_top_p(self) -> None:
        parser = hyperpod_tool.build_parser()
        args = parser.parse_args(
            [
                "invoke",
                "vllm",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--top-p",
                "1.1",
            ]
        )
        with self.assertRaisesRegex(hyperpod_tool.ToolError, "--top-p"):
            hyperpod_tool.validate_invocation_args(args)

    def test_validate_reports_missing_values_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = hyperpod_tool.main(
                [
                    "--config",
                    str(hyperpod_tool.SAMPLE_ROOT / "config/deployment.env.example"),
                    "--config",
                    str(hyperpod_tool.SAMPLE_ROOT / "config/release.env.example"),
                    "validate",
                    "vllm",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("missing configuration for vllm", stderr.getvalue())

    def test_cleanup_requires_exact_resource_confirmation(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = hyperpod_tool.main(
                ["cleanup", "vllm", "--confirm-name", "wrong-name"]
            )
        self.assertEqual(code, 2)
        self.assertIn("reflection-beam-vllm", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
