from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text(encoding="utf-8")
REFERENCES = {
    path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "references").rglob("*.md"))
}
ALL_GUIDANCE = "\n".join((SKILL, *REFERENCES.values()))
EVALS = json.loads((ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))


class SkillContractTests(unittest.TestCase):
    def test_is_general_with_reflection_beam_as_an_example(self) -> None:
        self.assertIn("generative AI models on Amazon SageMaker", SKILL)
        self.assertIn("Reflection Beam worked example", SKILL)
        self.assertNotEqual("reflection-beam-deploy", ROOT.name)

    def test_starts_with_required_deployment_inputs(self) -> None:
        opening = SKILL.split("## Make the deployment decision", maxsplit=1)[0]
        for phrase in (
            "Model and sample",
            "Workload",
            "Hardware",
            "Capacity product",
            "Serving path",
            "Operating limits",
            "Existing infrastructure",
        ):
            self.assertIn(phrase, opening)

    def test_covers_all_sagemaker_serving_paths(self) -> None:
        for phrase in (
            "JumpStart real-time endpoint",
            "JumpStart on HyperPod Inference",
            "Custom runtime on HyperPod Inference",
        ):
            self.assertIn(phrase, ALL_GUIDANCE)

    def test_requires_cost_expiry_and_cleanup_controls(self) -> None:
        for phrase in (
            "steady and peak GPUs",
            "Maximum spend",
            "non-production TTL",
            "Cleanup owner",
            "timeout is unresolved",
        ):
            self.assertIn(phrase, ALL_GUIDANCE)

    def test_training_plan_is_fail_closed(self) -> None:
        for phrase in (
            "cannot be cancelled",
            "upfront price",
            "`unknown` cost is not acceptable",
            "offering ID",
            "Availability Zone",
        ):
            self.assertIn(phrase, ALL_GUIDANCE)

    def test_benchmark_requires_preflight_and_stop_conditions(self) -> None:
        benchmarking = REFERENCES["references/benchmarking.md"]
        for phrase in (
            "total requests",
            "maximum input and output volume",
            "expected GPU-hours and cost",
            "stop conditions",
            "explicit confirmation",
        ):
            self.assertIn(phrase, benchmarking)

    def test_uses_existing_samples_as_sources_of_truth(self) -> None:
        example = REFERENCES["references/examples/reflection-beam.md"]
        self.assertIn(
            "01-models/ReflectionAI/Beam-1.0-501B-A23B/README.md",
            example,
        )
        self.assertIn("SageMakerHyperpod/reflection-beam/README.md", example)
        self.assertIn("source of truth", SKILL)

    def test_does_not_duplicate_release_identifiers(self) -> None:
        lower = ALL_GUIDANCE.lower()
        for forbidden in (
            "huggingface-llm-beam",
            "sha256:",
            "dflash_model_commit_sha",
            "vllm_image=",
            "sglang_image=",
            "reflection-oss",
            "oracle dataset",
            "beam verifier",
        ):
            self.assertNotIn(forbidden, lower)

    def test_initial_contribution_is_instruction_only(self) -> None:
        self.assertFalse((ROOT / "scripts").exists())

    def test_evals_cover_high_risk_behaviors(self) -> None:
        self.assertEqual(EVALS["version"], 1)
        cases = EVALS["cases"]
        self.assertGreaterEqual(len(cases), 6)
        names = {case["name"] for case in cases}
        self.assertTrue(
            {
                "training plan purchase",
                "benchmark load",
                "cleanup timeout",
                "Reflection Beam example",
            }.issubset(names)
        )
        for case in cases:
            self.assertTrue(case["prompt"])
            self.assertTrue(case["expected_behaviors"])
            self.assertTrue(case["forbidden_behaviors"])


if __name__ == "__main__":
    unittest.main()
