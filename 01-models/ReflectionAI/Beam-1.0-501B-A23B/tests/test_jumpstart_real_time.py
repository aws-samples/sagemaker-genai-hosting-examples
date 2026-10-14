"""Executable specification for the ReflectionAI Beam JumpStart sample."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SAMPLE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_MODULE_PATH = SAMPLE_DIR / "jumpstart_real_time.py"


def _load_sample_module():
    """Load the sample without requiring the SageMaker SDK or AWS credentials."""
    if not SAMPLE_MODULE_PATH.is_file():
        raise AssertionError(
            f"Missing production module: {SAMPLE_MODULE_PATH}. "
            "Implement the JumpStart real-time lifecycle there."
        )

    jumpstart_model = Mock(name="JumpStartModel")
    sagemaker_module = types.ModuleType("sagemaker")
    jumpstart_module = types.ModuleType("sagemaker.jumpstart")
    model_module = types.ModuleType("sagemaker.jumpstart.model")
    model_module.JumpStartModel = jumpstart_model

    module_name = "reflection_beam_jumpstart_real_time"
    spec = importlib.util.spec_from_file_location(module_name, SAMPLE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load production module: {SAMPLE_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "sagemaker": sagemaker_module,
            "sagemaker.jumpstart": jumpstart_module,
            "sagemaker.jumpstart.model": model_module,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)

    return module, jumpstart_model


class JumpStartRealTimeContractTests(unittest.TestCase):
    class MissingResourceError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.response = {
                "Error": {
                    "Code": "ValidationException",
                    "Message": message,
                }
            }

    def test_exposes_the_three_published_checkpoint_variants(self):
        # Arrange / Act
        sample, _ = _load_sample_module()

        # Assert
        self.assertEqual(
            sample.JUMPSTART_MODEL_IDS,
            {
                "bf16": "huggingface-llm-beam-1-0-501b-a23b",
                "fp8": "huggingface-llm-beam-1-0-501b-a23b-fp8",
                "nvfp4": "huggingface-llm-beam-1-0-501b-a23b-nvfp4",
            },
        )
        self.assertEqual(sample.DEFAULT_VARIANT, "nvfp4")

    def test_source_defines_only_the_published_model_ids(self):
        # Arrange
        self.assertTrue(
            SAMPLE_MODULE_PATH.is_file(),
            f"Missing production module: {SAMPLE_MODULE_PATH}",
        )

        # Act
        source = SAMPLE_MODULE_PATH.read_text(encoding="utf-8")

        # Assert
        for model_id in (
            "huggingface-llm-beam-1-0-501b-a23b",
            "huggingface-llm-beam-1-0-501b-a23b-fp8",
            "huggingface-llm-beam-1-0-501b-a23b-nvfp4",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(source.count(f'"{model_id}"'), 1)
        self.assertNotRegex(
            source,
            r"""os\.(?:environ\.get|getenv)\(\s*["'][^"']*MODEL_ID["']""",
        )

    def test_source_contains_no_credentials_or_account_specific_resources(self):
        # Arrange
        self.assertTrue(
            SAMPLE_MODULE_PATH.is_file(),
            f"Missing production module: {SAMPLE_MODULE_PATH}",
        )

        # Act
        source = SAMPLE_MODULE_PATH.read_text(encoding="utf-8")

        # Assert
        forbidden_patterns = {
            "AWS access key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
            "12-digit AWS account ID": r"\b\d{12}\b",
            "hard-coded IAM ARN": r"\barn:(?:aws|aws-us-gov|aws-cn):iam::",
            "hard-coded S3 resource": r"\bs3://",
            "secret access key setting": r"(?i)aws_secret_access_key\s*=",
        }
        for description, pattern in forbidden_patterns.items():
            with self.subTest(description=description):
                self.assertIsNone(re.search(pattern, source), description)

    def test_deploy_uses_jumpstart_model_and_bounded_timeouts(self):
        # Arrange
        sample, jumpstart_model_class = _load_sample_module()
        predictor = Mock(name="predictor")
        model = jumpstart_model_class.return_value
        model.deploy.return_value = predictor
        session = Mock(name="sagemaker_session")
        predictor.sagemaker_session = session
        predictor.endpoint_name = "beam-smoke-test"

        # Act
        result = sample.deploy_endpoint(
            sagemaker_session=session,
            endpoint_name="beam-smoke-test",
            timeout_seconds=900,
            variant="fp8",
        )

        # Assert
        jumpstart_model_class.assert_called_once_with(
            model_id="huggingface-llm-beam-1-0-501b-a23b-fp8",
            sagemaker_session=session,
        )
        model.deploy.assert_called_once_with(
            endpoint_name="beam-smoke-test",
            wait=False,
            model_data_download_timeout=900,
            container_startup_health_check_timeout=900,
        )
        session.sagemaker_client.get_waiter.assert_called_once_with(
            "endpoint_in_service"
        )
        session.sagemaker_client.get_waiter.return_value.wait.assert_called_once_with(
            EndpointName="beam-smoke-test",
            WaiterConfig={"Delay": 15, "MaxAttempts": 60},
        )
        self.assertIs(result, predictor)

    def test_deploy_generates_endpoint_name_before_creating_resources(self):
        sample, jumpstart_model_class = _load_sample_module()
        predictor = Mock(name="predictor")
        predictor.endpoint_name = "reflection-beam-generated"
        jumpstart_model_class.return_value.deploy.return_value = predictor

        with (
            patch.object(
                sample,
                "_resolve_endpoint_name",
                return_value="reflection-beam-generated",
            ),
        ):
            sample.deploy_endpoint(timeout_seconds=900)

        jumpstart_model_class.return_value.deploy.assert_called_once_with(
            endpoint_name="reflection-beam-generated",
            wait=False,
            model_data_download_timeout=900,
            container_startup_health_check_timeout=900,
        )

    def test_deploy_fails_clearly_before_aws_call_when_model_id_is_unavailable(self):
        # Arrange
        sample, jumpstart_model_class = _load_sample_module()

        # Act / Assert
        with (
            patch.object(
                sample,
                "JUMPSTART_MODEL_IDS",
                {"nvfp4": "REPLACE_WITH_MODEL_ID"},
            ),
            self.assertRaisesRegex(
                ValueError,
                r"(?i)nvfp4 variant.*supported",
            ),
        ):
            sample.deploy_endpoint(
                sagemaker_session=Mock(name="sagemaker_session"),
                timeout_seconds=900,
            )
        jumpstart_model_class.assert_not_called()

    def test_deploy_cleans_up_after_interruption(self):
        sample, jumpstart_model_class = _load_sample_module()
        predictor = Mock(name="predictor")
        predictor.endpoint_name = "beam-smoke-test"
        model = jumpstart_model_class.return_value
        model.deploy.return_value = predictor
        waiter = predictor.sagemaker_session.sagemaker_client.get_waiter.return_value
        waiter.wait.side_effect = KeyboardInterrupt()

        with (
            patch.object(sample, "cleanup_endpoint") as cleanup,
            self.assertRaises(KeyboardInterrupt),
        ):
            sample.deploy_endpoint(
                endpoint_name="beam-smoke-test",
                timeout_seconds=900,
            )

        cleanup.assert_called_once_with(
            predictor,
            endpoint_name="beam-smoke-test",
            model=model,
            timeout_seconds=900,
        )

    def test_deploy_reports_cleanup_failure_instead_of_swallowing_it(self):
        sample, jumpstart_model_class = _load_sample_module()
        predictor = Mock(name="predictor")
        predictor.endpoint_name = "beam-smoke-test"
        model = jumpstart_model_class.return_value
        model.deploy.return_value = predictor
        waiter = predictor.sagemaker_session.sagemaker_client.get_waiter.return_value
        waiter.wait.side_effect = TimeoutError("endpoint creation timed out")

        with (
            patch.object(
                sample,
                "cleanup_endpoint",
                side_effect=RuntimeError("endpoint deletion failed"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "automatic cleanup failed.*beam-smoke-test",
            ),
        ):
            sample.deploy_endpoint(
                endpoint_name="beam-smoke-test",
                timeout_seconds=900,
            )

    def test_lifecycle_fails_before_aws_call_when_model_id_is_unavailable(self):
        # Arrange
        sample, jumpstart_model_class = _load_sample_module()

        # Act / Assert
        with (
            patch.object(
                sample,
                "JUMPSTART_MODEL_IDS",
                {"nvfp4": "REPLACE_WITH_MODEL_ID"},
            ),
            self.assertRaisesRegex(
                ValueError,
                r"(?i)nvfp4 variant.*supported",
            ),
        ):
            sample.run_lifecycle(
                sagemaker_session=Mock(name="sagemaker_session"),
                endpoint_name="beam-smoke-test",
                timeout_seconds=900,
            )
        jumpstart_model_class.assert_not_called()

    def test_deploy_rejects_unbounded_timeout_values(self):
        # Arrange
        sample, jumpstart_model_class = _load_sample_module()

        # Act / Assert
        for timeout_seconds in (None, 0, -1):
            with self.subTest(timeout_seconds=timeout_seconds):
                with self.assertRaisesRegex(ValueError, "timeout_seconds"):
                    sample.deploy_endpoint(
                        sagemaker_session=Mock(name="sagemaker_session"),
                        timeout_seconds=timeout_seconds,
                    )
        jumpstart_model_class.assert_not_called()

    def test_deploy_rejects_an_unknown_variant_before_aws_call(self):
        sample, jumpstart_model_class = _load_sample_module()

        with self.assertRaisesRegex(ValueError, "bf16, fp8, nvfp4"):
            sample.deploy_endpoint(
                sagemaker_session=Mock(name="sagemaker_session"),
                variant="int8",
            )

        jumpstart_model_class.assert_not_called()

    def test_invoke_sends_a_bounded_chat_request_and_returns_text(self):
        # Arrange
        sample, _ = _load_sample_module()
        predictor = Mock(name="predictor")
        predictor.endpoint_name = "beam-smoke-test"
        body = io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": "Beam is ready."}}]}
            ).encode("utf-8")
        )
        runtime_client = (
            predictor.sagemaker_session.boto_session.client.return_value
        )
        runtime_client.invoke_endpoint.return_value = {"Body": body}

        # Act
        with patch.object(sample, "_runtime_config", return_value="bounded-config"):
            result = sample.invoke_endpoint(
                predictor,
                "Reply with a short greeting.",
                max_tokens=64,
                timeout_seconds=120,
            )

        # Assert
        predictor.sagemaker_session.boto_session.client.assert_called_once_with(
            "sagemaker-runtime",
            config="bounded-config",
        )
        runtime_client.invoke_endpoint.assert_called_once_with(
            EndpointName="beam-smoke-test",
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Reply with a short greeting."}
                    ],
                    "max_tokens": 64,
                }
            ).encode("utf-8"),
        )
        self.assertEqual(result, "Beam is ready.")

    def test_invoke_rejects_empty_or_malformed_responses(self):
        # Arrange
        sample, _ = _load_sample_module()

        # Act / Assert
        for response in ({}, {"choices": []}, {"choices": [{"message": {}}]}):
            with self.subTest(response=response):
                predictor = Mock(name="predictor")
                predictor.endpoint_name = "beam-smoke-test"
                runtime_client = (
                    predictor.sagemaker_session.boto_session.client.return_value
                )
                runtime_client.invoke_endpoint.return_value = {
                    "Body": io.BytesIO(json.dumps(response).encode("utf-8"))
                }
                with (
                    patch.object(sample, "_runtime_config", return_value=Mock()),
                    self.assertRaisesRegex(RuntimeError, "response"),
                ):
                    sample.invoke_endpoint(predictor, "Hello", max_tokens=32)

    def test_smoke_test_requires_non_empty_model_output(self):
        # Arrange
        sample, _ = _load_sample_module()
        predictor = Mock(name="predictor")

        # Act / Assert
        with patch.object(sample, "invoke_endpoint", return_value="  "):
            with self.assertRaisesRegex(RuntimeError, "smoke"):
                sample.smoke_test(predictor)

    def test_cleanup_deletes_endpoint_before_configuration_and_model(self):
        sample, _ = _load_sample_module()
        predictor = Mock(name="predictor")
        predictor.endpoint_name = "beam-smoke-test"
        client = predictor.sagemaker_session.sagemaker_client
        client.describe_endpoint.return_value = {
            "EndpointConfigName": "beam-smoke-test-config",
            "EndpointStatus": "Creating",
        }
        waiter = client.get_waiter.return_value
        model = predictor._reflection_beam_model
        operations = Mock(name="cleanup_operations")
        operations.attach_mock(client.delete_endpoint, "delete_endpoint")
        operations.attach_mock(waiter.wait, "wait_for_endpoint_deleted")
        operations.attach_mock(
            client.delete_endpoint_config,
            "delete_endpoint_config",
        )
        operations.attach_mock(model.delete_model, "delete_model")

        sample.cleanup_endpoint(None)
        sample.cleanup_endpoint(predictor, timeout_seconds=900)

        client.describe_endpoint.assert_called_once_with(
            EndpointName="beam-smoke-test"
        )
        client.get_waiter.assert_called_once_with("endpoint_deleted")
        self.assertEqual(
            operations.mock_calls,
            [
                unittest.mock.call.delete_endpoint(
                    EndpointName="beam-smoke-test"
                ),
                unittest.mock.call.wait_for_endpoint_deleted(
                    EndpointName="beam-smoke-test",
                    WaiterConfig={"Delay": 15, "MaxAttempts": 60},
                ),
                unittest.mock.call.delete_endpoint_config(
                    EndpointConfigName="beam-smoke-test-config"
                ),
                unittest.mock.call.delete_model(),
            ],
        )

    def test_cleanup_preserves_dependencies_when_endpoint_deletion_fails(self):
        sample, _ = _load_sample_module()
        predictor = Mock(name="predictor")
        predictor.endpoint_name = "beam-smoke-test"
        client = predictor.sagemaker_session.sagemaker_client
        client.describe_endpoint.return_value = {
            "EndpointConfigName": "beam-smoke-test-config",
            "EndpointStatus": "Creating",
        }
        client.delete_endpoint.side_effect = RuntimeError("delete failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "Could not delete endpoint.*preserved",
        ):
            sample.cleanup_endpoint(predictor)

        client.delete_endpoint_config.assert_not_called()
        predictor._reflection_beam_model.delete_model.assert_not_called()

    def test_cleanup_removes_partial_resources_when_endpoint_is_absent(self):
        sample, jumpstart_model_class = _load_sample_module()
        model = jumpstart_model_class.return_value
        client = model.sagemaker_session.sagemaker_client
        client.describe_endpoint.side_effect = self.MissingResourceError(
            'Could not find endpoint "beam-smoke-test".'
        )

        sample.cleanup_endpoint(
            None,
            endpoint_name="beam-smoke-test",
            model=model,
        )

        client.delete_endpoint.assert_not_called()
        client.delete_endpoint_config.assert_called_once_with(
            EndpointConfigName="beam-smoke-test"
        )
        model.delete_model.assert_called_once_with()

    def test_help_works_without_installed_sdk(self):
        sample, _ = _load_sample_module()
        with (
            patch.object(sample, "JumpStartModel", None),
            patch.object(sys, "argv", ["jumpstart_real_time.py", "--help"]),
            self.assertRaises(SystemExit) as exit_context,
        ):
            sample.main()
        self.assertEqual(exit_context.exception.code, 0)

    def test_lifecycle_always_cleans_up_after_success(self):
        # Arrange
        sample, _ = _load_sample_module()
        predictor = Mock(name="predictor")
        session = Mock(name="sagemaker_session")

        # Act
        with (
            patch.object(sample, "deploy_endpoint", return_value=predictor) as deploy,
            patch.object(sample, "smoke_test", return_value="Beam is ready.") as smoke,
            patch.object(sample, "cleanup_endpoint") as cleanup,
        ):
            result = sample.run_lifecycle(
                sagemaker_session=session,
                endpoint_name="beam-smoke-test",
                timeout_seconds=900,
                variant="fp8",
            )

        # Assert
        deploy.assert_called_once_with(
            sagemaker_session=session,
            endpoint_name="beam-smoke-test",
            timeout_seconds=900,
            variant="fp8",
        )
        smoke.assert_called_once_with(predictor)
        cleanup.assert_called_once_with(predictor)
        self.assertEqual(result, "Beam is ready.")

    def test_lifecycle_always_cleans_up_after_smoke_failure(self):
        # Arrange
        sample, _ = _load_sample_module()
        predictor = Mock(name="predictor")
        expected_error = RuntimeError("smoke failed")

        # Act / Assert
        with (
            patch.object(sample, "deploy_endpoint", return_value=predictor),
            patch.object(sample, "smoke_test", side_effect=expected_error),
            patch.object(sample, "cleanup_endpoint") as cleanup,
        ):
            with self.assertRaisesRegex(RuntimeError, "smoke failed"):
                sample.run_lifecycle(
                    sagemaker_session=Mock(name="sagemaker_session"),
                    endpoint_name="beam-smoke-test",
                    timeout_seconds=900,
                )
        cleanup.assert_called_once_with(predictor)


if __name__ == "__main__":
    unittest.main()
