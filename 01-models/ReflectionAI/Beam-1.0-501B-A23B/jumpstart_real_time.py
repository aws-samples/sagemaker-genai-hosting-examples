"""Deploy, invoke, smoke test, and clean up Beam through SageMaker JumpStart."""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from typing import Any

try:
    from sagemaker.jumpstart.model import JumpStartModel
except ModuleNotFoundError:
    JumpStartModel = None  # type: ignore[assignment,misc]


JUMPSTART_MODEL_IDS = {
    "bf16": "huggingface-llm-beam-1-0-501b-a23b",
    "fp8": "huggingface-llm-beam-1-0-501b-a23b-fp8",
    "nvfp4": "huggingface-llm-beam-1-0-501b-a23b-nvfp4",
}
DEFAULT_VARIANT = "nvfp4"
WAITER_DELAY_SECONDS = 15


def _resolve_model_id(variant: str) -> str:
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("variant must be one of: bf16, fp8, nvfp4.")
    normalized_variant = variant.strip().lower()
    try:
        model_id = JUMPSTART_MODEL_IDS[normalized_variant].strip()
    except KeyError:
        raise ValueError("variant must be one of: bf16, fp8, nvfp4.") from None
    if not model_id or model_id.startswith("REPLACE_WITH_"):
        raise ValueError(
            f"The {normalized_variant} variant must contain a supported "
            "Beam JumpStart model ID."
        )
    return model_id


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _resolve_endpoint_name(endpoint_name: str | None) -> str:
    if endpoint_name is None:
        return f"reflection-beam-{uuid.uuid4().hex[:12]}"
    resolved_name = endpoint_name.strip()
    if not resolved_name:
        raise ValueError("endpoint_name must be a non-empty string.")
    return resolved_name


def _is_missing_resource(error: Exception, resource_name: str) -> bool:
    response = getattr(error, "response", {})
    error_details = response.get("Error", {}) if isinstance(response, dict) else {}
    code = str(error_details.get("Code", ""))
    message = str(error_details.get("Message", "")).lower()
    return (
        code in {"ResourceNotFound", "ResourceNotFoundException", "ValidationException"}
        and resource_name.lower() in message
        and ("could not find" in message or "not found" in message)
    )


def deploy_endpoint(
    *,
    sagemaker_session: Any = None,
    endpoint_name: str | None = None,
    timeout_seconds: int = 900,
    variant: str = DEFAULT_VARIANT,
) -> Any:
    """Deploy the configured Beam JumpStart model to a real-time endpoint."""
    model_id = _resolve_model_id(variant)
    timeout = _require_positive_integer(timeout_seconds, "timeout_seconds")
    resolved_endpoint_name = _resolve_endpoint_name(endpoint_name)
    if JumpStartModel is None:
        raise RuntimeError(
            "The SageMaker Python SDK is not installed. Run: "
            "python -m pip install -r requirements.txt"
        )

    model = JumpStartModel(
        model_id=model_id,
        sagemaker_session=sagemaker_session,
    )
    predictor = None
    try:
        predictor = model.deploy(
            endpoint_name=resolved_endpoint_name,
            wait=False,
            model_data_download_timeout=timeout,
            container_startup_health_check_timeout=timeout,
        )
        setattr(predictor, "_reflection_beam_model", model)
        waiter = predictor.sagemaker_session.sagemaker_client.get_waiter(
            "endpoint_in_service"
        )
        waiter.wait(
            EndpointName=predictor.endpoint_name,
            WaiterConfig={
                "Delay": WAITER_DELAY_SECONDS,
                "MaxAttempts": max(1, math.ceil(timeout / WAITER_DELAY_SECONDS)),
            },
        )
        return predictor
    except BaseException:
        try:
            cleanup_endpoint(
                predictor,
                endpoint_name=resolved_endpoint_name,
                model=model,
                timeout_seconds=timeout,
            )
        except Exception as cleanup_error:
            raise RuntimeError(
                "Deployment did not complete and automatic cleanup failed for "
                f"endpoint {resolved_endpoint_name}. Check SageMaker before retrying."
            ) from cleanup_error
        raise


def _runtime_config(timeout_seconds: int) -> Any:
    try:
        from botocore.config import Config
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The SageMaker dependencies are not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from error
    return Config(
        connect_timeout=min(10, timeout_seconds),
        read_timeout=timeout_seconds,
        retries={"mode": "standard", "max_attempts": 3},
    )


def invoke_endpoint(
    predictor: Any,
    prompt: str,
    *,
    max_tokens: int = 64,
    timeout_seconds: int = 120,
) -> str:
    """Send one bounded chat request and return the assistant text."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    token_limit = _require_positive_integer(max_tokens, "max_tokens")
    request_timeout = _require_positive_integer(timeout_seconds, "timeout_seconds")

    runtime_client = predictor.sagemaker_session.boto_session.client(
        "sagemaker-runtime",
        config=_runtime_config(request_timeout),
    )
    response = runtime_client.invoke_endpoint(
        EndpointName=predictor.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": token_limit,
            }
        ).encode("utf-8"),
    )

    try:
        payload = json.loads(response["Body"].read())
        content = payload["choices"][0]["message"]["content"]
    except (AttributeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        raise RuntimeError("Endpoint returned a malformed response.") from None

    if not isinstance(content, str):
        raise RuntimeError("Endpoint response content must be text.")
    return content


def smoke_test(predictor: Any) -> str:
    """Run a small inference and require non-empty model output."""
    output = invoke_endpoint(
        predictor,
        "Reply with a short greeting.",
        max_tokens=64,
    )
    if not output.strip():
        raise RuntimeError("Endpoint smoke test returned empty output.")
    return output


def cleanup_endpoint(
    predictor: Any | None,
    *,
    endpoint_name: str | None = None,
    model: Any = None,
    timeout_seconds: int = 900,
) -> None:
    """Delete the endpoint before its endpoint configuration and model."""
    if predictor is None and model is None:
        return

    timeout = _require_positive_integer(timeout_seconds, "timeout_seconds")
    if predictor is not None:
        endpoint_name = predictor.endpoint_name
        model = model or getattr(predictor, "_reflection_beam_model", None)
        session = predictor.sagemaker_session
    else:
        session = getattr(model, "sagemaker_session", None)

    if session is None:
        raise RuntimeError("Cleanup requires a SageMaker session.")
    if endpoint_name is None or not endpoint_name.strip():
        raise RuntimeError("Cleanup requires the endpoint name.")

    endpoint_name = endpoint_name.strip()
    client = session.sagemaker_client
    endpoint_config_name = endpoint_name
    endpoint_exists = True
    endpoint_status = None

    try:
        endpoint_description = client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = (
            endpoint_description.get("EndpointConfigName") or endpoint_name
        )
        endpoint_status = endpoint_description.get("EndpointStatus")
    except Exception as error:
        if _is_missing_resource(error, "endpoint"):
            endpoint_exists = False
        else:
            raise RuntimeError(
                f"Could not inspect endpoint {endpoint_name} during cleanup."
            ) from error

    if endpoint_exists:
        try:
            if endpoint_status != "Deleting":
                client.delete_endpoint(EndpointName=endpoint_name)
            waiter = client.get_waiter("endpoint_deleted")
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={
                    "Delay": WAITER_DELAY_SECONDS,
                    "MaxAttempts": max(
                        1, math.ceil(timeout / WAITER_DELAY_SECONDS)
                    ),
                },
            )
        except Exception as error:
            if not _is_missing_resource(error, "endpoint"):
                raise RuntimeError(
                    f"Could not delete endpoint {endpoint_name}. "
                    "Its configuration and model were preserved."
                ) from error

    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except Exception as error:
        if not _is_missing_resource(error, "endpoint configuration"):
            raise RuntimeError(
                f"Endpoint {endpoint_name} was deleted, but endpoint configuration "
                f"{endpoint_config_name} could not be deleted."
            ) from error

    if model is not None:
        try:
            model.delete_model()
        except Exception as error:
            if not _is_missing_resource(error, "model"):
                raise RuntimeError(
                    f"Endpoint {endpoint_name} was deleted, but its model could not "
                    "be deleted."
                ) from error


def run_lifecycle(
    *,
    sagemaker_session: Any = None,
    endpoint_name: str | None = None,
    timeout_seconds: int = 900,
    variant: str = DEFAULT_VARIANT,
) -> str:
    """Deploy Beam, run a smoke test, and always clean up the endpoint."""
    _resolve_model_id(variant)
    predictor = None
    try:
        predictor = deploy_endpoint(
            sagemaker_session=sagemaker_session,
            endpoint_name=endpoint_name,
            timeout_seconds=timeout_seconds,
            variant=variant,
        )
        return smoke_test(predictor)
    finally:
        cleanup_endpoint(predictor)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Beam JumpStart real-time endpoint smoke-test lifecycle."
    )
    parser.add_argument("--endpoint-name")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--variant",
        choices=tuple(JUMPSTART_MODEL_IDS),
        default=DEFAULT_VARIANT,
        help="Beam checkpoint variant to deploy (default: nvfp4).",
    )
    args = parser.parse_args()

    try:
        output = run_lifecycle(
            endpoint_name=args.endpoint_name,
            timeout_seconds=args.timeout_seconds,
            variant=args.variant,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
