import importlib
import json
import sys
import types
from pathlib import Path

import pytest

DEPLOY_ROOT = Path(__file__).parents[1] / "deploy"
sys.path.insert(0, str(DEPLOY_ROOT))

deploy_standard_endpoint = importlib.import_module("deploy_standard_endpoint")
invoke_standard_endpoint = importlib.import_module("invoke_standard_endpoint")


def test_standard_deployment_requires_digest_pinned_image():
    with pytest.raises(ValueError, match="digest"):
        deploy_standard_endpoint.validate_image_uri(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/llama-cpp:latest"
        )


def test_standard_deployment_creates_sagemaker_resources(monkeypatch):
    calls = []

    class FakeSageMaker:
        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs))

        def create_endpoint_config(self, **kwargs):
            calls.append(("create_endpoint_config", kwargs))

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs))

    fake_client = FakeSageMaker()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(
            Session=lambda **kwargs: types.SimpleNamespace(
                client=lambda service: (
                    fake_client
                    if service == "sagemaker"
                    else pytest.fail(f"unexpected service: {service}")
                )
            )
        ),
    )

    result = deploy_standard_endpoint.deploy(
        region="us-east-1",
        name="standard-llamacpp",
        image_uri=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/llama-cpp@sha256:"
            + "a" * 64
        ),
        model_data_url="s3://sample-bucket/llamacpp/model.tar.gz",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        instance_type="ml.c6i.2xlarge",
        context_size=4096,
    )

    assert result["endpoint_name"] == "standard-llamacpp"
    assert [operation for operation, _ in calls] == [
        "create_model",
        "create_endpoint_config",
        "create_endpoint",
    ]
    container = calls[0][1]["PrimaryContainer"]
    assert calls[0][1]["EnableNetworkIsolation"] is True
    assert container["ModelDataUrl"] == "s3://sample-bucket/llamacpp/model.tar.gz"
    assert container["Environment"] == {"SM_LLAMA_CPP_CTX_SIZE": "4096"}
    assert not any(key.startswith("SEMIF_") for key in container["Environment"])


@pytest.mark.parametrize(
    "model_data_url",
    [
        "https://example.com/model.tar.gz",
        "s3://sample-bucket/model.zip",
        "s3://sample-bucket/not-model.tar.gz",
    ],
)
def test_standard_deployment_rejects_invalid_model_archive(model_data_url):
    with pytest.raises(ValueError, match="model.tar.gz"):
        deploy_standard_endpoint.deploy(
            region="us-east-1",
            name="standard-llamacpp",
            image_uri=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/llama-cpp@sha256:"
                + "a" * 64
            ),
            model_data_url=model_data_url,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.c6i.2xlarge",
        )


def test_standard_invocation_uses_openai_chat_completions_payload(monkeypatch):
    calls = []

    class FakeBody:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "account_access"}}]}
            ).encode()

    class FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            calls.append(kwargs)
            return {"Body": FakeBody()}

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(
            Session=lambda **kwargs: types.SimpleNamespace(
                client=lambda service: (
                    FakeRuntime()
                    if service == "sagemaker-runtime"
                    else pytest.fail(f"unexpected service: {service}")
                )
            )
        ),
    )

    output = invoke_standard_endpoint.invoke(
        region="us-east-1",
        endpoint_name="standard-llamacpp",
        prompt="Choose the support queue.",
        max_tokens=32,
    )

    request = calls[0]
    payload = json.loads(request["Body"])
    assert request["ContentType"] == "application/json"
    assert payload == {
        "messages": [{"role": "user", "content": "Choose the support queue."}],
        "max_tokens": 32,
        "stream": False,
    }
    assert output["choices"][0]["message"]["content"] == "account_access"
