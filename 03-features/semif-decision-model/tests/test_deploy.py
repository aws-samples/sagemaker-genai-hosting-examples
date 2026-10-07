import importlib
import io
import json
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

SAMPLE_ROOT = Path(__file__).parents[1]
DEPLOY_ROOT = SAMPLE_ROOT / "deploy"
sys.path.insert(0, str(DEPLOY_ROOT))

deploy_endpoint = importlib.import_module("deploy_endpoint")
package_model = importlib.import_module("package_model")
prepare_build = importlib.import_module("prepare_build")


def test_container_source_archive_contains_only_required_inputs(tmp_path):
    (tmp_path / "deploy").mkdir()
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (tmp_path / "server.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "deploy" / "buildspec.yml").write_text(
        "version: 0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "secret.txt").write_text("do not package", encoding="utf-8")

    archive_bytes = prepare_build.source_archive(tmp_path)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert sorted(archive.namelist()) == [
            "Dockerfile",
            "buildspec.yml",
            "server.py",
        ]
        assert "secret.txt" not in archive.namelist()


def test_container_source_archive_can_select_gpu_dockerfile(tmp_path):
    (tmp_path / "deploy").mkdir()
    (tmp_path / "Dockerfile.gpu").write_text("FROM cuda\n", encoding="utf-8")
    (tmp_path / "server.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "deploy" / "buildspec.yml").write_text(
        "version: 0.2\n",
        encoding="utf-8",
    )

    archive_bytes = prepare_build.source_archive(tmp_path, "Dockerfile.gpu")

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert archive.read("Dockerfile") == b"FROM cuda\n"


def test_container_source_archive_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing build inputs"):
        prepare_build.source_archive(tmp_path)


@pytest.mark.parametrize(
    "image_uri",
    [
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif:latest",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif@sha256:short",
        "semif@sha256:" + "z" * 64,
    ],
)
def test_deployment_requires_a_valid_digest_pinned_image(image_uri):
    with pytest.raises(ValueError, match="digest"):
        deploy_endpoint.validate_image_uri(image_uri)


def test_deployment_accepts_a_valid_digest_pinned_image():
    deploy_endpoint.validate_image_uri(
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif@sha256:" + "a" * 64
    )


@pytest.mark.parametrize(
    "model_data_url",
    [
        "https://example.com/model.tar.gz",
        "s3://sample-bucket/model.zip",
        "s3://sample-bucket/not-model.tar.gz",
    ],
)
def test_deploy_rejects_invalid_model_data_before_aws_calls(
    model_data_url,
    monkeypatch,
):
    fake_boto3 = types.SimpleNamespace(
        Session=lambda **_: pytest.fail("boto3 must not be called")
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    with pytest.raises(ValueError, match="model.tar.gz"):
        deploy_endpoint.deploy(
            region="us-east-1",
            name="semif-example",
            image_uri=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif@sha256:"
                + "a" * 64
            ),
            model_data_url=model_data_url,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.c6i.2xlarge",
        )


def test_deploy_calls_sagemaker_with_validated_arguments(monkeypatch):
    calls = []

    class FakeSageMaker:
        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs))

        def create_endpoint_config(self, **kwargs):
            calls.append(("create_endpoint_config", kwargs))

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs))

    fake_client = FakeSageMaker()
    fake_boto3 = types.SimpleNamespace(
        Session=lambda **kwargs: types.SimpleNamespace(
            client=lambda service: (
                fake_client
                if service == "sagemaker"
                else pytest.fail(f"unexpected service: {service}")
            )
        )
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    result = deploy_endpoint.deploy(
        region="us-east-1",
        name="semif-example",
        image_uri=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif@sha256:"
            + "a" * 64
        ),
        model_data_url="s3://sample-bucket/semif/model.tar.gz",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        instance_type="ml.c6i.2xlarge",
    )

    assert result == {
        "endpoint_name": "semif-example",
        "model_name": "semif-example",
        "endpoint_config_name": "semif-example",
        "region": "us-east-1",
        "instance_type": "ml.c6i.2xlarge",
        "gpu_layers": 0,
        "status": "Creating",
    }
    assert [operation for operation, _ in calls] == [
        "create_model",
        "create_endpoint_config",
        "create_endpoint",
    ]
    create_model = calls[0][1]
    assert create_model["ModelName"] == "semif-example"
    assert create_model["EnableNetworkIsolation"] is True
    assert create_model["PrimaryContainer"]["ModelDataUrl"].startswith("s3://")
    assert "@sha256:" in create_model["PrimaryContainer"]["Image"]
    environment = create_model["PrimaryContainer"]["Environment"]
    assert "SEMIF_THREADS" not in environment
    assert environment["SEMIF_MAX_REQUEST_TOKENS"] == "8192"
    assert environment["SEMIF_MAX_SHARED_DECISIONS"] == "8"
    assert environment["SEMIF_MAX_BODY_BYTES"] == "262144"


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.gpu"])
def test_child_images_preserve_the_sagemaker_root_user_contract(dockerfile):
    instructions = (SAMPLE_ROOT / dockerfile).read_text(encoding="utf-8").splitlines()

    assert not any(line.strip().startswith("USER ") for line in instructions)


def test_deploy_sets_explicit_thread_count(monkeypatch):
    calls = []

    class FakeSageMaker:
        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs))

        def create_endpoint_config(self, **kwargs):
            calls.append(("create_endpoint_config", kwargs))

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs))

    fake_client = FakeSageMaker()
    fake_boto3 = types.SimpleNamespace(
        Session=lambda **kwargs: types.SimpleNamespace(
            client=lambda service: (
                fake_client
                if service == "sagemaker"
                else pytest.fail(f"unexpected service: {service}")
            )
        )
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    deploy_endpoint.deploy(
        region="us-east-1",
        name="semif-example",
        image_uri=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif@sha256:"
            + "a" * 64
        ),
        model_data_url="s3://sample-bucket/semif/model.tar.gz",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        instance_type="ml.c6i.8xlarge",
        threads=32,
    )

    create_model = calls[0][1]
    assert create_model["PrimaryContainer"]["Environment"]["SEMIF_THREADS"] == "32"


def test_deploy_uses_ordered_instance_pool_and_gpu_offload(monkeypatch):
    calls = []

    class FakeSageMaker:
        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs))

        def create_endpoint_config(self, **kwargs):
            calls.append(("create_endpoint_config", kwargs))

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs))

    fake_client = FakeSageMaker()
    fake_boto3 = types.SimpleNamespace(
        Session=lambda **kwargs: types.SimpleNamespace(
            client=lambda service: (
                fake_client
                if service == "sagemaker"
                else pytest.fail(f"unexpected service: {service}")
            )
        )
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    result = deploy_endpoint.deploy(
        region="ap-northeast-2",
        name="semif-gpu",
        image_uri=(
            "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/semif@sha256:"
            + "a" * 64
        ),
        model_data_url="s3://sample-bucket/semif/model.tar.gz",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        instance_pools=[
            "ml.g6e.xlarge",
            "ml.g6.2xlarge",
            "ml.g5.2xlarge",
            "ml.g4dn.2xlarge",
        ],
        gpu_layers=-1,
        variant_instance_provision_timeout_seconds=1200,
    )

    assert result["gpu_layers"] == -1
    create_model = calls[0][1]
    assert create_model["PrimaryContainer"]["Environment"]["SEMIF_N_GPU_LAYERS"] == "-1"
    production_variant = calls[1][1]["ProductionVariants"][0]
    assert "InstanceType" not in production_variant
    assert production_variant["InstancePools"] == [
        {"InstanceType": "ml.g6e.xlarge", "Priority": 1},
        {"InstanceType": "ml.g6.2xlarge", "Priority": 2},
        {"InstanceType": "ml.g5.2xlarge", "Priority": 3},
        {"InstanceType": "ml.g4dn.2xlarge", "Priority": 4},
    ]


def test_deploy_uses_signed_fallback_when_sdk_model_lacks_instance_pools(
    monkeypatch,
):
    calls = []

    class ParamValidationError(Exception):
        pass

    class FakeSageMaker:
        meta = types.SimpleNamespace(
            endpoint_url="https://api.sagemaker.ap-northeast-2.amazonaws.com"
        )

        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs))

        def create_endpoint_config(self, **kwargs):
            calls.append(("sdk_create_endpoint_config", kwargs))
            raise ParamValidationError(
                'Unknown parameter in ProductionVariants[0]: "InstancePools"; '
                'Unknown parameter in ProductionVariants[0]: '
                '"VariantInstanceProvisionTimeoutInSeconds"'
            )

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs))

    fake_client = FakeSageMaker()
    fake_session = types.SimpleNamespace(
        client=lambda service: (
            fake_client
            if service == "sagemaker"
            else pytest.fail(f"unexpected service: {service}")
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(Session=lambda **kwargs: fake_session),
    )
    monkeypatch.setattr(
        deploy_endpoint,
        "_create_endpoint_config_raw",
        lambda **kwargs: calls.append(("raw_create_endpoint_config", kwargs)),
    )

    deploy_endpoint.deploy(
        region="ap-northeast-2",
        name="semif-gpu",
        image_uri=(
            "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/semif@sha256:"
            + "a" * 64
        ),
        model_data_url="s3://sample-bucket/semif/model.tar.gz",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        instance_pools=["ml.g6e.xlarge", "ml.g6.2xlarge"],
        gpu_layers=-1,
    )

    assert [operation for operation, _ in calls] == [
        "create_model",
        "sdk_create_endpoint_config",
        "raw_create_endpoint_config",
        "create_endpoint",
    ]


def test_deploy_does_not_mask_unrelated_endpoint_config_errors(monkeypatch):
    class FakeSageMaker:
        def create_model(self, **kwargs):
            pass

        def create_endpoint_config(self, **kwargs):
            raise RuntimeError("access denied")

    fake_client = FakeSageMaker()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(
            Session=lambda **kwargs: types.SimpleNamespace(
                client=lambda service: fake_client
            )
        ),
    )

    with pytest.raises(RuntimeError, match="access denied"):
        deploy_endpoint.deploy(
            region="ap-northeast-2",
            name="semif-gpu",
            image_uri=(
                "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/"
                "semif@sha256:" + "a" * 64
            ),
            model_data_url="s3://sample-bucket/semif/model.tar.gz",
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_pools=["ml.g6e.xlarge"],
            gpu_layers=-1,
        )


def test_deploy_rejects_fixed_type_and_instance_pool_together(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(Session=lambda **_: pytest.fail("boto3 must not be called")),
    )

    with pytest.raises(ValueError, match="exactly one"):
        deploy_endpoint.deploy(
            region="ap-northeast-2",
            name="semif-gpu",
            image_uri=(
                "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/semif@sha256:"
                + "a" * 64
            ),
            model_data_url="s3://sample-bucket/semif/model.tar.gz",
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g6e.xlarge",
            instance_pools=["ml.g6.2xlarge"],
        )


@pytest.mark.parametrize("threads", [0, -1])
def test_deploy_rejects_non_positive_thread_count(threads, monkeypatch):
    fake_boto3 = types.SimpleNamespace(
        Session=lambda **_: pytest.fail("boto3 must not be called")
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    with pytest.raises(ValueError, match="threads"):
        deploy_endpoint.deploy(
            region="us-east-1",
            name="semif-example",
            image_uri=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/semif@sha256:"
                + "a" * 64
            ),
            model_data_url="s3://sample-bucket/semif/model.tar.gz",
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.c6i.8xlarge",
            threads=threads,
        )


def test_model_bundle_uses_pinned_local_downloads_without_network(
    tmp_path,
    monkeypatch,
):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    gguf = downloads / package_model.GGUF_FILE
    gguf.write_bytes(b"local gguf")
    tokenizer = downloads / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(("file", kwargs))
        return str(gguf)

    def fake_snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(tokenizer)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=fake_hf_hub_download,
            snapshot_download=fake_snapshot_download,
        ),
    )
    monkeypatch.setattr(
        package_model,
        "GGUF_SHA256",
        package_model.sha256(gguf),
        raising=False,
    )
    output = tmp_path / "model.tar.gz"

    result = package_model.create_bundle(output, cache_dir=tmp_path / "cache")

    assert output.is_file()
    assert result["gguf"]["revision"] == package_model.GGUF_REVISION
    assert result["tokenizer"]["revision"] == package_model.TOKENIZER_REVISION
    assert all(call[1]["revision"] for call in calls)
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert package_model.GGUF_FILE in names
        assert "tokenizer/tokenizer.json" in names
        assert "tokenizer/tokenizer_config.json" in names
        manifest = json.load(archive.extractfile("model-manifest.json"))
    assert manifest["gguf"]["sha256"] == package_model.sha256(gguf)


def test_model_bundle_rejects_gguf_with_unexpected_sha256(
    tmp_path,
    monkeypatch,
):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    gguf = downloads / package_model.GGUF_FILE
    gguf.write_bytes(b"tampered gguf")
    tokenizer = downloads / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda **_: str(gguf),
            snapshot_download=lambda **_: str(tokenizer),
        ),
    )
    monkeypatch.setattr(
        package_model,
        "GGUF_SHA256",
        "0" * 64,
        raising=False,
    )
    output = tmp_path / "model.tar.gz"

    with pytest.raises(ValueError, match="SHA-256"):
        package_model.create_bundle(output, cache_dir=tmp_path / "cache")

    assert not output.exists()


def test_model_bundle_refuses_to_overwrite_existing_archive(tmp_path):
    output = tmp_path / "model.tar.gz"
    output.touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        package_model.create_bundle(output)
