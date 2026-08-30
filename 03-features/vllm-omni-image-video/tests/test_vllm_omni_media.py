import base64
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from vllm_omni_media import (
    DeploymentState,
    build_image_payload,
    build_video_multipart,
    container_image_uri,
    create_endpoint,
    decode_image_response,
    image_data_url,
    load_state,
    parse_s3_uri,
    save_state,
    submit_video,
    validate_mp4,
    wait_for_s3_object,
)


def test_container_image_uri_uses_pinned_dlc_release():
    assert container_image_uri("us-east-1") == (
        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
        "vllm:omni-sagemaker-cuda-v1.6"
    )


def test_build_image_payload_uses_flux_klein_defaults():
    payload = build_image_payload("A lighthouse above a stormy sea", seed=7)

    assert payload == {
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "prompt": "A lighthouse above a stormy sea",
        "size": "1024x1024",
        "num_inference_steps": 4,
        "seed": 7,
    }


def test_decode_image_response_returns_png_bytes():
    expected = b"\x89PNG\r\n\x1a\nimage"
    response = {"data": [{"b64_json": base64.b64encode(expected).decode("ascii")}]}

    assert decode_image_response(json.dumps(response).encode("utf-8")) == expected


def test_decode_image_response_rejects_missing_image():
    with pytest.raises(ValueError, match="b64_json"):
        decode_image_response(b'{"data": []}')


def test_image_data_url_encodes_png():
    assert image_data_url(b"png") == "data:image/png;base64,cG5n"


def test_build_video_multipart_contains_reference_and_controls():
    body, content_type = build_video_multipart(
        prompt="Slow camera push-in while clouds move",
        image_bytes=b"png",
        width=480,
        height=320,
        num_frames=17,
        fps=8,
        steps=4,
        seed=9,
        boundary="test-boundary",
    )

    assert content_type == "multipart/form-data; boundary=test-boundary"
    text = body.decode("utf-8")
    assert 'name="model"\r\n\r\nWan-AI/Wan2.1-VACE-1.3B-diffusers' in text
    assert 'name="prompt"\r\n\r\nSlow camera push-in while clouds move' in text
    assert 'name="image_reference"\r\n\r\n{"image_url":"data:image/png;base64,cG5n"}' in text
    assert 'name="width"\r\n\r\n480' in text
    assert 'name="height"\r\n\r\n320' in text
    assert 'name="num_frames"\r\n\r\n17' in text
    assert 'name="fps"\r\n\r\n8' in text
    assert 'name="num_inference_steps"\r\n\r\n4' in text
    assert 'name="seed"\r\n\r\n9' in text
    assert body.endswith(b"--test-boundary--\r\n")


def test_create_video_endpoint_uses_async_inference_and_gpu_ami():
    class FakeSageMaker:
        def __init__(self):
            self.calls = {}

        def create_model(self, **kwargs):
            self.calls["model"] = kwargs

        def create_endpoint_config(self, **kwargs):
            self.calls["config"] = kwargs

        def create_endpoint(self, **kwargs):
            self.calls["endpoint"] = kwargs

    client = FakeSageMaker()

    create_endpoint(
        client,
        model_name="video-model",
        endpoint_config_name="video-config",
        endpoint_name="video-endpoint",
        role_arn="arn:aws:iam::111122223333:role/SageMakerRole",
        image_uri=container_image_uri("us-east-1"),
        model_id="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        instance_type="ml.g6e.xlarge",
        startup_timeout_seconds=3600,
        async_output_path="s3://example-bucket/outputs/",
    )

    assert client.calls["model"]["PrimaryContainer"]["Environment"] == {
        "SM_VLLM_MODEL": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "SM_VLLM_VAE_USE_TILING": "true",
    }
    variant = client.calls["config"]["ProductionVariants"][0]
    assert variant["InstanceType"] == "ml.g6e.xlarge"
    assert variant["InferenceAmiVersion"] == (
        "al2023-ami-sagemaker-inference-gpu-4-1"
    )
    assert client.calls["config"]["AsyncInferenceConfig"] == {
        "OutputConfig": {"S3OutputPath": "s3://example-bucket/outputs/"},
        "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
    }
    assert client.calls["endpoint"] == {
        "EndpointName": "video-endpoint",
        "EndpointConfigName": "video-config",
    }


def test_submit_video_uploads_multipart_request_before_async_invocation():
    class FakeS3:
        def __init__(self):
            self.request = None

        def put_object(self, **kwargs):
            self.request = kwargs

    class FakeRuntime:
        def __init__(self):
            self.request = None

        def invoke_endpoint_async(self, **kwargs):
            self.request = kwargs
            return {
                "OutputLocation": "s3://example-bucket/outputs/result.out",
                "FailureLocation": "s3://example-bucket/failures/result.err",
            }

    state = DeploymentState(
        region="us-east-1",
        bucket="example-bucket",
        prefix="vllm-omni-media",
        image_model_name="image-model",
        image_endpoint_config_name="image-config",
        image_endpoint_name="image-endpoint",
        video_model_name="video-model",
        video_endpoint_config_name="video-config",
        video_endpoint_name="video-endpoint",
    )
    s3 = FakeS3()
    runtime = FakeRuntime()

    output_uri, failure_uri, request_key = submit_video(
        runtime,
        s3,
        state,
        "Move the clouds slowly",
        b"png",
    )

    assert output_uri == "s3://example-bucket/outputs/result.out"
    assert failure_uri == "s3://example-bucket/failures/result.err"
    assert request_key.startswith("vllm-omni-media/requests/")
    assert s3.request["Bucket"] == "example-bucket"
    assert s3.request["Key"] == request_key
    assert s3.request["ContentType"].startswith("multipart/form-data; boundary=")
    assert runtime.request == {
        "EndpointName": "video-endpoint",
        "InputLocation": f"s3://example-bucket/{request_key}",
        "ContentType": s3.request["ContentType"],
        "Accept": "video/mp4",
        "CustomAttributes": "route=/v1/videos/sync",
    }


def test_wait_for_s3_object_surfaces_async_failure():
    class FakeBody:
        def __init__(self, value):
            self.value = value

        def read(self):
            return self.value

    class FakeS3:
        def get_object(self, *, Bucket, Key):
            if Key == "outputs/result.out":
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            return {"Body": FakeBody(b'{"error":"model failed"}')}

    with pytest.raises(RuntimeError, match="model failed"):
        wait_for_s3_object(
            FakeS3(),
            "s3://example-bucket/outputs/result.out",
            failure_uri="s3://example-bucket/failures/result.err",
            timeout_seconds=1,
            poll_seconds=0,
        )


def test_state_round_trip(tmp_path: Path):
    state = DeploymentState(
        region="us-east-1",
        bucket="example-bucket",
        prefix="vllm-omni-media",
        image_model_name="image-model",
        image_endpoint_config_name="image-config",
        image_endpoint_name="image-endpoint",
        video_model_name="video-model",
        video_endpoint_config_name="video-config",
        video_endpoint_name="video-endpoint",
    )
    path = tmp_path / "state.json"

    save_state(state, path)

    assert load_state(path) == state


def test_parse_s3_uri():
    assert parse_s3_uri("s3://example-bucket/path/to/output.mp4") == (
        "example-bucket",
        "path/to/output.mp4",
    )


def test_validate_mp4_accepts_iso_base_media_header():
    validate_mp4(b"\x00\x00\x00\x18ftypisom")


def test_validate_mp4_rejects_non_video_response():
    with pytest.raises(ValueError, match="not an MP4"):
        validate_mp4(b'{"error":"generation failed"}')


@pytest.mark.parametrize("uri", ["https://example.com/file", "s3://bucket", "s3:///key"])
def test_parse_s3_uri_rejects_invalid_values(uri):
    with pytest.raises(ValueError, match="S3 URI"):
        parse_s3_uri(uri)
