#!/usr/bin/env python3
"""Configure CodeBuild and start one pinned model packaging build."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def source_archive(buildspec: Path) -> bytes:
    if not buildspec.is_file():
        raise FileNotFoundError(buildspec)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(buildspec, "model-buildspec.yml")
    return buffer.getvalue()


def start_build(
    *,
    region: str,
    buildspec: Path,
    bucket: str,
    source_key: str,
    project_name: str,
    codebuild_role_arn: str,
    model_s3_uri: str,
) -> dict:
    import boto3

    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    codebuild = session.client("codebuild")
    s3.put_object(
        Bucket=bucket,
        Key=source_key,
        Body=source_archive(buildspec),
        ContentType="application/zip",
    )
    project = {
        "name": project_name,
        "source": {
            "type": "S3",
            "location": f"{bucket}/{source_key}",
            "buildspec": "model-buildspec.yml",
        },
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "LINUX_CONTAINER",
            "computeType": "BUILD_GENERAL1_MEDIUM",
            "image": "aws/codebuild/standard:7.0",
            "privilegedMode": False,
            "environmentVariables": [
                {"name": "MODEL_S3_URI", "value": model_s3_uri},
            ],
        },
        "serviceRole": codebuild_role_arn,
        "timeoutInMinutes": 60,
    }
    existing = codebuild.batch_get_projects(names=[project_name])["projects"]
    if existing:
        codebuild.update_project(**project)
    else:
        codebuild.create_project(**project)
    build = codebuild.start_build(projectName=project_name)["build"]
    return {
        "build_id": build["id"],
        "project_name": project_name,
        "source_s3_uri": f"s3://{bucket}/{source_key}",
        "model_s3_uri": model_s3_uri,
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--buildspec",
        type=Path,
        default=Path(__file__).with_name("model-buildspec.yml"),
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--source-key", default=f"semif-llamacpp/model-source-{timestamp}.zip"
    )
    parser.add_argument("--project-name", default="semif-model-package-build")
    parser.add_argument("--codebuild-role-arn", required=True)
    parser.add_argument("--model-s3-uri", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            start_build(
                region=args.region,
                buildspec=args.buildspec,
                bucket=args.bucket,
                source_key=args.source_key,
                project_name=args.project_name,
                codebuild_role_arn=args.codebuild_role_arn,
                model_s3_uri=args.model_s3_uri,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
