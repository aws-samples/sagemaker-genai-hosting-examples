#!/usr/bin/env python3
"""Package source, configure CodeBuild, and start one container build."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def source_archive(root: Path, dockerfile: str = "Dockerfile") -> bytes:
    if Path(dockerfile).name != dockerfile:
        raise ValueError("dockerfile must be a file in the sample root")
    required = (dockerfile, "server.py", "deploy/buildspec.yml")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing build inputs: {missing}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / dockerfile, "Dockerfile")
        archive.write(root / "server.py", "server.py")
        archive.write(root / "deploy/buildspec.yml", "buildspec.yml")
    return buffer.getvalue()


def start_build(
    *,
    region: str,
    root: Path,
    bucket: str,
    source_key: str,
    project_name: str,
    codebuild_role_arn: str,
    repository_name: str,
    image_tag: str,
    dockerfile: str = "Dockerfile",
) -> dict:
    import boto3

    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    ecr = session.client("ecr")
    codebuild = session.client("codebuild")
    account_id = session.client("sts").get_caller_identity()["Account"]
    try:
        ecr.describe_repositories(repositoryNames=[repository_name])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(
            repositoryName=repository_name,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
        )
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
    s3.put_object(
        Bucket=bucket,
        Key=source_key,
        Body=source_archive(root, dockerfile),
        ContentType="application/zip",
    )
    project = {
        "name": project_name,
        "source": {
            "type": "S3",
            "location": f"{bucket}/{source_key}",
            "buildspec": "buildspec.yml",
        },
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "LINUX_CONTAINER",
            "computeType": "BUILD_GENERAL1_MEDIUM",
            "image": "aws/codebuild/amazonlinux-x86_64-standard:5.0",
            "privilegedMode": True,
            "environmentVariables": [
                {"name": "ECR_REPOSITORY_URI", "value": ecr_uri},
                {"name": "IMAGE_TAG", "value": image_tag},
            ],
        },
        "serviceRole": codebuild_role_arn,
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
        "expected_image_tag": f"{ecr_uri}:{image_tag}",
        "dockerfile": dockerfile,
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--source-key", default=f"semif-llamacpp/build-source-{timestamp}.zip"
    )
    parser.add_argument("--project-name", default="semif-llamacpp-dlc-build")
    parser.add_argument("--codebuild-role-arn", required=True)
    parser.add_argument("--repository-name", default="semif-llamacpp-dlc")
    parser.add_argument("--image-tag", default=f"sample-{timestamp}")
    parser.add_argument(
        "--dockerfile",
        choices=("Dockerfile", "Dockerfile.gpu"),
        default="Dockerfile",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            start_build(
                region=args.region,
                root=args.root,
                bucket=args.bucket,
                source_key=args.source_key,
                project_name=args.project_name,
                codebuild_role_arn=args.codebuild_role_arn,
                repository_name=args.repository_name,
                image_tag=args.image_tag,
                dockerfile=args.dockerfile,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
