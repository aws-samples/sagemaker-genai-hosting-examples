# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build the decision-model serving image with AWS CodeBuild and push it to Amazon ECR.

    python build_image.py --region us-east-1 --bucket <artifact-bucket> \\
        --codebuild-role-arn arn:aws:iam::<account>:role/<codebuild-role>

Prints the digest-pinned image URI. The notebooks call build_image() directly.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES = ("Dockerfile", "serve", "serve.py", "requirements.txt", "buildspec.yml")


def source_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o755 if name == "serve" else 0o644) << 16
            archive.writestr(info, (ROOT / name).read_bytes())
    return buffer.getvalue()


def ensure_bucket(s3, bucket: str, region: str) -> None:
    """Create the artifact bucket, with public access blocked, if it does not exist yet."""
    from botocore.exceptions import ClientError

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        args = {"Bucket": bucket}
        if region != "us-east-1":
            args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**args)
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
            },
        )


def build_image(*, region: str, bucket: str, codebuild_role_arn: str,
                repository_name: str = "decision-model-serving",
                project_name: str = "decision-model-serving-build",
                image_tag: str | None = None, poll_seconds: int = 20) -> str:
    """Start the build, wait for it, and return the image URI pinned by digest."""
    import boto3

    session = boto3.Session(region_name=region)
    s3, ecr, codebuild = session.client("s3"), session.client("ecr"), session.client("codebuild")
    account_id = session.client("sts").get_caller_identity()["Account"]
    image_tag = image_tag or datetime.now(timezone.utc).strftime("pt214-%Y%m%d%H%M%S")
    try:
        ecr.describe_repositories(repositoryNames=[repository_name])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=repository_name,
                              imageScanningConfiguration={"scanOnPush": True},
                              encryptionConfiguration={"encryptionType": "AES256"})
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
    source_key = f"decision-model-serving/build-source-{image_tag}.zip"
    ensure_bucket(s3, bucket, region)
    s3.put_object(Bucket=bucket, Key=source_key, Body=source_archive(), ContentType="application/zip")
    try:
        project = {
            "name": project_name,
            "source": {"type": "S3", "location": f"{bucket}/{source_key}", "buildspec": "buildspec.yml"},
            "artifacts": {"type": "NO_ARTIFACTS"},
            "environment": {
                "type": "LINUX_CONTAINER",
                # The PyTorch DLC base is about 10 GB compressed; LARGE builds it in about 7 minutes.
                "computeType": "BUILD_GENERAL1_LARGE",
                "image": "aws/codebuild/amazonlinux-x86_64-standard:5.0",
                "privilegedMode": True,
                "environmentVariables": [
                    {"name": "ECR_REPOSITORY_URI", "value": ecr_uri},
                    {"name": "IMAGE_TAG", "value": image_tag},
                ],
            },
            "serviceRole": codebuild_role_arn,
        }
        if codebuild.batch_get_projects(names=[project_name])["projects"]:
            codebuild.update_project(**project)
        else:
            codebuild.create_project(**project)
        build_id = codebuild.start_build(projectName=project_name)["build"]["id"]
        while True:
            build = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
            if build["buildStatus"] != "IN_PROGRESS":
                break
            time.sleep(poll_seconds)
    finally:
        s3.delete_object(Bucket=bucket, Key=source_key)
    if build["buildStatus"] != "SUCCEEDED":
        raise RuntimeError(f"Image build {build_id} ended {build['buildStatus']}; see its CloudWatch logs.")
    digest = ecr.describe_images(repositoryName=repository_name,
                                 imageIds=[{"imageTag": image_tag}])["imageDetails"][0]["imageDigest"]
    return f"{ecr_uri}@{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--codebuild-role-arn", required=True)
    parser.add_argument("--repository-name", default="decision-model-serving")
    parser.add_argument("--image-tag")
    args = parser.parse_args()
    print(json.dumps({"image_uri": build_image(
        region=args.region, bucket=args.bucket, codebuild_role_arn=args.codebuild_role_arn,
        repository_name=args.repository_name, image_tag=args.image_tag)}, indent=2))


if __name__ == "__main__":
    main()
