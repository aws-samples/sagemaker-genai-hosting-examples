#!/usr/bin/env bash

# This script shows how to build the docker image and push it to ECR to be ready for use
# by SageMaker.

# The arguments to this script are the image name and dockerfile name (optional). The image name will be used
# as the image on the local machine and combined with the account and region to form the repository name for ECR.
image=$1
filename=${2:-Dockerfile}

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name> [dockerfile-name]"
    exit 1
fi

# Determine which container tool to use
if command -v docker &> /dev/null; then
    CONTAINER_TOOL="docker"
else
    CONTAINER_TOOL="finch"
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Authenticate to AWS DLC ECR registry
aws ecr get-login-password --region "${region}" | ${CONTAINER_TOOL} login --username AWS --password-stdin 763104351884.dkr.ecr."${region}".amazonaws.com

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | ${CONTAINER_TOOL} login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the container image locally with the image name and then push it to ECR
${CONTAINER_TOOL} build  -t ${image} -f ${filename} . --no-cache
${CONTAINER_TOOL} tag ${image} ${fullname}
${CONTAINER_TOOL} push ${fullname}