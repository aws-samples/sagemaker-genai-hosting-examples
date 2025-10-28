#!/usr/bin/env bash
set -x
# This script builds a Docker image and pushes it to ECR for use with SageMaker
# Arguments: repo-name, version-tag, region, account

reponame=$1
versiontag=$2
regionname=$3
account=$4

# Validate input parameters
if [ "$reponame" == "" ] || [ "$versiontag" == "" ] || [ "$regionname" == "" ] || [ "$account" == "" ]
then
   echo "Usage: $0 <repo-name> <version-tag> <region> <account>"
   exit 1
fi

# Verify AWS CLI access and ECR permissions
echo "Verifying AWS credentials and ECR permissions..."
aws ecr get-authorization-token > /dev/null 2>&1
if [ $? -ne 0 ]; then
   echo "Error: Unable to get ECR authorization token. Check AWS credentials and permissions."
   exit 1
fi

fullname="${account}.dkr.ecr.${regionname}.amazonaws.com/${reponame}:${versiontag}"

# Check if repository exists in ECR, create if it doesn't
echo "Checking ECR repository..."
aws ecr describe-repositories --repository-names "${reponame}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
   echo "Creating ECR repository ${reponame}..."
   aws ecr create-repository --repository-name "${reponame}" > /dev/null
   if [ $? -ne 0 ]; then
       echo "Error: Failed to create ECR repository"
       exit 1
   fi
fi

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "$regionname" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${regionname}".amazonaws.com
if [ $? -ne 0 ]; then
   echo "Error: ECR login failed"
   exit 1
fi

# Build docker image
echo "Building docker image..."
pwd
docker build -f Dockerfile -t "${reponame}" .
if [ $? -ne 0 ]; then
   echo "Error: Docker build failed"
   exit 1
fi

# Tag image
echo "Tagging docker image..."
docker tag "${reponame}" "${fullname}"
if [ $? -ne 0 ]; then
   echo "Error: Docker tag failed"
   exit 1
fi

# Push to ECR
echo "Pushing image to ECR..."
docker push "${fullname}"
if [ $? -ne 0 ]; then
   echo "Error: Docker push failed"
   exit 1
fi

# Save image URI to file
echo "Saving image URI to file..."
echo "${fullname}" > dockerfile-image.txt
if [ $? -ne 0 ]; then
   echo "Error: Failed to save image URI to file"
   exit 1
fi

# # Clean up local images
# echo "Cleaning up local images..."
# docker rmi ${reponame}
# docker rmi ${fullname}

echo "Successfully built and pushed image: ${fullname}"
echo "Image URI saved to: dockerfile-image.txt"