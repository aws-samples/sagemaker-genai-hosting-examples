# Bidirectional Streaming with Amazon SageMaker Endpoints

**Building Interactive AI Applications with Real-Time Bidirectional Communication**

This repository contains a comprehensive Jupyter notebook demonstrating Amazon SageMaker's new bidirectional streaming capabilities. The implementation showcases real-time, two-way communication between applications and machine learning models.

## 🚀 Features

- **Complete Implementation**: End-to-end bidirectional streaming example with SageMaker
- **Container Support**: Streaming-enabled sample Docker container with required labels


## 📋 Prerequisites

- AWS Account with SageMaker permissions
- Docker installed locally
- Python 3.12+
- Basic understanding of Amazon SageMaker and Docker

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment (recommended)
python -m venv sagemaker-streaming-env
source sagemaker-streaming-env/bin/activate  # On Windows: sagemaker-streaming-env\Scripts\activate

# Install SageMaker Runtime AWS SDK for Python 
pip install aws-sdk-sagemaker-runtime-http2
```

### 2. Configure AWS credentials:
The application uses environment variables for AWS authentication. Set these before running the application:

export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"

### 3. Create a BYO container 

This section creates a simple container with bidirectional streaming capability. For the purpose of demonstrating the feature, this container simply echos back the input chunk as its output stream.

#### Container Contract
For a container to be compatible with Bidirectional Streaming on SageMaker endpoints, it must follow the follow contract
1. Follow existing health checks as decribed in https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests
2. To receive bidirectional streaming inference requests, the container must support:
  1. Listening on port 8080 via a websocket connection at /invocations-bidirectional-stream.
  2. Respond to ping frames with pong frames (details in RFC 6455)
  3. Containers must accept socket connection requests within 250 ms.

#### 📁 Repository Structure
```
BidiBlogNotebook/
├── README.md                                           # This file
├── create_sagemaker_endpoint.py                        # Create SageMaker resources
├── cleanup_sagemaker_resources.py                      # Cleanup SageMaker resources
├── sagemaker-byo-bidi-invoke.py                        # Invoke SageMaker with Bidirectional Streaming API
├── container/                                          # Streaming container source
│   ├── Dockerfile                                      # Container definition
│   ├── app.py                                          # FastAPI streaming app
│   └── requirements.txt                                # Container dependencies
└── python-sdk-new/                                     # Preview SDK
    └── sagemaker_runtime_http2-0.1.0-py3-none-any.whl
```

Now we will build the container as below:

```
container_name="sagemaker-bidirectional-streaming"
container_tag="latest"

cd container

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

container_image_uri="${account}.dkr.ecr.${region}.amazonaws.com/${container_name}:${container_tag}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${container_name}" --region "${region}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${container_name}" --region "${region}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com/${container_name}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build --platform linux/amd64 --provenance=false -t ${container_name} .
docker tag ${container_name} ${container_image_uri}

docker push ${container_image_uri}
```

### 4. Create SageMaker Endpoint
The script creates a SageMaker endpoint using the test container created above. Additionally, create an IAM role with AmazonSageMakerFullAccess policy attached to it to create the endpoint. See https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html for more details.

Execute the script as:
```
python3 create_sagemaker_endpoint.py --container-image-uri <container_image_uri> --role-arn <role_arn>
```

### 5. Test InvokeEndpointWithBidirectionalStream API for SageMaker Endpoint
The script tests a very simple birectional streaming communicaton. 

Execute the script as:
```
python3 sagemaker-byo-bidi-invoke.py <endpoint_name>
```

This application can be updated to enable your birectional streaming use-cases

### 6. [Optional] Cleanup Test Resources

```
python3 cleanup_sagemaker_resources.py --model-name <MODEL_NAME> --endpoint-config-name <ENDPOINT_CONFIG_NAME> --endpoint-name <ENDPOINT_NAME>
```
