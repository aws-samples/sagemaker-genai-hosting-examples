# Deploy modern Foundation Models on SageMaker AI

In this notebook we show how to deploy modern Foundation Models on SageMaker AI endpoints.

We will start with simple use-cases (deploying JumpStart Model) and then move to more advanced use-cases - deploying model from S3, using Inference Component enabled endpoints and custom containers.


## 1. Deployment (simple)

In this section, we will:
- Deploy a `Llama-3.2-3B-Instruct` from SageMaker JumpStart using SageMaker Python SDK
- Deploy a `Qwen3-4B-Thinking-2507` model from HuggingFace hub using managed LMI container


## 2. Deployment (advanced)

In this section, we will show how to use more advanced deployment scenarios:
- How to deploy a model from S3 location and update Python libraries inside the conatiner at startup
- How to deploy a ***quantized*** model on Inference Component enabled endpoint
- How to deploy a model using CloudFormation
- How to deploy a model using your own custom container

