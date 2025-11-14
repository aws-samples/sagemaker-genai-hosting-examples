# Deploy modern Open AI Automatic Speech Recognition ( ASR ) Models on SageMaker AI

In this notebook we show how to deploy whisper-large-v2 Model on SageMaker AI endpoints.

We will deploy the ASR Model from  HuggingFace hub using managed LMI container then move to more advanced use-case - deploying model from S3 using SageMaker HuggingFace Pytorch container


## 1. Deployment

In this section, we will:
- Deploy a `whisper-large-v2` model from HuggingFace hub using managed LMI container
- See How to deploy `whisper-large-v2` model from S3 location with SageMaker HuggingFace Pytorch container
- We will use it to transcribe speech and also transalate from one language to another