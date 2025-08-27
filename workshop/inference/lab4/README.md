
## Title
Lab 4b: Deploying with Whisper ASR on SageMaker

## Introduction
In this hands-on lab, we'll explore deploying production-ready Automatic Speech Recognition (ASR) models using Amazon SageMaker's HuggingFace PyTorch container. We'll implement  OpenAI's Whisper ASR model as distinct endpoints. Using SageMaker's flexible serving infrastructure, we'll demonstrate how to handle image tasks through our vision endpoint and audio transcription through our Whisper ASR endpoint. This lab focuses on practical implementation aspects, from model artifact organization to creating robust inference pipelines that can handle real-world deployment scenarios for each model type.

## Model Architecture
Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation, proposed in the paper Robust Speech Recognition via Large-Scale Weak Supervision by Alec Radford et al. from OpenAI. Trained on >5M hours of labeled data, Whisper demonstrates a strong ability to generalise to many datasets and domains in a zero-shot setting.

Whisper large-v3-turbo is a finetuned version of a pruned Whisper large-v3. In other words, it's the exact same model, except that the number of decoding layers have reduced from 32 to 4. As a result, the model is way faster, at the expense of a minor quality degradation.

## Other  details
[Accelerate](https://github.com/huggingface/accelerate) sharding libraries that will be leveraged to host the Whisper large-v3-turbo model.

Accelerate is a library that makes distributed training and inference using PyTorch simple and efficient.

## Let's Build
To start learning under lab 4b you would be seeing jupyter files 
- deploy_whisper_minimal.ipynb: Use this file to run the LMI container with Accelerate library to shard the model using ml.g5.2xlarge instance
- deploy_whisper.ipynb: Use this file to run the LMI container with Accelerate library to shard the model using ml.g5.2xlarge or  ml.g5.12xlarge instance

## Additional Reading Section
[Resources for using Hugging Face with Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
## Conclusion
In this lab, we learned how to deploy a Whisper large-v3-turbo model ASR generation model from OpenAI on multiple GPUs using the SageMaker Large Model Inference container, HuggingFace Accelerate.
