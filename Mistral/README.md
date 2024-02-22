# Mistral7B SageMaker Deployment
This directory contains examples for deploying Mistral7B on SageMaker Inference. Below you can find a structure of the content that is present within this directory.

- [Mistral7B Real-Time Inference Large Model Inference (LMI) Deployment](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Mistral/LMI/mistral-lmi-sme-dept.ipynb)
    - load-testing-scripts
        - locust_script.py: Locust script to measure invoke_endpoint API call
        - distributed.sh: Tune users and workers to increase traffic
- [Mistral7B Real-Time Inference Text Generation Inference (TGI) Deployment](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Mistral/TGI/mistral-tgi-sme-dept.ipynb)
- To-Do:
    - Asynchronous Inference Mistral7B
    - Inferentia2 Mistral7B
