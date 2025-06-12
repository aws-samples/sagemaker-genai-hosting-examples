In this notebook, you will learn how to deploy the Magistral-Small-2506 model (HuggingFace model ID: [mistralai/Magistral-Small-2506](https://huggingface.co/mistralai/Magistral-Small-2506)) using Amazon SageMaker AI. The inference image will be the SageMaker-managed [LMI (Large Model Inference) v15](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-container-docs.html) Docker image. LMI images features a [DJL serving](https://github.com/deepjavalibrary/djl-serving) stack powered by the [Deep Java Library](https://djl.ai/). 

Building upon Mistral Small 3.1 (2503), with added reasoning capabilities, undergoing SFT from Magistral Medium traces and RL on top, it's a small, efficient reasoning model with 24B parameters.

Magistral Small can be deployed locally, fitting within a single RTX 4090 or a 32GB RAM MacBook once quantized.

Learn more about Magistral in Mistral AI's [blog post](https://mistral.ai/news/magistral/).
