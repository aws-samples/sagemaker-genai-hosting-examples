# Use Amazon Bedrock tooling with Amazon SageMaker Jumpstart Models

<p align="center">

[![Static Badge](https://img.shields.io/badge/Python-Notebook-blue)](/jumpstart-bedrock/amazon-bedrock-with-amazon-sageMaker-jumpstart.ipynb)

[![Static Badge](https://img.shields.io/badge/Blog-Get_Started-green)](https://aws-preview.aka.amazon.com/blogs/machine-learning/use-amazon-bedrock-tooling-with-amazon-sagemaker-jumpstart-models/)

</p>

[Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker-ai/jumpstart/) has long been the go-to service for developers and data scientists seeking to deploy state-of-the-art generative AI models. SageMaker JumpStart helps you get started with machine learning (ML) by providing fully customizable solutions and one-click deployment and fine-tuning of more than 400 popular open-weight and proprietary generative AI models.

[Amazon Bedrock](https://aws.amazon.com/bedrock/) provides the easiest way to build and scale generative AI applications with foundation models. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with.

In this module we will discuss how to leverage Amazon Bedrock tooling with Amazon SageMaker Jumpstart Models. We will walk through the following steps:

1. Deploy the Gemma 2 9B Instruct model using SageMaker JumpStart.
2. Register the model with Amazon Bedrock.
3. Use the Amazon Bedrock RetrieveAndGenerate API to query the Amazon Bedrock knowledge base.
4. Set up Amazon Bedrock Guardrails to help block harmful content and personally identifiable information (PII) data.
5. Invoke models with Converse APIs to show an end-to-end Retrieval Augmented Generation (RAG) pipeline.
