# Stateful inference using SageMaker endpoints

This is an example of how customers can leverage sticky routing feature in Amazon SageMaker to achieve stateful model Inference, when using Llama 3.2 vision model.

we will use Llama 3.2 vision model to upload a image and then ask questions about the image without having to resend the image for every request. The image is cached CPU memory.

We will be using TorchServe as our model server for this example.

Please take a look at the accompanying [workshop TBD]

## Create role to run this notebook
```
aws cloudformation create-stack --stack-name sm-vision-stateful-role \
--template-body https://raw.githubusercontent.com/aws-samples/sagemaker-genai-hosting-examples/refs/heads/main/Llama3/llama3-11b-vision/stateful/sm_vision_stateful_role.yaml \
--capabilities CAPABILITY_NAMED_IAM \
--region us-west-2
```
