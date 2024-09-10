# Stateful inference using SageMaker endpoints

This is an example of how customers can leverage sticky routing feature in Amazon SageMaker to achieve stateful model Inference, we will use the [LLaVA: Large Language and Vision Assistant model](https://github.com/haotian-liu/LLaVA/tree/main) to demostrate this.

we will use LLava to upload a image and then ask questions about the image without having to resend the image for every request. The image is cached in the GPU Memory, so we don’t have to incur the cost of moving this  image from  CPU Memory to GPU Memory on every call.

We will be using TorchServe as our model server for this example.

Please take a look at the accompaning blog <TODO add link>
