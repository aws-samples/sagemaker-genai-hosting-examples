# Stateful inference using SageMaker endpoints (Work in progress, please dont use this code yet!)

This is an example of how customers can leverage sticky routing feature in Amazon SageMaker to achieve stateful model Inference, when using Llama 3.2 vision model.

we will use Llama 3.2 vision model to upload a image and then ask questions about the image without having to resend the image for every request. The image is cached CPU memory.

We will be using TorchServe as our model server for this example.

Please take a look at the accompanying [workshop]
