# SageMaker AI features

This directory contains example notebooks that demonstrates SageMaker AI features

- [Inference Components](./inference-component/) - examples of how to use Inference Components on Amazon SageMaker AI
- [Autoscaling](./autoscaling/) - examples of endpoint autoscaling (including scaling down to zero)
- [Deployments](./deployment/) - examples of rolling and blue/green deployments on SageMaker AI
- [Asynchronous Inference inline payloads](./async-inference-inline-payload/) - send the request payload inline with `Body` on `InvokeEndpointAsync` (no S3 upload) for payloads up to 128,000 bytes