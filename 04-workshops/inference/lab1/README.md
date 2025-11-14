# Deploy MaDeepchine Learning models on SageMaker AI
---
In this notebook we show how to deploy deep learning models to SageMaker AI endpoints using different container options.

We will start with simple options (deploying from JumpStart or Hugging Face model) and then move to more advanced use-cases - deploying model from S3 to PyTorch container, using NVIDIA's specialized Triton containers.

## SageMaker Deep Learning Model Deployment Options

Amazon SageMaker provides multiple deployment options for deep learning models, each optimized for different use cases and requirements.

### 1. SageMaker Deep Learning Containers (DLCs)

Pre-built containers maintained by AWS that include popular ML frameworks and optimizations.

**Benefits:**
- Pre-configured with optimized frameworks (PyTorch, TensorFlow, etc.)
- Regular security updates and patches
- Built-in performance optimizations
- Easy integration with SageMaker features

**Use Cases:**
- Standard model deployments
- Quick prototyping
- Models using common frameworks

### 2. Specialized Containers

Custom containers designed for specific inference engines and optimizations.

#### NVIDIA Triton Inference Server
- Multi-framework support (TensorFlow, PyTorch, ONNX, TensorRT)
- Dynamic batching and model ensembling
- GPU memory optimization
- Concurrent model execution

#### Other Specialized Options
- **DJL Serving**: Java-based serving with advanced batching
- **Text Generation Inference (TGI)**: Optimized for LLM inference
- **TensorRT**: NVIDIA's high-performance inference optimizer
- **Custom containers**: Built for specific model architectures

**Benefits:**
- Specialized optimizations for specific workloads
- Advanced features like dynamic batching
- Better resource utilization
- Framework-specific optimizations

### Choosing the Right Option

- **Use DLCs** for standard deployments with common frameworks
- **Use specialized containers** for performance-critical applications or specific optimization requirements
- **Consider model size, latency requirements, and throughput needs** when selecting deployment options

## LAB1
### 1. Deployment models to SageMaker native Deep Learning Containers(DCSs)

In this section, we will deploy image classification models to SageMaker using three different options:
- Deploy a `ic-efficientnet-v2-imagenet21k-ft1k-m` from SageMaker JumpStart using SageMaker Python SDK
- Deploy a `microsoft/resnet-50` from Hugging Face to dedicated SageMaker container using SageMaker Python SDK
- Deploy a `resnet-50` model from pytorch.org. We will create a model artifact using a pretrained weights binary and a custom inference script. Then deploy the artifact to a PyTorch container.


### 2. Deployment models to NVIDIA Triton container.

In this section, we will show how to deploy a model to NVIDIA Triton Inference Server integrated with SageMaker.
- How to deploy PyTorch BERT model from Hugging Face to a Triton Server.
- How to improve the performance by exporting the model to NVDIA TensorRT and host it on SageMaker.  


