# NVIDIA Triton Inference Server

[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) is an open-source serving framework that runs models from multiple frameworks (ONNX, TensorRT, PyTorch, TensorFlow, Python) behind a single C++ inference runtime.

The value over a generic framework DLC (HuggingFace, PyTorch) is that, once the model is loaded, the inference hot path is pure C++ with no Python involvement — which unlocks lower-level GPU optimisations (TensorRT, FP16, graph fusion) that cannot be applied to a live Python model.

Triton features:

- **Multi-framework backends** — ONNX Runtime, TensorRT, PyTorch, TensorFlow, Python, custom C++
- **TensorRT integration** — JIT-compile GPU-specific kernels and layer fusion plans for FP16 / INT8 precision
- **ONNX Runtime graph optimisation** — operator fusion, constant folding, redundant-transpose elimination
- **Dynamic batching** — coalesce concurrent requests into a single GPU forward pass
- **Model ensembles** — chain multiple models server-side without client round-trips
- **Concurrent model execution** — multiple models or model instances on one GPU

## SageMaker Triton container

SageMaker provides a managed Triton Deep Learning Container. The container is pulled from the AWS DLC registry — see [available images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#nvidia-triton-inference-containers-sm-support-only) for versions and regions.

## List of examples

- [deberta-tensorrt](./deberta-tensorrt): Deploy a DeBERTa NLI model with ONNX Runtime + TensorRT FP16 on a SageMaker Inference Component. Includes a baseline-vs-optimised benchmark showing ~3.5× latency reduction.

## Additional resources

- Triton [documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- Official GitHub [repo](https://github.com/triton-inference-server/server)
