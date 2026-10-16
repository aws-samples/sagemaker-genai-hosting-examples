# Reflection Beam worked example

Reflection Beam demonstrates the general deployment workflow across both
SageMaker JumpStart and SageMaker HyperPod Inference.

Use these repository samples:

- [JumpStart real-time endpoint](../../../../../01-models/ReflectionAI/Beam-1.0-501B-A23B/README.md)
- [HyperPod Inference with JumpStart, SGLang, or vLLM](../../../../../SageMakerHyperpod/reflection-beam/README.md)

## Apply the framework

1. Ask which Beam checkpoint variants and GPU hardware are available.
2. Read the current sample to confirm a supported checkpoint, instance,
   runtime, GPU count, and topology. Do not select BF16, FP8, or NVFP4 from the
   accelerator's data-type support alone.
3. Choose JumpStart real-time for the simplest managed endpoint, JumpStart on
   HyperPod for an AWS-managed container with cluster control, or SGLang/vLLM
   only when the workload needs a specific runtime configuration.
4. Confirm Region support, quota, actual capacity, steady and peak GPUs, cost,
   expiry, and cleanup ownership.
5. Use the selected sample's ignored configuration, validation, deployment,
   smoke-test, and cleanup instructions.

Keep Beam model IDs, image digests, model commits, parser names, topology
settings, and speculative-decoding configuration in the existing samples.
Validate with their bounded health checks and inference smoke tests. Do not
direct users to private or partner-only validation assets.
