# Reflection Beam-1.0-501B-A23B (NVFP4) on ml.g7e.24xlarge (4x NVIDIA RTX PRO 6000 Blackwell, SM120) — SageMaker BYOC image.
#
# Beam (BeamForCausalLM) and its `beam` reasoning / tool-call parsers are not in a vLLM release or an AWS vLLM DLC yet
# (upstream: vllm-project/vllm#61978). The model-specific public image vllm/vllm-openai:reflection-beam carries them,
# so this image is that tag unchanged plus a small nginx layer for the SageMaker /ping + /invocations contract.
# Pin the base by digest: the digest is what was validated (2026-10-16). Re-pin to a vLLM release once Beam ships in one.
ARG BASE=vllm/vllm-openai:reflection-beam@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77
FROM ${BASE}
USER root

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      nginx >/dev/null && rm -rf /var/lib/apt/lists/*

# The base image's NVIDIA_REQUIRE_CUDA driver constraint can reject the managed host driver before the entrypoint
# runs (CannotStartContainerError with no logs). The image is built for CUDA 13; the endpoint uses the
# driver-580 / CUDA 13 inference AMI (al2023-ami-sagemaker-inference-gpu-4-1).
ENV NVIDIA_DISABLE_REQUIRE=true

# SageMaker contract: nginx on :8080 proxies /ping -> vLLM /health and /invocations -> /v1/chat/completions.
COPY nginx.conf /etc/nginx/nginx.conf
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/serve"]
