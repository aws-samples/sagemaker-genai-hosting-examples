# DeepSeek-V4.1-Flash on ml.g7e (8x NVIDIA RTX PRO 6000 Blackwell, SM120) — SageMaker BYOC image.
#
# The stock vLLM v0.31.0 release cannot start DeepSeek-V4.1-Flash on SM120 (vllm-project/vllm#59203). This image is the
# official vLLM v0.31.0 release plus three changes, then a small nginx layer for the SageMaker /ping + /invocations contract:
#   1. FlashInfer 0.7.1rc5  — SM120 sparse-MLA accepts runtime page sizes (flashinfer-ai/flashinfer#5197)
#   2. DeepGEMM 17ca18f7    — SM120 paged-MQA supports 32-token pages (vllm-project/DeepGEMM#14)
#   3. sm120-dsv41-block64.patch — 64-token kernel block for the DeepSeek-V4.1 attention + indexer backends on SM12x
# Serve with --block-size 64 (set by `serve`). Pin the base by digest: the tag is what was validated.
ARG BASE=vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${BASE}
ARG FLASHINFER_VERSION=0.7.1rc5
ARG DEEPGEMM_REF=17ca18f73e19c54dd79ad8f2848b4ab893997edc
USER root

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      git patch nginx libdw-dev libelf-dev >/dev/null && rm -rf /var/lib/apt/lists/*

# 1. FlashInfer: replace every 0.7.0.post1 package (jit-cache must match flashinfer-python exactly).
#    flashinfer-cubin has no 0.7.1 release candidate on PyPI and is not needed for this model.
RUN pip uninstall -y $(pip list 2>/dev/null | awk '/^flashinfer/{print $1}') && \
    pip install --no-cache-dir --no-deps "flashinfer-python==${FLASHINFER_VERSION}" && \
    pip install --no-cache-dir --no-deps --index-url https://flashinfer.ai/whl/cu130 \
      "flashinfer-jit-cache==${FLASHINFER_VERSION}" "flashinfer-jit-cache-sm120f==${FLASHINFER_VERSION}" && \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# 2. DeepGEMM: a top-level `deep_gemm` package takes precedence over vLLM's vendored copy.
#    The runtime image has no CUDA toolkit headers, so point the build at the pip-installed ones.
RUN git clone -q --recursive --shallow-submodules https://github.com/vllm-project/DeepGEMM.git /tmp/dg && \
    cd /tmp/dg && git checkout -q ${DEEPGEMM_REF} && git submodule update -q --init --recursive && \
    CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include TORCH_CUDA_ARCH_LIST="12.0" \
      python3 setup.py -q bdist_wheel && pip install --no-cache-dir --no-deps dist/*.whl && \
    cd / && rm -rf /tmp/dg && python3 -c "import deep_gemm; print('deep_gemm', deep_gemm.__file__)"

# 3. vLLM patch, applied to the installed package.
COPY sm120-dsv41-block64.patch /tmp/sm120-dsv41-block64.patch
RUN cd "$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" && \
    git apply --verbose /tmp/sm120-dsv41-block64.patch && rm /tmp/sm120-dsv41-block64.patch

# SageMaker contract: nginx on :8080 proxies /ping -> vLLM /health and /invocations -> /v1/chat/completions.
COPY nginx.conf /etc/nginx/nginx.conf
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/serve"]
