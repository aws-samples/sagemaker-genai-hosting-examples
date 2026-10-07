# Refresh this digest deliberately when validating a newer AWS DLC for llama.cpp.
ARG DLC_IMAGE=public.ecr.aws/deep-learning-containers/llama-cpp@sha256:1f19c220f5f169228f20cdddca00f4838be22a3a1ce9002bc444006adc3d6d43
FROM ${DLC_IMAGE}

ARG LLAMA_CPP_PYTHON_COMMIT=8a0810fa04bde084bd3446d126831ae5bd65e373
ARG SEMIF_COMMIT=23cf1f39fc9534fe81437200959b6dfc7106e45a

# The binding commit matches the llama.cpp ABI in the pinned DLC. Disabling
# LLAMA_BUILD makes the child image reuse the DLC's libllama implementation.
ENV LLAMA_CPP_LIB_PATH=/opt/llama.cpp/bin \
    SEMIF_LLAMA_CPP_LIB_DIR=/opt/llama.cpp/bin \
    SEMIF_PORT=8080 \
    SEMIF_MAX_TOKENS=4096 \
    SEMIF_MAX_REQUEST_TOKENS=8192 \
    SEMIF_MAX_SHARED_DECISIONS=8 \
    SEMIF_MAX_BODY_BYTES=262144

# SemIf uses the low-level API, so exclude the unused disk-cache import.
RUN dnf install -y --allowerasing --setopt=install_weak_deps=False \
      cmake gcc gcc-c++ git make \
    && python3 -m pip install --no-cache-dir \
         "typing-extensions>=4.5.0" \
         "jinja2>=2.11.3" \
         "transformers==5.17.0" \
         "numpy==2.2.6" \
         "sentencepiece==0.2.1" \
         "protobuf==7.36.1" \
    && CMAKE_ARGS="-DLLAMA_BUILD=OFF -DLLAVA_BUILD=OFF" \
       python3 -m pip install --no-cache-dir --no-deps \
         "git+https://github.com/abetlen/llama-cpp-python.git@${LLAMA_CPP_PYTHON_COMMIT}" \
    && python3 -c 'import site; from pathlib import Path; p = next(candidate for root in site.getsitepackages() if (candidate := Path(root) / "llama_cpp" / "__init__.py").is_file()); p.write_text(p.read_text().replace("from .llama import *\n", ""), encoding="utf-8")' \
    && python3 -m pip install --no-cache-dir --no-deps \
         "git+https://github.com/TheoLeeCJ/SemIf-OpenJev.git@${SEMIF_COMMIT}" \
    && dnf remove -y cmake gcc gcc-c++ git make \
    && dnf clean all \
    && rm -rf /var/cache/dnf /root/.cache

COPY server.py /opt/semif/server.py

EXPOSE 8080
ENTRYPOINT ["python3", "/opt/semif/server.py"]
