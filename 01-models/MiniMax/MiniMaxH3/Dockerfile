# SageMaker-compatible SGLang Diffusion container for MiniMax-H3.
#
# There is no AWS Deep Learning Container that ships SGLang's diffusion extras
# today, so this image is built from the upstream SGLang dev image and adds:
#   1. the platform-specific diffusion extra (not present in the base image)
#   2. ffmpeg, for MP4 muxing of the H.264 + AAC output
#   3. a Flask shim exposing the /ping and /invocations contract on port 8080
#
# VERIFY before running: pin :dev to a digest. The MiniMax-H3 pipeline is new
# model code and the floating tag will move underneath you.
FROM lmsysorg/sglang:dev

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SGLANG_PORT=30010 \
    MEDIA_DIR=/data/minimax-h3

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# The base image carries SGLang source but not the optional diffusion deps.
# Use [diffusion_hip] instead of [diffusion] for AMD Instinct hosts.
RUN python -m pip install --no-cache-dir -e "/sgl-workspace/sglang/python[diffusion]"

RUN python -m pip install --no-cache-dir flask boto3 requests

RUN mkdir -p /data/minimax-h3 /opt/program

COPY sagemaker_serve.py /opt/program/sagemaker_serve.py

WORKDIR /opt/program
EXPOSE 8080

# SageMaker invokes the image with the literal argument "serve".
ENTRYPOINT ["python", "/opt/program/sagemaker_serve.py"]
