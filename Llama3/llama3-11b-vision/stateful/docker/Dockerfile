ARG BASE_IMAGE

FROM $BASE_IMAGE

ENV TEMP=/tmp

COPY config.properties /home/model-server/config.properties
COPY sessions.py /home/model-server

RUN pip install --no-cache-dir -U torchserve-nightly==2024.7.24
RUN pip install --no-cache-dir -U dataclasses_json boto3 pillow transformers requests torchvision cog sentencepiece protobuf accelerate
