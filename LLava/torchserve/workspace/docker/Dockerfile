ARG BASE_IMAGE

FROM $BASE_IMAGE

ENV TEMP=/tmp

COPY config.properties /home/model-server/config.properties
COPY sessions.py /home/model-server
COPY pull_model_weights.py /home/model-server

RUN pip install --no-cache-dir -U torchserve-nightly==2024.7.24
RUN pip install --no-cache-dir -U dataclasses_json boto3 pillow transformers requests torchvision cog sentencepiece protobuf accelerate

RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)"
RUN chmod +x /usr/local/bin/pget

RUN python /home/model-server/pull_model_weights.py
