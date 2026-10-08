# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# A small SageMaker AI hosting image for the decision-model samples, built on the
# maintained AWS PyTorch DLC. There is no PyTorch *inference* DLC after 2.6, and the
# PyTorch 2.14 DLC is built for training, so this adds the hosting contract: `serve`
# answers GET /ping and POST /invocations on port 8080 by calling the sample's own
# model_fn and transform_fn from code/inference.py, as the PyTorch inference toolkit did.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.14-cu133-amzn2023-sagemaker
FROM ${BASE_IMAGE}

COPY requirements.txt /opt/serving/requirements.txt
RUN pip install --no-cache-dir -r /opt/serving/requirements.txt

COPY serve.py /opt/serving/serve.py
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
WORKDIR /opt/ml/model
# entrypoint.sh applies the DLC's CUDA forward-compatibility check, then runs `serve`.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
