# Bidirectional streaming with vLLM-Omni on SageMaker AI

This sample deploys Qwen3-TTS with the vLLM-Omni SageMaker Deep Learning
Container. It includes a repeatable streaming smoke test and a Gradio
application for trying text-to-speech in a browser.

This example complements
[`../bidirectional-streaming-vLLM/`](../bidirectional-streaming-vLLM/).
The earlier example streams audio to a standard vLLM Realtime API endpoint
for speech-to-text. This example streams text to the purpose-built
vLLM-Omni container for text-to-speech.

## Architecture

The Python client opens an HTTP/2 WebSocket to SageMaker Runtime on port
`8443`. The SageMaker inference sidecar forwards the connection to the
vLLM-Omni `v1/audio/speech/stream` route. Qwen3-TTS returns audio lifecycle
and chunk events over the same connection.

## Prerequisites

- Python 3.12 or newer
- AWS credentials configured through an AWS profile, IAM Identity Center, or
  an IAM role
- A SageMaker execution role
- Permission to create, invoke, and delete SageMaker models, endpoint
  configurations, and endpoints
- `ml.g6.xlarge` capacity in `us-east-1`, or another supported AWS Region

The sample creates a GPU endpoint that incurs charges while it is running.
Its cleanup routine waits for endpoint deletion before deleting the endpoint
configuration and model.

## Install

Clone the repository and enter this sample directory.

```bash
git clone https://github.com/aws-samples/sagemaker-genai-hosting-examples.git
cd sagemaker-genai-hosting-examples/03-features/bidirectional-streaming-vLLM-Omni
```

Create a virtual environment and install the pinned dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Set your SageMaker execution role, then deploy and validate the endpoint.
The `--keep-endpoint` option leaves the endpoint running for the Gradio
application.

```bash
export SAGEMAKER_EXECUTION_ROLE_ARN="arn:aws:iam::<account-id>:role/<role-name>"
python deploy_bidi_stream.py \
  --endpoint-name vllm-omni-bidi \
  --keep-endpoint
```

The default Region is `us-east-1`. Override the Region or container image
when required.

```bash
export AWS_REGION="us-west-2"
export VLLM_OMNI_IMAGE_URI="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1.5"
python deploy_bidi_stream.py
```

A successful run reports `audio.start` and `audio.done`, receives more than
2,000 streamed PCM bytes, and saves `validation-output.wav`.

## Try the Gradio application

Launch the application against the endpoint that you kept running.

```bash
python sagemaker_bidi_tts_client.py \
  --endpoint-name vllm-omni-bidi \
  --region us-east-1
```

Open `http://127.0.0.1:6006`, enter text, and choose **Generate speech**.
The application sends the text over a SageMaker bidirectional stream and
plays the returned PCM chunks as they arrive.

## Clean up

Delete the endpoint, endpoint configuration, and model after testing.

```bash
python deploy_bidi_stream.py \
  --endpoint-name vllm-omni-bidi \
  --delete-endpoint
```

## Files

- `deploy_bidi_stream.py` deploys, validates, retains, or deletes the endpoint.
- `sagemaker_bidi_tts.py` contains the shared HTTP/2 streaming transport.
- `sagemaker_bidi_tts_client.py` contains the Gradio application.
- `requirements.txt` pins the SageMaker bidirectional streaming client and
  the Gradio version used by this sample.
