# Speech-to-text with WhisperX on Amazon SageMaker AI

This example deploys the AWS Deep Learning Containers (DLC) [WhisperX image](https://aws.github.io/deep-learning-containers/whisperx/) to Amazon SageMaker AI **real-time** and **asynchronous** endpoints, and demonstrates transcription, word-level timestamp alignment, speaker diarization, and SRT subtitle generation through the container's OpenAI-compatible multipart API.

## Notebook

[`speech-to-text-whisperx-sagemaker.ipynb`](./speech-to-text-whisperx-sagemaker.ipynb)

## What it covers

- Resolve the regional WhisperX DLC image URI and configure a SageMaker execution role, GPU instance type, and the asynchronous S3 location.
- Create and wait for isolated **real-time** and **asynchronous** endpoints.
- Send a short clip to the real-time endpoint (which has a 60-second response limit) and the full recording to the asynchronous endpoint.
- Validate and render diarized results, write an SRT subtitle file, and clean up both endpoints and this run's S3 objects.

## Endpoints

| Endpoint | Input | Best for |
|----------|-------|----------|
| Real-time (sync) | A short 16-kHz mono WAV sent inline in the request | Low-latency transcription of short audio |
| Asynchronous | The full recording uploaded to Amazon S3 and invoked by S3 URI | Longer audio that exceeds the real-time 60-second response limit |

Both endpoints use the WhisperX `large-v2` model on an `ml.g4dn.xlarge` GPU instance by default.

## Sample audio

The demo uses a public-domain recording of the air traffic control communications from US Airways Flight 1549, the 2009 "Miracle on the Hudson" emergency landing. It is a real multi-party radio exchange with noise, radio compression, and rapid callsign and frequency readouts, which makes it a strong test of transcription accuracy, word-level timestamps, and speaker diarization. The full ~3-minute recording is sent to the asynchronous endpoint and a 40-second segment to the real-time endpoint. Set `LOCAL_AUDIO_PATH` in the notebook to use your own audio instead.

**Source:** ["Flight 1549 FAA New York TRACON audio extract"](https://commons.wikimedia.org/wiki/File:Flight_1549_FAA_New_York_TRACON_audio_extract.ogg), Wikimedia Commons — public domain (U.S. federal government work).

## Prerequisites

- A SageMaker Studio notebook or another Python environment with AWS credentials.
- An IAM role trusted by SageMaker AI (set `ROLE_ARN` / `SAGEMAKER_ROLE_ARN` when not running in SageMaker Studio), with permission to create, describe, and delete SageMaker models, endpoint configurations, and endpoints, plus standard ECR pull permissions for AWS DLCs. The asynchronous workflow additionally needs read/write access to the selected S3 bucket and prefix.
- GPU instance quota for the selected instance type (default `ml.g4dn.xlarge`).
- Network egress to download the Whisper and alignment model weights and the sample audio.

## Cost and cleanup

GPU endpoints incur charges while they exist. The notebook wraps deployment, invocation, and rendering in a `try`/`finally` block that deletes both endpoints, their configurations and models, and this run's S3 objects, including after a failed run. Confirm no `whisperx-*` endpoints remain after execution.
