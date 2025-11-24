# Deepgram SageMaker Python Demo

This Python client demonstrates how to use AWS SageMaker's bidirectional streaming API to stream audio to a Deepgram speech-to-text model deployed on SageMaker and receive real-time transcription results.

## Overview

This demo showcases SageMaker's new bidirectional streaming capability, which allows for:
- **Real-time audio streaming** to SageMaker-hosted models
- **Bidirectional communication** for interactive AI applications
- **Low-latency transcription** using Deepgram's industry-leading speech recognition

## Prerequisites

Before running this demo, you'll need:

### AWS Account Setup
1. **AWS Account** with SageMaker access
2. **IAM Permissions**:
   - `AmazonSageMakerFullAccess` policy for endpoint creation
   - `AWSMarketplaceManageSubscriptions` policy to subscribe to Deepgram
   - Permissions that allow the `sagemaker:InvokeEndpoint*` action for endpoint invocation
3. **AWS Credentials** configured (see [AWS Credentials Configuration](#aws-credentials-configuration) below)

### Deepgram SageMaker Endpoint Deployment

You must deploy a Deepgram model on SageMaker before using this demo:

1. **Subscribe to Deepgram on AWS Marketplace**
   - Navigate to the [AWS Marketplace Model Packages](https://console.aws.amazon.com/sagemaker/home#/model-packages) section in the SageMaker console
   - Search for "Deepgram" and filter by "SageMaker Model" delivery method
   - Subscribe to **Deepgram Nova-3** or your preferred model
   - Note: Deepgram offers a 14-day free trial (infrastructure costs still apply)

2. **Create a SageMaker Real-Time Endpoint**
   - After subscribing, proceed to the launch wizard
   - Provide an endpoint name (you'll use this when running the client)
   - Configure the production variant:
     - Click "Edit" in the production variant table (you may need to scroll right)
     - Select instance type: `ml.g6.2xlarge` recommended for testing
     - Set initial instance count: `1`
   - Create the endpoint and wait for it to reach "InService" status (this takes several minutes)

3. **For detailed deployment instructions**, see:
   - [Deepgram SageMaker Documentation](https://developers.deepgram.com/docs/deploy-amazon-sagemaker)
   - Upcoming AWS blog post: "Introducing bidirectional streaming for real-time inference on Amazon SageMaker AI" (Link to be updated after publication)

### Python Requirements
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository and navigate to the python-demo directory:
```bash
git clone <repository-url>
cd deepgram-sagemaker-demo/python-demo
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## AWS Credentials Configuration

The client uses boto3's standard credential chain. Configure your credentials using one of these methods:

### Option 1: AWS CLI (Recommended)
```bash
aws configure
```

### Option 2: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_SESSION_TOKEN=your_token  # Optional, for temporary credentials
```

### Option 3: AWS Credentials File
Create or edit `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

### Option 4: IAM Role
When running on AWS infrastructure (EC2, Lambda, ECS, etc.), the client automatically uses the IAM role attached to the resource.

## Usage

### Basic Usage

```bash
python deepgram_sagemaker_client.py <endpoint-name> <audio-file>
```

Example:
```bash
python deepgram_sagemaker_client.py my-deepgram-endpoint audio.wav
```

Some sample wav files are published on the [Deepgram Getting Started Guide](https://developers.deepgram.com/docs/pre-recorded-audio)

### Advanced Usage

#### Specify Region
```bash
python deepgram_sagemaker_client.py my-deepgram-endpoint audio.wav --region us-east-1
```

#### Use Different Deepgram Model
```bash
python deepgram_sagemaker_client.py my-deepgram-endpoint audio.wav --model nova-2
```

#### Enable Speaker Diarization
```bash
python deepgram_sagemaker_client.py my-deepgram-endpoint audio.wav --diarize true
```

#### Fast Streaming Mode
For longer files, use fast mode to minimize delay between chunks:
```bash
python deepgram_sagemaker_client.py my-deepgram-endpoint audio.wav --fast
```

#### Custom Delay Between Chunks
```bash
python deepgram_sagemaker_client.py my-deepgram-endpoint audio.wav --delay 0.1
```

### All Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `endpoint_name` | (required) | SageMaker endpoint name hosting the Deepgram model |
| `audio_file` | (required) | Path to audio file (WAV format recommended) |
| `--region` | `us-east-1` | AWS region where the SageMaker endpoint is deployed |
| `--model` | `nova-3` | Deepgram model to use (nova-3, nova-2, etc.) |
| `--language` | `en` | Language code for transcription |
| `--diarize` | `false` | Enable speaker diarization (true/false) |
| `--punctuate` | `true` | Enable automatic punctuation (true/false) |
| `--delay` | `0.5` | Delay between audio chunks in seconds |
| `--fast` | (flag) | Use fast streaming mode (sets delay to 0.01s) |
| `--log-level` | `INFO` | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Logging

The client uses Python's standard logging module. By default, it logs at INFO level. To change the logging level, use the `--log-level` argument:

```bash
# Enable debug logging
python deepgram_sagemaker_client.py my-endpoint audio.wav --log-level DEBUG

# Show only warnings and errors
python deepgram_sagemaker_client.py my-endpoint audio.wav --log-level WARNING
```

Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Example Output

```
============================================================
Deepgram SageMaker Bidirectional Streaming Client
============================================================
Endpoint: my-deepgram-endpoint
Audio File: audio.wav
Model: nova-3
Language: en
Region: us-east-1
============================================================

============================================================
🎤 LIVE TRANSCRIPTION
============================================================

  Hello [interim]
  Hello world [interim]
✓ Hello world. (95.2%)
  This is a [interim]
✓ This is a test. (97.8%)

============================================================
```

## Programmatic Usage

You can also use the client as a library in your own Python code:

```python
import asyncio
from deepgram_sagemaker_client import DeepgramSageMakerClient

async def transcribe():
    # Create client
    client = DeepgramSageMakerClient(
        endpoint_name="my-deepgram-endpoint",
        region="us-east-1"
    )

    try:
        # Start session
        await client.start_session(
            model="nova-3",
            language="en",
            punctuate="true",
            diarize="false"
        )

        # Stream audio file
        await client.stream_audio_file("audio.wav", delay=0.5)

        # Wait for processing
        await asyncio.sleep(10)

    finally:
        # Clean up
        await client.end_session()

# Run
asyncio.run(transcribe())
```

## Troubleshooting

### "AWS credentials not found"
- Verify your AWS credentials are configured correctly using one of the methods above
- Try running `aws sts get-caller-identity` to test your credentials
- Ensure your credentials have permissions to invoke SageMaker endpoints

### "FileNotFoundError: Audio file not found"
- Check that the audio file path is correct
- Use absolute paths if relative paths don't work

### "Connection timeout" or "Endpoint not responding"
- Verify the SageMaker endpoint name is correct
- Check that the endpoint is in the `InService` state in the SageMaker console
- Ensure you're using the correct AWS region
- Verify your network allows outbound HTTPS connections to SageMaker

### No transcription output
- Verify the audio file format is compatible (WAV recommended)
- Check that the Deepgram model is properly deployed on the endpoint
- Review CloudWatch logs for the SageMaker endpoint for errors

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│             │      │                  │      │             │
│  Audio File │─────▶│  Python Client   │◀────▶│  SageMaker  │
│   (WAV)     │      │  (This Script)   │      │  Endpoint   │
│             │      │                  │      │  (Deepgram) │
└─────────────┘      └──────────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Real-time    │
                     │ Transcription│
                     │ Output       │
                     └──────────────┘
```

## About SageMaker Bidirectional Streaming

This demo uses SageMaker's bidirectional streaming API that enables real-time, two-way communication with models deployed on SageMaker. This feature opens up new possibilities for:

- Real-time speech recognition and transcription
- Interactive conversational AI
- Live translation services
- Real-time audio/video analysis

## Learn More

- [Deepgram SageMaker Documentation](https://developers.deepgram.com/docs/deploy-amazon-sagemaker)
- [AWS SageMaker Bidirectional Streaming](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html#realtime-endpoints-test-endpoints-sdk)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review SageMaker endpoint logs in CloudWatch
- Verify Deepgram model deployment configuration
- Consult the [Deepgram documentation](https://developers.deepgram.com/docs/deploy-amazon-sagemaker)

## License

This demo is provided as-is for educational and demonstration purposes.
