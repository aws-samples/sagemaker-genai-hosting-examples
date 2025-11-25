# Deepgram SageMaker JavaScript/TypeScript Demo

This TypeScript client demonstrates how to use AWS SageMaker's bidirectional streaming API to stream audio to a Deepgram speech-to-text model deployed on SageMaker and receive real-time transcription results.

## Overview

This demo showcases SageMaker's new bidirectional streaming capability using JavaScript/TypeScript:
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
   - [Introducing bidirectional streaming for real-time inference on Amazon SageMaker AI](https://aws.amazon.com/blogs/machine-learning/introducing-bidirectional-streaming-for-real-time-inference-on-amazon-sagemaker-ai/)

### Node.js Requirements
- Node.js 18.0.0 or higher
- npm or yarn package manager

## Installation

1. Clone this repository and navigate to the javascript-demo directory:
```bash
git clone <repository-url>
cd deepgram-sagemaker-demo/javascript-demo
```

2. Install dependencies:
```bash
npm install
```

## AWS Credentials Configuration

The client uses the AWS SDK's standard credential chain. Configure your credentials using one of these methods:

### Option 1: Environment Variables (Recommended for Development)
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_SESSION_TOKEN=your_token  # Optional, for temporary credentials
export AWS_REGION=us-east-1
```

### Option 2: AWS Credentials File
Create or edit `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

And `~/.aws/config`:
```ini
[default]
region = us-east-1
```

### Option 3: AWS CLI
```bash
aws configure
```

### Option 4: IAM Role
When running on AWS infrastructure (EC2, Lambda, ECS, etc.), the client automatically uses the IAM role attached to the resource.

## Configuration

Before running the demo, update the following values in `deepgram-sagemaker-client.ts`:

```typescript
const region = "us-east-1";              // Your AWS region
const endpointName = "your-endpoint-name"; // Your SageMaker Deepgram endpoint name
const audioFile = "your-audio.wav";       // Your local audio file path
```

You can also modify the Deepgram model and parameters:

```typescript
const modelInvocationPath = "v1/listen";
const modelQueryString = "model=nova-3&language=en&punctuate=true";
```

## Usage

### Run with ts-node (Development)
```bash
npm start
```

### Build and Run (Production)
```bash
npm run build
node dist/deepgram-sagemaker-client.js
```

### Direct Execution with ts-node
```bash
npx ts-node deepgram-sagemaker-client.ts
```

Some sample wav files are published on the [Deepgram Getting Started Guide](https://developers.deepgram.com/docs/pre-recorded-audio)

## How It Works

The client:

1. **Reads the audio file** in 512KB chunks for optimal streaming performance
2. **Establishes a bidirectional stream** with the SageMaker endpoint using HTTP/2
3. **Streams audio chunks** as binary payload parts to the Deepgram model
4. **Implements keep-alive support** - waits 15 seconds after sending audio to allow processing
5. **Sends a CloseStream message** to properly terminate the connection
6. **Receives and displays** real-time transcription responses

### Keep-Alive Packets

For long audio files, the client includes a 15-second wait after streaming completes. This allows Deepgram to finish processing the final audio chunks. According to [Deepgram's documentation](https://developers.deepgram.com/docs/audio-keep-alive), for very long files you may need to send keep-alive packets:

```typescript
// Example keep-alive packet (not implemented in this basic demo)
yield {
  PayloadPart: {
    Bytes: new TextEncoder().encode('{"type":"KeepAlive"}'),
    DataType: "UTF8"
  }
};
```

### CloseStream Message

The client sends a proper close message to terminate the stream:

```typescript
yield {
  PayloadPart: {
    Bytes: new TextEncoder().encode('{"type":"CloseStream"}'),
    DataType: "UTF8"
  }
};
```

## Example Output

```
Streaming audio: /path/to/audio.wav
Sending audio to Deepgram via SageMaker...
Transcript: Hello world
Transcript: This is a test
Transcript: of real-time speech to text
Transcript: using Deepgram on SageMaker
Audio sent, waiting for transcription to finish...
Streaming finished.
```

## Code Structure

The demo consists of two main functions:

### 1. streamWavFile()
An async generator function that:
- Reads the audio file in chunks
- Yields each chunk as a `RequestEventStream` payload
- Waits for processing to complete
- Sends the CloseStream message

### 2. run()
The main execution function that:
- Creates the SageMaker Runtime HTTP/2 client
- Builds the bidirectional stream command
- Sends the command with the audio stream
- Processes and displays the transcription responses

## Troubleshooting

### "Audio file not found"
- Check that the file path in the code is correct
- Use absolute paths if relative paths don't work

### "No streaming response received"
- Verify the SageMaker endpoint name is correct in the code
- Check that the endpoint is in the `InService` state in the SageMaker console
- Ensure you're using the correct AWS region

### AWS Credential Errors
- Verify your AWS credentials are configured correctly using one of the methods above
- Try running `aws sts get-caller-identity` to test your credentials
- Ensure your credentials have permissions to invoke SageMaker endpoints

### No transcription output
- Verify the audio file format is compatible (WAV recommended)
- Check that the Deepgram model is properly deployed on the endpoint
- Review CloudWatch logs for the SageMaker endpoint for errors
- Ensure the modelQueryString includes valid Deepgram parameters

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│             │      │                  │      │             │
│  Audio File │─────▶│ TypeScript       │◀────▶│  SageMaker  │
│   (WAV)     │      │ Client (HTTP/2)  │      │  Endpoint   │
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

## Advanced Usage

### Custom Deepgram Parameters

You can customize the transcription by modifying the `modelQueryString`:

```typescript
// Enable speaker diarization
const modelQueryString = "model=nova-3&diarize=true";

// Multi-language support
const modelQueryString = "model=nova-3&language=es";

// Multiple parameters
const modelQueryString = "model=nova-3&language=en&punctuate=true&diarize=true&smart_format=true";
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
