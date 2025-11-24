#!/usr/bin/env python3
"""
Deepgram SageMaker Bidirectional Streaming Client

This client demonstrates how to use AWS SageMaker's bidirectional streaming API
to stream audio to a Deepgram model deployed on SageMaker and receive real-time transcription.
"""

import asyncio
import argparse
import json
import logging
import os
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

# Configuration constants
DEFAULT_REGION = "us-east-1"
CHUNK_SIZE = 512_000  # 512 KB chunks for optimal streaming performance

logger = logging.getLogger(__name__)


class DeepgramSageMakerClient:
    """
    Client for streaming audio to Deepgram models deployed on AWS SageMaker.

    This client uses SageMaker's bidirectional streaming API to send audio data
    and receive real-time transcription results from a Deepgram model.
    """

    def __init__(self, endpoint_name, region=DEFAULT_REGION):
        self.endpoint_name = endpoint_name
        self.region = region
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self.client = None
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.response_task = None

    def _initialize_client(self):
        """Initialize the SageMaker Runtime HTTP2 client with AWS credentials"""
        logger.debug("Initializing SageMaker client")

        # Use boto3 to resolve credentials via standard AWS credential chain
        # This automatically checks: environment vars, ~/.aws/credentials, IAM roles, etc.
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()

            if credentials is None:
                raise NoCredentialsError()

            # Ensure credentials are available in environment for smithy client
            frozen_creds = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen_creds.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen_creds.secret_key
            if frozen_creds.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen_creds.token

            logger.debug("AWS credentials successfully loaded")

            # Optionally log the credential source for debugging
            caller_identity = session.client('sts').get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found")
            logger.error("Please configure AWS credentials using one of these methods:")
            logger.error("  1. AWS CLI: aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            logger.error("  3. AWS credentials file: ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e
        except Exception as e:
            logger.error(f"Error initializing AWS credentials: {e}")
            raise

        logger.debug(f"Using SageMaker endpoint: {self.bidi_endpoint}")
        logger.debug(f"Region: {self.region}")

        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")}
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)
        logger.info("SageMaker client initialized successfully")

    async def start_session(self, model="nova-3", language="en", **kwargs):
        """
        Start a bidirectional streaming session with Deepgram on SageMaker
        
        Args:
            model: Deepgram model to use (default: nova-3)
            language: Language code (default: en)
            **kwargs: Additional Deepgram query parameters (diarize, punctuate, etc.)
        """
        if not self.client:
            self._initialize_client()

        # Build query string for Deepgram parameters
        query_params = {
            "model": model,
            "language": language,
        }
        query_params.update(kwargs)

        # Convert dict to query string
        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

        logger.debug(f"Starting session with endpoint: {self.endpoint_name}")
        logger.debug(f"Deepgram parameters: {query_string}")

        # Create the bidirectional stream
        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path="v1/listen",
            model_query_string=query_string
        )

        logger.debug("Invoking endpoint with bidirectional stream")
        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(stream_input)
        self.is_active = True

        logger.debug("Stream created, connecting to output stream")
        # Get output stream immediately before starting background task
        output = await self.stream.await_output()
        self.output_stream = output[1]
        logger.debug("Connected to output stream")

        # Start processing responses in the background
        logger.debug("Starting response processor")
        self.response_task = asyncio.create_task(self._process_responses())

        # Give the response processor a moment to start
        await asyncio.sleep(0.1)

        logger.info("Session started successfully")

    async def stream_audio_file(self, file_path, delay=0.5):
        """
        Stream an audio file to Deepgram in chunks.

        Args:
            file_path: Path to the audio file
            delay: Delay between chunks in seconds (simulates real-time streaming)
        """
        audio_path = Path(file_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logger.info(f"Streaming audio file: {audio_path}")
        logger.debug(f"Using {delay}s delay between chunks")

        # Stream the file in chunks
        with open(audio_path, 'rb') as audio_file:
            chunk_count = 0
            while True:
                chunk = audio_file.read(CHUNK_SIZE)
                if not chunk:
                    break

                chunk_count += 1
                await self.send_audio_chunk(chunk)
                logger.debug(f"Sent chunk {chunk_count} ({len(chunk)} bytes)")

                # Delay to simulate real-time streaming and allow transcription to process
                await asyncio.sleep(delay)

        logger.info(f"Finished streaming {chunk_count} chunks")
        logger.debug("Waiting for Deepgram to process all audio")

    async def send_audio_chunk(self, audio_bytes):
        """Send a chunk of audio data to the stream"""
        if not self.is_active:
            raise RuntimeError("Session not active")
        
        payload = RequestPayloadPart(bytes_=audio_bytes)
        event = RequestStreamEventPayloadPart(value=payload)
        await self.stream.input_stream.send(event)

    async def _process_responses(self):
        """Process streaming responses from Deepgram"""
        try:
            logger.debug("Response processor started, waiting for transcriptions")
            print("\n" + "="*60)
            print("🎤 LIVE TRANSCRIPTION")
            print("="*60 + "\n")

            while self.is_active:
                result = await self.output_stream.receive()

                if result is None:
                    logger.debug("No more responses from server")
                    break
                
                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    
                    try:
                        # Parse JSON response from Deepgram
                        parsed = json.loads(response_data)
                        
                        # Extract and print transcript if available
                        if 'channel' in parsed:
                            alternatives = parsed.get('channel', {}).get('alternatives', [])
                            if alternatives and alternatives[0].get('transcript'):
                                transcript = alternatives[0]['transcript']
                                if transcript.strip():  # Only print non-empty transcripts
                                    confidence = alternatives[0].get('confidence', 0)
                                    is_final = parsed.get('is_final', False)
                                    speech_final = parsed.get('speech_final', False)
                                    
                                    # Show final vs interim results differently
                                    if is_final and speech_final:
                                        print(f"✓ {transcript} ({confidence:.1%})")
                                    else:
                                        print(f"  {transcript} [interim]")
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON response: {response_data}")
            
            # Continue processing even after is_active becomes False
            # to catch any final responses that were already in flight
            logger.debug("Processing any remaining buffered responses...")
            remaining_count = 0
            while remaining_count < 10:  # Check up to 10 more times
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), timeout=0.5)
                    if result is None:
                        break
                    
                    if result.value and result.value.bytes_:
                        remaining_count += 1
                        response_data = result.value.bytes_.decode('utf-8')
                        try:
                            parsed = json.loads(response_data)
                            if 'channel' in parsed:
                                alternatives = parsed.get('channel', {}).get('alternatives', [])
                                if alternatives and alternatives[0].get('transcript'):
                                    transcript = alternatives[0]['transcript']
                                    if transcript.strip():
                                        confidence = alternatives[0].get('confidence', 0)
                                        is_final = parsed.get('is_final', False)
                                        speech_final = parsed.get('speech_final', False)
                                        if is_final and speech_final:
                                            print(f"✓ {transcript} ({confidence:.1%})")
                                        else:
                                            print(f"  {transcript} [interim]")
                        except json.JSONDecodeError:
                            pass
                except asyncio.TimeoutError:
                    break
            
            if remaining_count > 0:
                logger.debug(f"Processed {remaining_count} additional responses after stream close")
                        
        except Exception as e:
            logger.error(f"Error processing responses: {e}", exc_info=True)
        finally:
            print("\n" + "="*60)

    async def end_session(self):
        """Close the streaming session"""
        if not self.is_active:
            return

        logger.debug("Ending session")
        self.is_active = False

        # Close the input stream - this signals to Deepgram that no more audio is coming
        await self.stream.input_stream.close()
        logger.debug("Input stream closed, waiting for final responses")

        # Wait for the response processing task to complete naturally
        if self.response_task and not self.response_task.done():
            try:
                # Give it up to 15 seconds to finish processing remaining responses
                await asyncio.wait_for(self.response_task, timeout=15.0)
                logger.debug("All responses received")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for final responses (15s elapsed)")
                self.response_task.cancel()
            except asyncio.CancelledError:
                pass

        logger.info("Session ended successfully")


async def main():
    """Main function to run the Deepgram streaming client"""
    parser = argparse.ArgumentParser(
        description="Stream audio to Deepgram on SageMaker for transcription"
    )
    parser.add_argument(
        "endpoint_name",
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (WAV format)"
    )
    parser.add_argument(
        "--model",
        default="nova-3",
        help="Deepgram model to use (default: nova-3)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--diarize",
        default="false",
        help="Enable speaker diarization (default: false)"
    )
    parser.add_argument(
        "--punctuate",
        default="true",
        help="Enable punctuation (default: true)"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between audio chunks in seconds for demo (default: 0.5, use 0.01 for faster)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast streaming mode (minimal delay, good for long files)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Configure logging based on command-line argument
    # Configure logging based on command-line argument
    # Must be called before any loggers are created to ensure proper configuration
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration even if basicConfig was called before
    )
    
    # Apply fast mode if requested
    if args.fast:
        args.delay = 0.01
    
    print("=" * 60)
    print("Deepgram SageMaker Bidirectional Streaming Client")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Audio File: {args.audio_file}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Region: {args.region}")
    print("=" * 60)
    
    # Create client
    client = DeepgramSageMakerClient(
        endpoint_name=args.endpoint_name,
        region=args.region
    )
    
    try:
        # Start session with Deepgram parameters
        await client.start_session(
            model=args.model,
            language=args.language,
            diarize=args.diarize,
            punctuate=args.punctuate
        )
        
        # Stream the audio file
        await client.stream_audio_file(args.audio_file, delay=args.delay)

        # Wait for Deepgram to process all audio
        logger.debug("Audio streaming complete, waiting for final transcriptions")
        await asyncio.sleep(10)  # Wait 10 seconds for processing
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        return 1
    finally:
        # Clean up
        await client.end_session()
    
    logger.info("✅ Transcription complete!")
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
