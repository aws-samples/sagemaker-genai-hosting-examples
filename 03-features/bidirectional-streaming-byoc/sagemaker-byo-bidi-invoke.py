#!/usr/bin/env python3
"""
SageMaker Bidirectional Streaming Python SDK Script.
This script connects to a SageMaker endpoint for bidirectional streaming communication.
"""

import argparse
import asyncio
import sys
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import InvokeEndpointWithBidirectionalStreamInput, RequestStreamEventPayloadPart, RequestPayloadPart
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
import logging


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Connect to SageMaker endpoint for bidirectional streaming"
    )
    parser.add_argument(
        "ENDPOINT_NAME",
        help="Name of the SageMaker endpoint to connect to"
    )
    
    return parser.parse_args()


# Configuration
AWS_REGION = "us-west-2"
BIDI_ENDPOINT = f"https://runtime.sagemaker.{AWS_REGION}.amazonaws.com:8443"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleClient:
    def __init__(self, endpoint_name, region=AWS_REGION):
        self.endpoint_name = endpoint_name
        self.region = region
        self.client = None
        self.stream = None
        self.response = None
        self.is_active = False

    def _initialize_client(self):
        config = Config(
            endpoint_uri=BIDI_ENDPOINT,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")}
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)

    async def start_session(self):
        if not self.client:
            self._initialize_client()

        logger.info(f"Starting session with endpoint: {self.endpoint_name}")
        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(endpoint_name=self.endpoint_name)
        )
        self.is_active = True

        self.response = asyncio.create_task(self._process_responses())

    async def send_words(self, words):
        for i, word in enumerate(words):
            logger.info(f"Sending payload: {word}")
            await self.send_event(word.encode('utf-8'))
            await asyncio.sleep(1)

    async def send_event(self, data_bytes):
        payload = RequestPayloadPart(bytes_=data_bytes)
        event = RequestStreamEventPayloadPart(value=payload)
        await self.stream.input_stream.send(event)

    async def end_session(self):
        if not self.is_active:
            return

        await self.stream.input_stream.close()
        logger.info("Stream closed")

    async def _process_responses(self):
        try:
            output = await self.stream.await_output()
            output_stream = output[1]

            while self.is_active:
                result = await output_stream.receive()

                if result is None:
                    logger.info("No more responses")
                    break

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    logger.info(f"Received: {response_data}")
        except Exception as e:
            logger.error(f"Error processing responses: {e}")


def main():
    """Main function to parse arguments and run the streaming client."""
    args = parse_arguments()
    
    print("=" * 60)
    print("SageMaker Bidirectional Streaming Client")
    print("=" * 60)
    print(f"Endpoint Name: {args.ENDPOINT_NAME}")
    print(f"AWS Region: {AWS_REGION}")
    print("=" * 60)
    
    async def run_client():
        sagemaker_client = SimpleClient(endpoint_name=args.ENDPOINT_NAME)

        try:
            await sagemaker_client.start_session()

            words = ["I need help with", "my account balance", "I can help with that", "and recent charges"]
            await sagemaker_client.send_words(words)

            await asyncio.sleep(2)

            await sagemaker_client.end_session()
            sagemaker_client.is_active = False

            if sagemaker_client.response and not sagemaker_client.response.done():
                sagemaker_client.response.cancel()

            logger.info("Session ended successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Client error: {e}")
            return 1

    try:
        exit_code = asyncio.run(run_client())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
