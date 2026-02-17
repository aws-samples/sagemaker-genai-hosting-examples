import json
import boto3

from dataclasses import dataclass
from strands.models.sagemaker import SageMakerAIModel
from strands.types.streaming import StreamEvent, ContentBlockDelta
from strands.types.tools import ToolSpec
from typing import Dict, Any, Optional, List


@dataclass
class InferenceResponse:
    """Structured response from the model provider.

    This data class provides a clean interface for model responses,
    abstracting away the underlying sglang format.

    Attributes:
        content: The actual text response from the model
        finish_reason: Why the model stopped generating (stop, length, etc.)
        usage: Token usage statistics for cost tracking
        model: The model that generated the response
    """
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str


class LlamaModelProvider(SageMakerAIModel):
    """Custom model provider for Llama 3.1 deployed with sglang on SageMaker.

    This class extends SageMakerAIModel to provide custom parsing logic
    for sglang's OpenAI-compatible response format. It handles the complexity
    of extracting the actual response text from the nested JSON structure
    and provides meaningful error messages when things go wrong.

    Key Features:
    - Parses sglang OpenAI-compatible responses
    - Validates response structure before accessing fields
    - Provides detailed error messages for debugging
    - Extracts token usage for cost tracking
    - Handles edge cases (empty responses, missing fields, etc.)

    Example Usage:
        provider = LlamaModelProvider(
            endpoint_name="llama-31-deployment",
            region_name="us-east-1"
        )
        response = provider.generate("Hello, how are you?")
        print(response.content)
    """

    def __init__(
        self,
        endpoint_name: str,
        region_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Initialize the Llama model provider.

        Args:
            endpoint_name: Name of the SageMaker endpoint to invoke
                This is the endpoint we created in the deployment step.
                Example: "llama-31-deployment"

            region_name: AWS region where the endpoint is deployed
                Must match the region where you created the endpoint.
                Example: "us-east-1"

            max_tokens: Maximum number of tokens to generate
                Controls the length of the response. Higher values allow
                longer responses but cost more and take longer.
                Default: 1000 (roughly 750 words)

            temperature: Sampling temperature (0.0 to 2.0)
                Controls randomness in generation:
                - 0.0: Deterministic (always picks most likely token)
                - 0.7: Balanced (default, good for most use cases)
                - 1.0: More creative
                - 2.0: Very random

            top_p: Nucleus sampling parameter (0.0 to 1.0)
                Controls diversity by limiting token choices:
                - 0.9: Default, good balance
                - 1.0: Consider all tokens
                - Lower values: More focused responses
        """
        # Store configuration for later use
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Create boto3 client for SageMaker Runtime
        # This is what actually invokes the endpoint
        self.runtime_client = boto3.client(
            'sagemaker-runtime',
            region_name=region_name
        )

        # Initialize the parent class
        # This sets up the base SageMaker invocation logic
        super().__init__(
            endpoint_config={
                "endpoint_name": endpoint_name,
                "region_name": region_name
            },
            payload_config={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        )

        print("\n✓ LlamaModelProvider initialized")
        print(f"  Endpoint: {endpoint_name}")
        print(f"  Region: {region_name}")
        print(f"  Max Tokens: {max_tokens}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-p: {top_p}\n")

    def parse_response(self, raw_response: Dict[str, Any]) -> InferenceResponse:
        """Parse sglang OpenAI-compatible response format.

        This method extracts the actual response content from sglang's nested
        JSON structure. It handles the complexity of navigating through the
        'choices' array and 'message' object to get the text we actually want.

        sglang Response Structure:
            {
              "choices": [{
                "message": {
                  "content": "<-- This is what we want!"
                },
                "finish_reason": "stop"
              }],
              "usage": {...},
              "model": "meta-llama/Llama-3.1-8B-Instruct"
            }

        Args:
            raw_response: The raw JSON response from the SageMaker endpoint
                This is what sglang returns directly, before any parsing.

        Returns:
            InferenceResponse: Structured response with content, finish_reason,
                usage statistics, and model name.

        Raises:
            ValueError: If the response structure is invalid or missing required fields
                We provide detailed error messages to help debug issues.

        Example:
            raw = {"choices": [{"message": {"content": "Hello!"}, ...}], ...}
            response = provider.parse_response(raw)
            print(response.content)  # "Hello!"
        """
        try:
            if not raw_response:
                raise ValueError(
                    "Received empty response from endpoint. "
                )

            if 'choices' not in raw_response:
                raise ValueError(
                    f"Response missing 'choices' field. "
                    f"Received keys: {list(raw_response.keys())}. "
                )

            choices = raw_response['choices']

            if not choices or len(choices) == 0:
                raise ValueError(
                    "Response contains empty 'choices' array. "
                    "The model did not generate any completions. "
                )

            first_choice = choices[0]
            if 'message' not in first_choice:
                raise ValueError(
                    f"First choice missing 'message' field. "
                    f"Received keys: {list(first_choice.keys())}. "
                )

            message = first_choice['message']
            if 'content' not in message:
                raise ValueError(
                    f"Message missing 'content' field. "
                    f"Received keys: {list(message.keys())}. "
                )

            content = message['content']

            if content is None:
                raise ValueError(
                    "Message content is None. "
                    "The model did not generate any text."
                )

            finish_reason = first_choice.get('finish_reason', 'unknown')

            usage = raw_response.get('usage', {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            })

            model = raw_response.get('model', 'unknown')

            # Step 11: Create and return structured response
            # This provides a clean interface for the rest of our code
            return InferenceResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                model=model
            )

        except KeyError as e:
            # KeyError means we tried to access a field that doesn't exist
            # This shouldn't happen if our validation is correct, but just in case...
            raise ValueError(
                f"Failed to parse response: missing key {e}. "
                f"Response structure: {json.dumps(raw_response, indent=2)}"
            )
        except Exception as e:
            # Catch any other unexpected errors
            # We want to provide helpful debugging information
            raise ValueError(
                f"Unexpected error parsing response: {str(e)}. "
                f"Response type: {type(raw_response)}. "
                f"Response: {json.dumps(raw_response, indent=2) if isinstance(raw_response, dict) else str(raw_response)}"
            )

    async def stream(self, messages: List[Dict[str, Any]], tool_specs: list[ToolSpec], system_prompt: Optional[str], **kwargs):
        """Override stream() to handle sglang OpenAI-compatible responses."""

        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})

        for prompt in messages:
            payload_messages.append({"role": "user", "content": prompt['content'][0]['text']})

        payload = {
            "messages": payload_messages,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "stream": True
        }

        response = self.runtime_client.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )

        yield StreamEvent(messageStart={"role": "assistant"})
        yield StreamEvent(
            contentBlockStart={
                "start": {"type": "text"},
                "contentBlockIndex": 0
            }
        )

        buffer = ""
        for event in response['Body']:
            chunk = event['PayloadPart']['Bytes'].decode('utf-8')

            if not chunk.strip():
                continue
                
            buffer += chunk
            lines = buffer.split('\n')
            
            for line in lines[:-1]:
                if line.startswith('data: '):
                    json_str = line.replace('data: ', '').strip()
                    if json_str and json_str != '[DONE]':
                        try:
                            chunk_data = json.loads(json_str)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content is not None:
                                    yield StreamEvent(
                                        contentBlockDelta={
                                            "delta": ContentBlockDelta(text=delta['content']),
                                            "contentBlockIndex": 0
                                        }
                                    )
                        except json.JSONDecodeError:
                            continue
                            
            buffer = lines[-1]

        # Process any remaining data in the buffer
        if buffer:
            if buffer.startswith('data: '):
                json_str = buffer.replace('data: ', '').strip()
                if json_str and json_str != '[DONE]':
                    try:
                        chunk_data = json.loads(json_str)
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            content = delta.get('content')
                            if content is not None:
                                yield StreamEvent(
                                    contentBlockDelta={
                                        "delta": ContentBlockDelta(text=content),
                                        "contentBlockIndex": 0
                                    }
                                )
                    except json.JSONDecodeError:
                        pass

        yield StreamEvent(
            contentBlockStop={
                "contentBlockIndex": 0
            }
        )

        # Signal message stop
        yield StreamEvent(
            messageStop={
                "stopReason": "end_turn"
            }
        )