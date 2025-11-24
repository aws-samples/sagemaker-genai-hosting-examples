#!/usr/bin/env python3
"""
Test script for the deployed SageMaker LMI endpoint.
Usage: python test_endpoint.py [endpoint-name]
Default endpoint: lmi-v16-endpoint
"""

import sys
import json
import boto3

def test_endpoint(endpoint_name):
    """Test the SageMaker endpoint with a sample inference request."""
    
    client = boto3.client('sagemaker-runtime')
    
    payload = {
        "inputs": "What is the meaning of life?",
        "parameters": {
            "max_new_tokens": 200
        }
    }
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nInvoking endpoint...")
    
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        print("\n" + "="*80)
        print("Response:")
        print("="*80)
        print(result.get('generated_text', result))
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nError invoking endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    # Default endpoint name
    endpoint_name = "lmi-v16-endpoint"
    
    # Override with command line argument if provided
    if len(sys.argv) > 1:
        endpoint_name = sys.argv[1]
    
    success = test_endpoint(endpoint_name)
    sys.exit(0 if success else 1)
