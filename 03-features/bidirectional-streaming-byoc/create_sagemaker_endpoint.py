#!/usr/bin/env python3
"""
Create SageMaker endpoint with bidirectional streaming support.
This script creates a SageMaker model, endpoint configuration, and endpoint.
"""

import argparse
import boto3
import sys
from datetime import datetime


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create SageMaker endpoint with specified container image"
    )
    parser.add_argument(
        "--container-image-uri",
        required=True,
        help="URI of the container image to use for the SageMaker model"
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2)"
    )
    parser.add_argument(
        "--role-arn",
        required=True,
        help="IAM role ARN for SageMaker execution"
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.xlarge",
        help="Instance type for the endpoint (default: ml.g5.xlarge)"
    )
    parser.add_argument(
        "--model-name-prefix",
        default="bidirectional-streaming",
        help="Prefix for model name (default: bidirectional-streaming)"
    )
    
    return parser.parse_args()


def create_sagemaker_endpoint(container_image_uri, region, role, instance_type, model_name_prefix):
    """Create SageMaker model, endpoint configuration, and endpoint"""
    
    # Generate unique names with timestamp
    timestamp = int(datetime.now().timestamp())
    model_name = f"{model_name_prefix}-{timestamp}"
    endpoint_config_name = f"{model_name_prefix}-config-{timestamp}"
    endpoint_name = f"{model_name_prefix}-endpoint-{timestamp}"
    
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        print(f"🚀 Creating SageMaker model with container: {container_image_uri}")
        
        # Create model
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': container_image_uri,
                'Mode': 'SingleModel'
            },
            ExecutionRoleArn=role
        )
        print(f"✅ Model created: {model_name}")
        
        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        print(f"✅ Endpoint config created: {endpoint_config_name}")
        
        # Create endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"✅ Endpoint creation initiated: {endpoint_name}")
        print("⏳ Waiting for endpoint (5-10 minutes)...")
        
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
        )
        
        print(f"🎉 Endpoint ready: {endpoint_name}")
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Endpoint creation failed: {e}")
        return None


def main():
    """Main function to parse arguments and create endpoint."""
    args = parse_arguments()
    
    print("=" * 60)
    print("SageMaker Endpoint Creation")
    print("=" * 60)
    print(f"Container Image URI: {args.container_image_uri}")
    print(f"AWS Region: {args.region}")
    print(f"IAM Role ARN: {args.role_arn}")
    print(f"Instance Type: {args.instance_type}")
    print(f"Model Name Prefix: {args.model_name_prefix}")
    print("=" * 60)
    
    # Create endpoint
    endpoint_name = create_sagemaker_endpoint(
        container_image_uri=args.container_image_uri,
        region=args.region,
        role=args.role_arn,
        instance_type=args.instance_type,
        model_name_prefix=args.model_name_prefix
    )
    
    if endpoint_name:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Active endpoint: {endpoint_name}")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED!")
        print("Endpoint creation was not successful.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
