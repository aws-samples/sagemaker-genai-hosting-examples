#!/usr/bin/env python3
"""
Cleanup SageMaker resources created by create_sagemaker_endpoint.py script.
This script deletes specific SageMaker endpoints, endpoint configurations, and models
by their exact names provided as command-line arguments.
"""

import argparse
import boto3
import sys
from typing import List, Tuple


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cleanup specific SageMaker resources by name"
    )
    parser.add_argument(
        "--model-name",
        help="Name of the SageMaker model to delete"
    )
    parser.add_argument(
        "--endpoint-config-name",
        help="Name of the SageMaker endpoint configuration to delete"
    )
    parser.add_argument(
        "--endpoint-name",
        help="Name of the SageMaker endpoint to delete"
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    return parser.parse_args()


def validate_resource_exists(sagemaker_client, resource_type: str, resource_name: str) -> bool:
    """Check if a specific SageMaker resource exists."""
    try:
        if resource_type == "endpoint":
            sagemaker_client.describe_endpoint(EndpointName=resource_name)
        elif resource_type == "endpoint_config":
            sagemaker_client.describe_endpoint_config(EndpointConfigName=resource_name)
        elif resource_type == "model":
            sagemaker_client.describe_model(ModelName=resource_name)
        return True
    except sagemaker_client.exceptions.ResourceNotFound:
        return False
    except Exception as e:
        print(f"⚠️ Error checking {resource_type} {resource_name}: {e}")
        return False


def get_endpoint_status(sagemaker_client, endpoint_name: str) -> str:
    """Get the current status of an endpoint."""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except Exception:
        return "Unknown"


def delete_endpoint(sagemaker_client, endpoint_name: str, dry_run: bool = False) -> bool:
    """Delete a SageMaker endpoint."""
    if dry_run:
        print(f"[DRY RUN] Would delete endpoint: {endpoint_name}")
        return True
    
    try:
        status = get_endpoint_status(sagemaker_client, endpoint_name)
        print(f"🗑️ Deleting endpoint: {endpoint_name} (Status: {status})")
        
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Endpoint deletion initiated: {endpoint_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete endpoint {endpoint_name}: {e}")
        return False


def delete_endpoint_config(sagemaker_client, config_name: str, dry_run: bool = False) -> bool:
    """Delete a SageMaker endpoint configuration."""
    if dry_run:
        print(f"[DRY RUN] Would delete endpoint config: {config_name}")
        return True
    
    try:
        print(f"🗑️ Deleting endpoint config: {config_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"✅ Endpoint config deleted: {config_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete endpoint config {config_name}: {e}")
        return False


def delete_model(sagemaker_client, model_name: str, dry_run: bool = False) -> bool:
    """Delete a SageMaker model."""
    if dry_run:
        print(f"[DRY RUN] Would delete model: {model_name}")
        return True
    
    try:
        print(f"🗑️ Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"✅ Model deleted: {model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete model {model_name}: {e}")
        return False


def cleanup_resources(model_name: str = None, endpoint_config_name: str = None, 
                     endpoint_name: str = None, region: str = "us-west-2", 
                     dry_run: bool = False, force: bool = False) -> Tuple[int, int]:
    """Clean up specific SageMaker resources by name."""
    
    print("=" * 60)
    print("SageMaker Resource Cleanup")
    print("=" * 60)
    print(f"AWS Region: {region}")
    print(f"Dry Run: {dry_run}")
    print("=" * 60)
    
    # Collect specified resources
    resources_to_delete = []
    if endpoint_name:
        resources_to_delete.append(("endpoint", endpoint_name))
    if endpoint_config_name:
        resources_to_delete.append(("endpoint_config", endpoint_config_name))
    if model_name:
        resources_to_delete.append(("model", model_name))
    
    if not resources_to_delete:
        print("❌ No resources specified for deletion")
        print("Use --model-name, --endpoint-config-name, or --endpoint-name to specify resources")
        return 0, 0
    
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Validate resources exist
        print("🔍 Validating specified resources...")
        valid_resources = []
        for resource_type, resource_name in resources_to_delete:
            if validate_resource_exists(sagemaker_client, resource_type, resource_name):
                valid_resources.append((resource_type, resource_name))
                print(f"✅ Found {resource_type}: {resource_name}")
            else:
                print(f"⚠️ Resource not found - {resource_type}: {resource_name}")
        
        if not valid_resources:
            print("❌ No valid resources found to delete")
            return 0, len(resources_to_delete)
        
        total_resources = len(valid_resources)
        print(f"\n📋 {total_resources} resources will be deleted:")
        
        # Group resources for display
        endpoints_to_delete = [name for rtype, name in valid_resources if rtype == "endpoint"]
        configs_to_delete = [name for rtype, name in valid_resources if rtype == "endpoint_config"]
        models_to_delete = [name for rtype, name in valid_resources if rtype == "model"]
        
        if endpoints_to_delete:
            print(f"\n📝 Endpoints to delete:")
            for endpoint in endpoints_to_delete:
                status = get_endpoint_status(sagemaker_client, endpoint)
                print(f"   - {endpoint} (Status: {status})")
        
        if configs_to_delete:
            print(f"\n📝 Endpoint Configs to delete:")
            for config in configs_to_delete:
                print(f"   - {config}")
        
        if models_to_delete:
            print(f"\n📝 Models to delete:")
            for model in models_to_delete:
                print(f"   - {model}")
        
        # Confirmation prompt
        if not dry_run and not force:
            print(f"\n⚠️ This will permanently delete {total_resources} resources!")
            response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
            if response not in ['yes', 'y']:
                print("❌ Cleanup cancelled by user")
                return 0, total_resources
        
        print(f"\n🧹 Starting cleanup...")
        deleted_count = 0
        
        # Delete endpoints first (they depend on endpoint configs)
        for endpoint in endpoints_to_delete:
            if delete_endpoint(sagemaker_client, endpoint, dry_run):
                deleted_count += 1
        
        # Delete endpoint configurations (they depend on models)
        for config in configs_to_delete:
            if delete_endpoint_config(sagemaker_client, config, dry_run):
                deleted_count += 1
        
        # Delete models last
        for model in models_to_delete:
            if delete_model(sagemaker_client, model, dry_run):
                deleted_count += 1
        
        return deleted_count, total_resources
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        print("Please manually delete resources through AWS Console if needed")
        return 0, 0


def main():
    """Main function to parse arguments and perform cleanup."""
    args = parse_arguments()
    
    deleted_count, total_count = cleanup_resources(
        model_name=args.model_name,
        endpoint_config_name=args.endpoint_config_name,
        endpoint_name=args.endpoint_name,
        region=args.region,
        dry_run=args.dry_run,
        force=args.force
    )
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETED")
        print(f"Would have deleted {total_count} resources")
    elif deleted_count == total_count and total_count > 0:
        print("SUCCESS!")
        print(f"Successfully deleted {deleted_count}/{total_count} resources")
    elif total_count == 0:
        print("NO RESOURCES FOUND")
        print("No matching resources to delete")
    else:
        print("PARTIAL SUCCESS")
        print(f"Deleted {deleted_count}/{total_count} resources")
        print("Some resources may require manual cleanup")
    print("=" * 60)
    
    if args.dry_run or deleted_count == total_count:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
