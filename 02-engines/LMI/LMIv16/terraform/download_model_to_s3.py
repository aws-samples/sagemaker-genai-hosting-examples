#!/usr/bin/env python3
"""
Download a HuggingFace model and upload it to S3 for use with SageMaker.
Usage: python download_model_to_s3.py <model-id> <s3-bucket> [s3-prefix]
"""

import sys
import os
import boto3
from pathlib import Path

def download_model(model_id, local_path):
    """Download model from HuggingFace using the Python API."""
    print(f"Downloading model {model_id} to {local_path}...")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        print(f"✓ Model downloaded successfully to {local_path}")
        return True
    except ImportError:
        print("✗ huggingface_hub not found. Install it with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

def upload_to_s3(local_path, bucket, prefix):
    """Upload model files to S3."""
    s3_client = boto3.client('s3')
    local_path = Path(local_path)
    
    print(f"\nUploading model to s3://{bucket}/{prefix}/...")
    
    files_uploaded = 0
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{prefix}/{relative_path}".replace('\\', '/')
            
            print(f"  Uploading {relative_path}...")
            s3_client.upload_file(str(file_path), bucket, s3_key)
            files_uploaded += 1
    
    print(f"✓ Uploaded {files_uploaded} files to S3")
    print(f"\nModel S3 URI: s3://{bucket}/{prefix}/")
    return f"s3://{bucket}/{prefix}/"

def main():
    if len(sys.argv) < 3:
        print("Usage: python download_model_to_s3.py <model-id> <s3-bucket> [s3-prefix]")
        print("\nExample:")
        print("  python download_model_to_s3.py Qwen/Qwen3-1.7B my-bucket models/qwen3-1.7b")
        print("\nThis will:")
        print("  1. Download the model from HuggingFace")
        print("  2. Upload it to S3")
        print("  3. Print the S3 URI to use in your terraform.tfvars")
        sys.exit(1)
    
    model_id = sys.argv[1]
    bucket = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else f"models/{model_id.replace('/', '-')}"
    
    # Create temporary directory for download
    local_path = f"/tmp/{model_id.replace('/', '-')}"
    
    print("="*80)
    print(f"Model ID: {model_id}")
    print(f"S3 Bucket: {bucket}")
    print(f"S3 Prefix: {prefix}")
    print(f"Local Path: {local_path}")
    print("="*80)
    
    # Download model
    if not download_model(model_id, local_path):
        sys.exit(1)
    
    # Upload to S3
    try:
        s3_uri = upload_to_s3(local_path, bucket, prefix)
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nUpdate your terraform.tfvars with:")
        print(f'hf_model_id = "{s3_uri}"')
        print(f'hf_token    = ""  # Not needed for S3 models')
        print("\nThen run: terraform apply")
        
    except Exception as e:
        print(f"\n✗ Error uploading to S3: {e}")
        sys.exit(1)
    
    # Cleanup
    print(f"\nCleaning up local files at {local_path}...")
    import shutil
    shutil.rmtree(local_path, ignore_errors=True)
    print("✓ Cleanup complete")

if __name__ == "__main__":
    main()
