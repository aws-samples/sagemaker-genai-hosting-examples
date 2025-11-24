#!/bin/bash
set -e

echo "=========================================="
echo "LMI v16 Terraform Deployment Script"
echo "=========================================="
echo ""

# Check if terraform.tfvars exists
if [ ! -f "terraform.tfvars" ]; then
    echo "❌ Error: terraform.tfvars not found"
    echo "Please copy terraform.tfvars.example to terraform.tfvars and configure it:"
    echo "  cp terraform.tfvars.example terraform.tfvars"
    exit 1
fi

echo "Step 1: Initializing Terraform..."
terraform init

echo ""
echo "Step 2: Creating IAM role, policies, and S3 bucket..."
terraform apply -auto-approve \
    -target=aws_iam_role.sagemaker_execution_role \
    -target=aws_iam_role_policy_attachment.sagemaker_full_access \
    -target=aws_iam_role_policy.sagemaker_s3_policy \
    -target=aws_iam_role_policy.sagemaker_ecr_policy \
    -target=aws_s3_bucket.custom_model \
    -target=aws_s3_bucket_versioning.custom_model \
    -target=aws_s3_bucket_public_access_block.custom_model \
    -target=aws_s3_bucket_server_side_encryption_configuration.custom_model

echo ""
echo "Step 3: Waiting 30 seconds for IAM role propagation..."
sleep 30

echo ""
echo "Step 4: Creating SageMaker model, endpoint configuration, and endpoint..."
terraform apply

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Endpoint name: $(terraform output -raw endpoint_name)"
echo "S3 Bucket: $(terraform output -raw custom_model_bucket_name)"
echo ""
echo "Test the endpoint with:"
echo "  python test_endpoint.py"
