#!/bin/bash
set -e

echo "=========================================="
echo "LMI v16 Terraform Destroy Script"
echo "=========================================="
echo ""
echo "⚠️  This will destroy all resources including:"
echo "  - SageMaker endpoint"
echo "  - SageMaker model"
echo "  - Endpoint configuration"
echo "  - IAM role and policies"
echo "  - S3 bucket (if empty)"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Destroy cancelled."
    exit 0
fi

echo ""
echo "Destroying all resources..."
terraform destroy

echo ""
echo "=========================================="
echo "✅ All resources destroyed!"
echo "=========================================="
