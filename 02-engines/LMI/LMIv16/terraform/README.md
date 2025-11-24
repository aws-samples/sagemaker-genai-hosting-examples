# LMI v16 Terraform Deployment

This Terraform configuration deploys a SageMaker endpoint with the LMI v16 container using vLLM backend.

## Prerequisites

- Terraform >= 1.0
- AWS CLI configured with appropriate credentials
- HuggingFace access token for model access (if using HuggingFace models)
- **Pre-existing IAM role for SageMaker** (or wait 30-60 seconds between creating the role and endpoint if creating fresh)

## Configuration

1. Copy the example variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```

2. Edit `terraform.tfvars` with your values:
   - `aws_region`: Your AWS region
   - `hf_token`: Your HuggingFace access token
   - Adjust other variables as needed (instance type, scaling limits, etc.)

## Deployment

Run the deployment script:

```bash
./deploy.sh
```

This script automatically:
1. Initializes Terraform
2. Creates IAM resources first
3. Waits 30 seconds for IAM propagation
4. Creates SageMaker resources
5. Shows deployment outputs

## Resources Created

- **S3 Bucket**: `{name_prefix}-rl-checkpoints-custom-{account_id}` for storing custom model weights
- **IAM Role**: SageMaker execution role with necessary permissions
- **SageMaker Model**: Model configuration with LMI v16 container
- **Endpoint Configuration**: Configuration for the endpoint
- **SageMaker Endpoint**: The deployed inference endpoint

### Using Custom Model Weights from S3

By default, the configuration loads models from HuggingFace Hub using the `HF_MODEL_ID` variable. However, if you want to use custom or fine-tuned model weights, you can store them in the created S3 bucket.

#### Option 1: Use the Helper Script (Recommended)

The helper script downloads a model from HuggingFace and uploads it to S3:

```bash
# Install dependencies
pip install -r requirements.txt

# Get your S3 bucket name
terraform output custom_model_bucket_name

# Download and upload model
python download_model_to_s3.py Qwen/Qwen3-1.7B <bucket-name> models/qwen3-1.7b
```

The script will output the S3 URI to use. Update your `terraform.tfvars`:
```hcl
hf_model_id = "s3://<bucket-name>/models/qwen3-1.7b/"
hf_token    = ""  # Not needed for S3 models
```

Then apply the changes:
```bash
terraform apply
```

#### Option 2: Manual Upload

If you already have model files locally:

```bash
# Get bucket name
BUCKET=$(terraform output -raw custom_model_bucket_name)

# Upload your model weights
aws s3 cp /path/to/model/ s3://${BUCKET}/my-custom-model/ --recursive --no-cli-pager

# Update terraform.tfvars
# hf_model_id = "s3://<bucket-name>/my-custom-model/"
# hf_token    = ""

# Apply changes
terraform apply
```

**Note:** Your model directory should contain all necessary files (config.json, model weights, tokenizer files, etc.) in HuggingFace format.

## Testing the Endpoint

### Setup Python Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Test Script

After deployment, test the endpoint using the provided script:

```bash
python test_endpoint.py <endpoint-name>
```

Example:
```bash
python test_endpoint.py lmi-v16-endpoint
```

### Manual Testing

You can also test manually using Python:

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

payload = {
    "inputs": "What is the capitol of the United States?",
    "parameters": {
        "max_new_tokens": 200
    }
}

response = client.invoke_endpoint(
    EndpointName='<endpoint-name>',
    ContentType='application/json',
    Accept='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read().decode())
print(result['generated_text'])
```

## Troubleshooting

### Service Quota Exceeded Error

If you encounter this error during `terraform apply`:

```
Error: creating SageMaker AI Endpoint (lmi-v16-endpoint): operation error SageMaker: CreateEndpoint, 
https response error StatusCode: 400, RequestID: aecd83f9-e9cb-4d18-b8c6-9b82fd2b7166, 
ResourceLimitExceeded: The account-level service limit 'ml.g5.4xlarge for endpoint usage' is 0 Instances, 
with current utilization of 0 Instances and a request delta of 1 Instances.
```

This means your AWS account doesn't have quota for ml.g5.4xlarge instances.

**Check current quota:**
```bash
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-C1B9A48D \
  --region us-east-1 \
  --no-cli-pager \
  --output json
```

**Request quota increase:**
```bash
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-C1B9A48D \
  --desired-value 1 \
  --region us-east-1 \
  --no-cli-pager \
  --output json
```

**Check request status:**
```bash
aws service-quotas list-requested-service-quota-change-history-by-quota \
  --service-code sagemaker \
  --quota-code L-C1B9A48D \
  --region us-east-1 \
  --no-cli-pager \
  --output json
```

Quota increase requests typically take a few hours to a day for approval. You can also request increases through the [AWS Service Quotas console](https://console.aws.amazon.com/servicequotas/).

**Alternative:** Use a different instance type that you have quota for by updating `instance_type` in your `terraform.tfvars`:
```hcl
instance_type = "ml.g5.2xlarge"  # or another available instance type
```

## Cleanup

To destroy all resources:
```bash
terraform destroy
```

## Cost Considerations

- ml.g5.4xlarge instances cost approximately $1.62/hour
- Consider using autoscaling to optimize costs
- Remember to destroy resources when not in use

## Notes

- The endpoint uses inference components for better resource utilization
- Container startup health check timeout is set to 600 seconds
- Adjust `num_gpu` and `instance_type` based on your model requirements
- For models requiring more GPU memory, consider ml.g5.12xlarge or larger instances
