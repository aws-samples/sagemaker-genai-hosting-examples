terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data source to get current AWS account ID
data "aws_caller_identity" "current" {}

# Data source to get current AWS region
data "aws_region" "current" {}

# S3 Bucket for RL checkpoints
resource "aws_s3_bucket" "custom_model" {
  bucket = "${var.name_prefix}-rl-checkpoints-custom-${data.aws_caller_identity.current.account_id}"

  tags = merge(
    var.tags,
    {
      Name = "rl-checkpoints-custom"
    }
  )
}

# Enable versioning for the S3 bucket
resource "aws_s3_bucket_versioning" "custom_model" {
  bucket = aws_s3_bucket.custom_model.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "custom_model" {
  bucket = aws_s3_bucket.custom_model.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "custom_model" {
  bucket = aws_s3_bucket.custom_model.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.name_prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Attach SageMaker Full Access policy
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Additional policy for S3 access
resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "${var.name_prefix}-s3-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.custom_model.arn,
          "${aws_s3_bucket.custom_model.arn}/*"
        ]
      }
    ]
  })
}

# Additional policy for ECR access
resource "aws_iam_role_policy" "sagemaker_ecr_policy" {
  name = "${var.name_prefix}-ecr-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# SageMaker Model
resource "aws_sagemaker_model" "lmi_model" {
  name               = "${var.name_prefix}-lmi-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image = "${data.aws_caller_identity.current.account_id == "763104351884" ? "" : "763104351884.dkr.ecr.${data.aws_region.current.name}.amazonaws.com/"}djl-inference:${var.container_version}"

    environment = {
      HF_MODEL_ID                      = var.hf_model_id
      HF_TOKEN                         = var.hf_token
      SERVING_FAIL_FAST                = "true"
      OPTION_ASYNC_MODE                = "true"
      OPTION_ROLLING_BATCH             = "disable"
      OPTION_TENSOR_PARALLEL_DEGREE    = tostring(var.num_gpu)
      OPTION_ENTRYPOINT                = "djl_python.lmi_vllm.vllm_async_service"
      OPTION_TRUST_REMOTE_CODE         = "true"
    }
  }

  tags = var.tags
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "lmi_endpoint_config" {
  name = "${var.name_prefix}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.lmi_model.name
    initial_instance_count = var.initial_instance_count
    instance_type          = var.instance_type

    managed_instance_scaling {
      status          = "ENABLED"
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }
  }

  tags = var.tags
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "lmi_endpoint" {
  name                 = "${var.name_prefix}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.lmi_endpoint_config.name

  tags = var.tags
}
