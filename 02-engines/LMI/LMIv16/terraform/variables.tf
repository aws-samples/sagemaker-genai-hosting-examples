variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "lmi-v16"
}

variable "hf_model_id" {
  description = "HuggingFace model ID or S3 path (s3://bucket/path/to/model/)"
  type        = string
  default     = "Qwen/Qwen3-1.7B"
}

variable "hf_token" {
  description = "HuggingFace access token (optional if using S3 model)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "container_version" {
  description = "LMI container version"
  type        = string
  default     = "0.34.0-lmi16.0.0-cu128"
}

variable "instance_type" {
  description = "SageMaker instance type"
  type        = string
  default     = "ml.g5.4xlarge"
}

variable "initial_instance_count" {
  description = "Initial number of instances"
  type        = number
  default     = 1
}

variable "min_instance_count" {
  description = "Minimum number of instances for autoscaling"
  type        = number
  default     = 1
}

variable "max_instance_count" {
  description = "Maximum number of instances for autoscaling"
  type        = number
  default     = 3
}

variable "num_gpu" {
  description = "Number of GPUs for tensor parallelism"
  type        = number
  default     = 1
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "LMI-v16"
    Environment = "dev"
    ManagedBy   = "Terraform"
  }
}
