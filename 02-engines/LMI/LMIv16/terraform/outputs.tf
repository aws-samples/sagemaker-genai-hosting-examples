output "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.lmi_endpoint.name
}

output "endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.lmi_endpoint.arn
}

output "model_name" {
  description = "Name of the SageMaker model"
  value       = aws_sagemaker_model.lmi_model.name
}

output "execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "endpoint_config_name" {
  description = "Name of the endpoint configuration"
  value       = aws_sagemaker_endpoint_configuration.lmi_endpoint_config.name
}

output "custom_model_bucket_name" {
  description = "Name of the custom model S3 bucket"
  value       = aws_s3_bucket.custom_model.id
}

output "custom_model_bucket_arn" {
  description = "ARN of the custom model S3 bucket"
  value       = aws_s3_bucket.custom_model.arn
}
