import boto3

def cleanup_autoscaling(inference_component_names, aas_client, cloudwatch_client):
    """Clean up autoscaling resources for workshop"""
    
    if isinstance(inference_component_names, str):
        inference_component_names = [inference_component_names]
    
    print("🧹 Cleaning up autoscaling resources...")
    
    for ic_name in inference_component_names:
        resource_id = f"inference-component/{ic_name}"
        
        # Clean up all autoscaling components
        try:
            # Get and delete policies
            policies = aas_client.describe_scaling_policies(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:inference-component:DesiredCopyCount"
            )['ScalingPolicies']
            
            for policy in policies:
                aas_client.delete_scaling_policy(
                    PolicyName=policy['PolicyName'],
                    ServiceNamespace="sagemaker",
                    ResourceId=resource_id,
                    ScalableDimension="sagemaker:inference-component:DesiredCopyCount"
                )
            
            # Delete alarm
            cloudwatch_client.delete_alarms(AlarmNames=[f"scale-from-zero-{ic_name}"])
            
            # Deregister target
            aas_client.deregister_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:inference-component:DesiredCopyCount"
            )
            
            print(f"✅ Cleaned up autoscaling for {ic_name}")
            
        except Exception as e:
            print(f"ℹ️  Partial cleanup for {ic_name} (some resources may not exist)")
    
    print("🎉 Autoscaling cleanup complete!")

def cleanup_workshop_resources(model_name, endpoint_config_name, endpoint_name, sagemaker_client):
    """Clean up all workshop resources"""
    try:
        # Delete inference components
        response = sagemaker_client.list_inference_components()
        for ic in response['InferenceComponents']:
            ic_details = sagemaker_client.describe_inference_component(
                InferenceComponentName=ic['InferenceComponentName']
            )
            if ic_details.get('EndpointName') == endpoint_name:
                sagemaker_client.delete_inference_component(
                    InferenceComponentName=ic['InferenceComponentName']
                )
                print(f"✅ Deleted inference component: {ic['InferenceComponentName']}")
        
        # Delete endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Deleted endpoint: {endpoint_name}")
        
        # Delete endpoint configuration
        sagemaker_client.delete_endpoint_config(EndpointConfigurationName=endpoint_config_name)
        print(f"✅ Deleted endpoint configuration: {endpoint_config_name}")

        # Delete model
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"✅ Deleted model: {model_name}")
        
    except Exception as e:
        print(f"ℹ️  Cleanup note: {e}")
