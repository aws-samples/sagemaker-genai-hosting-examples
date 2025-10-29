from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

agent_name = "qwen_research_agent"
boto_session = Session()
region = "us-west-2"

agentcore_runtime = Runtime()

# Configure the agent
print("Configuring Bedrock Agent Core...")
response = agentcore_runtime.configure(
    entrypoint="main.py",
    auto_create_execution_role=False,
    execution_role="arn:aws:iam::058264502866:role/BedrockAgentCoreQwenRole",  # Use the role we created
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name=agent_name,
)

print("Configuration completed. Starting deployment...")

# Deploy using CodeBuild (recommended for SageMaker Studio)
launch_result = agentcore_runtime.launch(use_codebuild=True)

agent_arn = launch_result.agent_arn
print(f"\n🎉 Deployment successful!")
print(f"Agent ARN: {agent_arn}")
print(f"\nYou can now test your agent using the ARN above.")
