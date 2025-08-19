# Building an Agentic Loan Underwriter: LangGraph + Amazon SageMaker Jumpstart + Amazon Bedrock AgentCore

This project demonstrates how to build and deploy an intelligent loan underwriting agent using LangGraph, Amazon SageMaker with OpenAI GPT-OSS models, and Amazon Bedrock AgentCore Runtime. The agent transforms conversational loan applications into comprehensive financial analysis with automated approval/denial decisions.

## Overview

The loan underwriting agent performs a complete three-step analysis workflow:

1. **Application Parsing**: Extracts structured data from conversational loan applications
2. **Creditworthiness Analysis**: Performs comprehensive financial analysis and risk assessment
3. **Final Decision**: Makes approval/denial decisions with detailed loan terms and reasoning

## Architecture

The solution combines several AWS services and frameworks:

- **LangGraph**: Multi-agent orchestration framework for complex workflows
- **Amazon SageMaker**: Hosts the OpenAI GPT-OSS model via JumpStart
- **Amazon Bedrock AgentCore Runtime**: Scalable cloud deployment platform
- **Custom Tools**: Specialized loan underwriting functions for parsing, analysis, and decision-making

## Prerequisites

### 1. Deploy GPT-OSS Model on Amazon SageMaker

Before using this project, you must deploy the OpenAI GPT-OSS model:

1. Open the notebook `openai_gpt_oss.ipynb` in the `./deploy_sagemaker/gpt-oss` folder
2. Follow the instructions to deploy the GPT-OSS model to a SageMaker endpoint
3. Note the endpoint name for configuration

### 2. Required Permissions

The agent requires specific IAM permissions to access AWS services. Use the provided role creation function or ensure your execution role has:

- SageMaker endpoint invocation permissions
- ECR repository management and access
- CloudWatch logs write permissions
- Bedrock AgentCore workload access
- X-Ray tracing and CloudWatch metrics permissions

## Project Structure

```
langgraph-on-amazon-sagemaker/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── Loan_UnderWriter_GPT_OSS_Agent_Core.ipynb  # Main tutorial notebook
├── langgraph_loan_sagemaker_gpt_oss.py        # AgentCore deployment version
├── langgraph_loan_local.py                    # Local testing version
├── create_agentcore_role.py                   # IAM role creation utility
├── loan_response_parser.py                    # Response parsing utilities
└── sagemaker_prerequisite.md                  # Setup instructions
```

## Key Components

### Loan Processing Tools

The agent uses three specialized tools:

#### 1. parse_loan_application
- Extracts structured information from conversational text
- Uses regex patterns to identify key data points
- Returns formatted applicant details including name, income, credit score, loan amount, and purpose

#### 2. analyze_creditworthiness
- Performs comprehensive financial analysis
- Calculates loan-to-income ratios and risk assessments
- Evaluates credit scores and employment stability
- Provides detailed risk factor analysis

#### 3. make_final_decision
- Makes approval/denial decisions using scoring algorithms
- Calculates loan terms including interest rates and monthly payments
- Provides detailed reasoning for all decisions
- Returns comprehensive loan terms for approved applications

### SageMaker Integration

The `SagemakerLLMWrapper` class provides:
- Integration between LangGraph and SageMaker endpoints
- Harmony format payload handling for GPT-OSS models
- Automatic tool execution for loan applications
- Proper response formatting for LangGraph compatibility

## Getting Started

### 1. Local Development and Testing

For local experimentation, use the `langgraph_loan_local.py` file:

```python
from langgraph_loan_local import langgraph_loan_sagemaker

# Test the agent locally
result = langgraph_loan_sagemaker({
    "prompt": "My name is Lisa Chen, 26 years old. I'm a nurse making $68,000 per year, been working for 18 months. Need $22,000 for student loan consolidation. My credit score is 710."
})
print(result)
```

### 2. Create IAM Role

Create the required IAM role with proper permissions:

```python
from create_agentcore_role import create_bedrock_agentcore_role

role_arn = create_bedrock_agentcore_role(
    role_name="MyLoanUnderwriterRole",
    sagemaker_endpoint_name="your-endpoint-name",
    region="us-west-2"
)
```

### 3. Deploy to AgentCore Runtime

Use the Bedrock AgentCore Starter Toolkit to deploy:

```python
from bedrock_agentcore_starter_toolkit import Runtime

agentcore_runtime = Runtime()

# Configure the agent
response = agentcore_runtime.configure(
    entrypoint="langgraph_loan_sagemaker_gpt_oss.py",
    execution_role=role_arn,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region="us-west-2",
    agent_name="loan_underwriter_agent"
)

# Deploy to the cloud
launch_result = agentcore_runtime.launch(local=False, local_build=False)
```

### 4. Invoke the Deployed Agent

```python
import boto3
import json

agentcore_client = boto3.client('bedrock-agentcore', region_name='us-west-2')

response = agentcore_client.invoke_agent_runtime(
    agentRuntimeArn=launch_result.agent_arn,
    qualifier="DEFAULT",
    payload=json.dumps({
        "prompt": "My name is John Doe, 30 years old. I am a teacher making $45,000 per year, been working for 3 years. Need $15,000 for home improvement. My credit score is 680."
    })
)

# Parse and display results
from loan_response_parser import parse_bedrock_agentcore_response
loan_analysis = parse_bedrock_agentcore_response(response)
```

## Response Format

The agent returns a comprehensive analysis with three main sections:

### Step 1 - Application Parsing
```
PARSED APPLICATION DATA:
========================
Name: Lisa Chen
Age: 26
Annual Income: $68,000
Loan Amount Requested: $22,000
Credit Score: 710
Occupation: nurse
Employment Tenure: 18 months
Loan Purpose: student loan consolidation
```

### Step 2 - Creditworthiness Analysis
```
CREDITWORTHINESS ANALYSIS:
==========================
Financial Metrics:
- Annual Income: $68,000
- Monthly Income: $5,666.67
- Loan Amount: $22,000
- Loan-to-Income Ratio: 32.4%

Credit Assessment:
- Credit Score: 710
- Credit Rating: Good
- Credit Risk Level: Low-Medium

Employment Analysis:
- Employment Tenure: 18 months
- Stability Assessment: Moderate (1+ years)

Risk Factors:
- Loan-to-Income Ratio: MODERATE
- Credit Risk: Low-Medium
- Employment Risk: MODERATE
```

### Step 3 - Final Decision
```
FINAL LOAN DECISION: APPROVED
===================
Approval Score: 7/8

Decision Factors:
- Strong credit score (710)
- Moderate loan-to-income ratio (32.4%)
- Strong income ($68,000)

LOAN TERMS:
- Loan Amount: $22,000
- Interest Rate: 5.90% APR
- Term: 60 months
- Monthly Payment: $424.30
- Total Interest: $3,457.96

Recommendation: Proceed with loan origination
```

## Configuration

### Environment Variables

The agent can be configured through environment variables or direct parameter modification:

- `SAGEMAKER_ENDPOINT_NAME`: Name of your SageMaker endpoint
- `AWS_REGION`: AWS region for deployment (default: us-west-2)
- `SAGEMAKER_REGION`: Region where SageMaker endpoint is deployed (default: us-east-2)

### Model Parameters

Adjust model behavior in the `ContentHandler` class:

```python
payload = {
    "max_output_tokens": 2048,  # Maximum response length
    "temperature": 0.1,         # Response randomness (0.0-1.0)
    "top_p": 1                  # Nucleus sampling parameter
}
```

### Decision Logic

Modify the approval criteria in `make_final_decision`:

- Credit score thresholds (currently 700+ for strong, 650+ for acceptable)
- Loan-to-income ratio limits (currently 25% low, 40% moderate)
- Income adequacy levels (currently $60k+ strong, $40k+ adequate)

## Monitoring and Debugging

### CloudWatch Logs

Agent execution logs are available in CloudWatch:

```bash
aws logs tail "/aws/bedrock-agentcore/runtimes/your-agent-id-DEFAULT" --follow --region us-west-2
```

### Response Parsing

Use the provided parsing utilities to extract structured information:

```python
from loan_response_parser import parse_bedrock_agentcore_response

# Parse and display formatted results
analysis_data = parse_bedrock_agentcore_response(invoke_response)

# Access specific data programmatically
decision = analysis_data['final_decision']
applicant_info = analysis_data['parsed_application']
```

## Error Handling

The agent includes comprehensive error handling:

- **JSON parsing errors**: Falls back to plain text display
- **Tool execution errors**: Returns error messages with context
- **SageMaker endpoint errors**: Provides debugging information
- **Missing data**: Handles incomplete loan applications gracefully

## Security Considerations

- **IAM roles**: Use least-privilege permissions for production deployments
- **Data handling**: Ensure PII is handled according to compliance requirements
- **Endpoint access**: Restrict SageMaker endpoint access to authorized services only
- **Logging**: Be mindful of sensitive data in CloudWatch logs

## Customization

### Adding New Tools

To add additional analysis tools:

1. Create a new function decorated with `@tool`
2. Add the tool to the `tools` list in `create_agent()`
3. Update the `SagemakerLLMWrapper` to handle the new tool
4. Redeploy the agent

### Modifying Decision Logic

Update the scoring algorithm in `make_final_decision()`:

```python
# Custom scoring logic
if custom_criteria:
    approval_score += custom_points
    reasons.append("Custom criteria met")
```

### Changing Model Parameters

Adjust the SageMaker model configuration:

```python
model_kwargs = {
    "max_new_tokens": 4096,     # Increase for longer responses
    "temperature": 0.2,         # Adjust creativity vs consistency
    "top_p": 0.9               # Modify response diversity
}
```

## Troubleshooting

### Common Issues

1. **"Unsupported message type" error**: Ensure `AIMessage` is returned from tools
2. **SageMaker endpoint timeout**: Check endpoint status and increase timeout
3. **Permission denied**: Verify IAM role has all required permissions
4. **Tool not executing**: Check tool detection logic in `SagemakerLLMWrapper`

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

- **Batch processing**: Process multiple applications in a single request
- **Caching**: Cache parsed application data for repeated analysis
- **Async processing**: Use async patterns for high-throughput scenarios
- **Model optimization**: Fine-tune model parameters for your use case

## Contributing

When contributing to this project:

1. Test changes locally using `langgraph_loan_local.py`
2. Ensure all tools return proper data types
3. Update documentation for any new features
4. Test deployment to AgentCore Runtime
5. Verify response parsing works correctly

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.

## Support

For issues and questions:

1. Check CloudWatch logs for runtime errors
2. Verify SageMaker endpoint is healthy
3. Ensure IAM permissions are correctly configured
4. Test locally before deploying to AgentCore
