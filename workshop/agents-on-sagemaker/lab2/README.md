# AIM406 - Bedrock Agent Core with Qwen Integration

A comprehensive implementation of AWS Bedrock Agent Core integrated with Qwen language model for intelligent research report generation.

## Overview

This project demonstrates how to build and deploy a production-ready AI research agent using:
- **AWS Bedrock Agent Core** for scalable agent deployment
- **Qwen Language Model** via SageMaker endpoints for intelligent analysis
- **Internet Search Integration** for real-time information gathering
- **Clean Report Generation** with professional formatting

## Architecture

```
User Query → Bedrock Agent Core → Internet Search → Qwen Analysis → Formatted Report
```

## Features

- 🔍 **Real-time Research**: Internet search with multiple authoritative sources
- 🧠 **AI Analysis**: Qwen model integration for intelligent insights
- 📊 **Professional Reports**: Clean, structured output without emojis
- 🚀 **Scalable Deployment**: Bedrock Agent Core with proper IAM permissions
- 🧪 **Local Testing**: Test functionality before cloud deployment
- 🧹 **Resource Management**: Automated cleanup to prevent unnecessary costs

## Quick Start

### Prerequisites
- AWS CLI configured
- SageMaker Studio or EC2 environment
- Python 3.8+
- Docker (for local development)

### 1. Setup Environment
```bash
# Clone and setup
git clone <repository>
cd aim406

# Install dependencies
pip install -r requirements.txt
```

### 2. Create IAM Role
```python
# Run the IAM role creation from the notebook
python create_iam_role.py
```

### 3. Test Locally
```bash
# Test Qwen integration locally
python test_working.py
```

### 4. Deploy to Bedrock Agent Core
```bash
# Deploy using CodeBuild
python deploy.py
```

### 5. Test Deployed Agent
```python
# Update agent ARN and test
python test_client.py
```

## Project Structure

```
aim406/
├── README.md                           # This file
├── Bedrock_Agent_Core_Complete_Guide.ipynb  # Complete tutorial notebook
├── main.py                            # Bedrock Agent Core application
├── qwen_client.py                     # SageMaker endpoint client
├── agent/
│   ├── __init__.py
│   └── tools.py                       # Internet search functionality
├── test_working.py                    # Local testing script
├── deploy.py                          # Deployment automation
├── test_client.py                     # Remote testing client
├── response_parser.py                 # Response formatting utility
├── cleanup.py                         # Resource cleanup
└── requirements.txt                   # Dependencies
```

## Key Components

### 1. Qwen Client (`qwen_client.py`)
Handles communication with SageMaker Qwen endpoint:
```python
client = QwenClient("your-endpoint-name")
response = client.invoke(messages)
```

### 2. Internet Search (`agent/tools.py`)
Provides real-time information gathering:
```python
results = internet_search(query="latest AI news", max_results=5)
```

### 3. Main Application (`main.py`)
Bedrock Agent Core entrypoint with 3-step workflow:
1. Trigger search based on user query
2. Execute internet search for current information
3. Generate AI-powered analysis using Qwen

### 4. Response Parser (`response_parser.py`)
Cleans and formats streaming responses:
```python
clean_report = parse_bedrock_response(raw_response)
```

## Configuration

### Environment Variables
```bash
export AWS_REGION=us-east-1
export QWEN_ENDPOINT_NAME=your-qwen-endpoint
```

### IAM Permissions Required
- `sagemaker:InvokeEndpoint` - For Qwen model access
- `bedrock:InvokeModel` - For Bedrock services
- `logs:CreateLogGroup` - For CloudWatch logging

## Usage Examples

### Local Testing
```python
from test_working import test_local_qwen

# Test with custom query
result = test_local_qwen("What are the latest developments in quantum computing?")
print(result)
```

### Remote API Call
```python
import boto3
import json

client = boto3.client('bedrock-agentcore', region_name='us-east-1')
response = client.invoke_agent_runtime(
    agentRuntimeArn='your-agent-arn',
    runtimeSessionId='unique-session-id',
    payload=json.dumps({"prompt": "Research AI trends 2024"}),
    qualifier="DEFAULT"
)
```

### Response Parsing
```python
from response_parser import parse_bedrock_response

# Parse streaming response
clean_report = parse_bedrock_response(response_body)
print(clean_report)
```

## Sample Output

```
Comprehensive Research Report

Research Query:
==================================================
What is the latest in Quantum Computing today?

Sources Analyzed:
==================================================
- The Quantum Insider: Quantum Computing News & Top Stories
- Quantum Computers News -- ScienceDaily
- Google Quantum AI
- IBM Quantum Computing
- MIT Quantum Computing News

Key Findings:
==================================================
The Quantum Insider: Quantum Companies Join Forces in Italy's New Quantum Alliance
ScienceDaily: Scientists may have uncovered the missing piece of quantum computing
Google Quantum AI: Willow chip represents major step toward fault-tolerant quantum computing

Analysis Summary:
==================================================
[Detailed AI-generated analysis from Qwen model...]

Research Methodology:
==================================================
- Conducted targeted internet search via Qwen endpoint
- Analyzed multiple authoritative sources
- Generated insights using Qwen model
- Provided source citations for verification
```

## Deployment Options

### 1. SageMaker Studio (Recommended)
- Built-in Docker support
- Integrated AWS services
- Easy debugging and development

### 2. EC2 Instance
- Full control over environment
- Custom Docker configurations
- Cost-effective for long-running deployments

### 3. Local Development
- Test functionality locally
- Debug before cloud deployment
- Faster iteration cycles

## Troubleshooting

### Common Issues

**1. Qwen Endpoint Unavailable**
```
Error: Analysis completed (endpoint unavailable)
```
- Check endpoint name in configuration
- Verify IAM permissions for SageMaker access
- Ensure endpoint is in "InService" status

**2. Streaming Errors**
```
Error: ResponseStreamingError
```
- Use non-streaming response reading
- Check network connectivity
- Verify Bedrock Agent Core deployment

**3. Permission Denied**
```
Error: AccessDenied
```
- Update IAM role with required permissions
- Check resource ARNs in policies
- Verify role trust relationships

### Debug Commands
```bash
# Check SageMaker endpoints
aws sagemaker list-endpoints --region us-east-1

# Check Bedrock agents
aws bedrock-agent list-agents --region us-east-1

# View CloudWatch logs
aws logs tail /aws/bedrock-agentcore/runtimes/your-agent --follow
```

## Cost Optimization

### Resource Management
- Use `cleanup.py` to remove unused resources
- Stop SageMaker endpoints when not in use
- Monitor CloudWatch for usage patterns

### Estimated Costs
- **Bedrock Agent Core**: ~$0.10 per 1K requests
- **SageMaker Endpoint**: ~$0.50-2.00 per hour (depending on instance)
- **CloudWatch Logs**: ~$0.50 per GB ingested

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 **Documentation**: See `Bedrock_Agent_Core_Complete_Guide.ipynb`
- 🐛 **Issues**: Open GitHub issue with detailed description
- 💬 **Discussions**: Use GitHub Discussions for questions

## Acknowledgments

- AWS Bedrock Agent Core team for the deployment framework
- Qwen model developers for the language model
- AWS SageMaker team for endpoint infrastructure

---

**Built with ❤️ for AIM406 - Advanced AI Agent Development**
