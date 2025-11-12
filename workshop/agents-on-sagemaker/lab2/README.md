# Bedrock Agent Core with MCP Gateway Integration - Complete Guide

This notebook demonstrates a complete end-to-end implementation of AWS Bedrock Agent Core integrated with MCP Gateway for intelligent research report generation and structured PDF output.

## Architecture Overview

```
User Query → Bedrock Agent Core → Internet Search → Qwen Analysis → MCP Gateway → Lambda PDF Generator → S3 Storage
```

## Components

1. **Bedrock Agent Core Runtime**: Main research agent with Qwen integration
2. **Internet Search Tool**: Real-time information gathering
3. **Qwen Model Integration**: AI-powered analysis via SageMaker endpoints
4. **MCP Gateway**: Secure Lambda function orchestration
5. **PDF Generation Lambda**: Structured report creation
6. **S3 Storage**: Persistent report storage


## Tutorial Details

| Information          | Details                                                   |
|:---------------------|:----------------------------------------------------------|
| Tutorial type        | Interactive End-to-End                                    |
| AgentCore components | AgentCore Runtime + AgentCore Gateway                     |
| Agentic Framework    | Bedrock Agent Core with LangChain                         |
| Gateway Target type  | AWS Lambda (PDF Generation)                               |
| Inbound Auth         | AWS IAM                                                   |
| Outbound Auth        | AWS IAM                                                   |
| LLM model            | Qwen via SageMaker Endpoint                               |
| Tutorial components  | Research Agent + MCP Gateway + PDF Generation             |
| Tutorial vertical    | Research and Document Generation                          |
| Example complexity   | Advanced                                                  |
| SDK used             | boto3, bedrock-agentcore-starter-toolkit                 |

## Prerequisites

- SageMaker Studio environment
- AWS CLI configured with appropriate permissions
- Python 3.8+ environment
- Qwen model deployed on SageMaker endpoint
- Docker support in SageMaker Studio

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
