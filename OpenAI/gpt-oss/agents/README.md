# Building an Intelligent Stock Analysis Agent: LangGraph + Amazon SageMaker Jumpstart + Amazon Bedrock AgentCore

This project demonstrates how to build and deploy an intelligent stock analysis agent using LangGraph, Amazon SageMaker with OpenAI GPT-OSS models, and Amazon Bedrock AgentCore Runtime. The agent transforms simple stock ticker requests into comprehensive stock analysis with professional PDF reports for educational and documentation purposes.

## Overview

The stock analysis agent performs a complete three-step research workflow:

1. **Stock Data Gathering**: Collects real-time market data, financial metrics, and company information
2. **Performance Analysis**: Performs comprehensive technical and fundamental analysis with risk assessment
3. **Report Generation**: Creates professional PDF reports with detailed analysis and uploads to Amazon S3

## Architecture

The solution combines several AWS services and frameworks:

- **LangGraph**: Multi-agent orchestration framework for complex investment workflows
- **Amazon SageMaker**: Hosts the OpenAI GPT-OSS model via JumpStart
- **Amazon Bedrock AgentCore Runtime**: Scalable cloud deployment platform
- **Amazon S3**: Automated PDF report storage with organized folder structure
- **Custom Tools**: Specialized stock analysis functions for data gathering, analysis, and decision-making
- **yfinance API**: Real-time stock market data integration

## Prerequisites

### 1. Deploy GPT-OSS Model on Amazon SageMaker

Before using this project, you must deploy the OpenAI GPT-OSS model:

1. Open the notebook `openai_gpt_oss.ipynb` in the `./deploy_sagemaker/gpt-oss` folder
2. Follow the instructions to deploy the GPT-OSS model to a SageMaker endpoint
3. Note the endpoint name for configuration

### 2. Required Permissions

The agent requires specific IAM permissions to access AWS services. Use the provided role creation function or ensure your execution role has:

- SageMaker endpoint invocation permissions
- S3 bucket read/write permissions for PDF storage
- ECR repository management and access
- CloudWatch logs write permissions
- Bedrock AgentCore workload access
- X-Ray tracing and CloudWatch metrics permissions

### 3. S3 Bucket Setup

Create an S3 bucket for PDF report storage or use an existing bucket. The agent will organize reports in date-based folders: `YYYY/MM/DD/`

## Project Structure

```
langgraph-on-amazon-sagemaker/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── Stock_Analyzer_GPT_OSS_Agent_Core.ipynb    # Main tutorial notebook
```

## Key Components

### Stock Analysis Tools

The agent uses three specialized tools:

#### 1. gather_stock_data
- Collects real-time stock data using yfinance API
- Retrieves current price, market cap, financial ratios, and historical performance
- Gathers company information including sector, industry, and trading metrics
- Simulates recent news headlines (can be enhanced with real news APIs)

#### 2. analyze_stock_performance
- Performs comprehensive technical analysis (price trends, volatility, momentum)
- Conducts fundamental analysis (P/E ratios, profit margins, dividend yields)
- Calculates risk assessment using beta analysis and volatility metrics
- Provides descriptive analysis for informational purposes only

#### 3. generate_stock_report
- Creates professional PDF reports with comprehensive analysis
- Uploads reports to Amazon S3 with organized folder structure
- Provides detailed documentation for educational purposes
- Generates executive summaries and risk assessments

### SageMaker Integration

The `SagemakerLLMWrapper` class provides:
- Integration between LangGraph and SageMaker endpoints
- Harmony format payload handling for GPT-OSS models
- Automatic tool execution for stock analysis requests
- Proper response formatting for LangGraph compatibility

## Getting Started

### 1. Local Development and Testing

For local experimentation, use the `langgraph_stock_local.py` file:

```python
from langgraph_stock_local import langgraph_stock_sagemaker

# Test the agent locally
result = langgraph_stock_sagemaker({
    "prompt": "Analyze AAPL stock for investment"
})
print(result)
```

### 2. Create IAM Role and S3 Bucket

Create the required IAM role with proper permissions:

```python
from create_stock_agentcore_role import create_stock_agentcore_role, create_s3_bucket_if_not_exists

# Create S3 bucket for PDF reports
bucket_name = create_s3_bucket_if_not_exists("your-stock-reports-bucket")

# Create IAM role with all necessary permissions
role_arn = create_stock_agentcore_role(
    role_name="StockAnalysisAgentCoreRole",
    sagemaker_endpoint_name="your-endpoint-name",
    s3_bucket_name=bucket_name,
    region="us-west-2"
)
```

### 3. Generate Deployment File

Create the AgentCore deployment file dynamically:

```python
from create_agentcore_deployment import create_agentcore_deployment_file

# Generate deployment file with your specific configuration
create_agentcore_deployment_file(
    endpoint_name="your-sagemaker-endpoint",
    bucket_name="your-s3-bucket",
    filename="langgraph_stock_sagemaker_gpt_oss.py"
)
```

### 4. Deploy to AgentCore Runtime

Use the Bedrock AgentCore Starter Toolkit to deploy:

```python
from bedrock_agentcore_starter_toolkit import Runtime

agentcore_runtime = Runtime()

# Configure the agent
response = agentcore_runtime.configure(
    entrypoint="langgraph_stock_sagemaker_gpt_oss.py",
    execution_role=role_arn,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region="us-west-2",
    agent_name="stock_analyzer_agent"
)

# Deploy to the cloud
launch_result = agentcore_runtime.launch(local=False, local_build=False)
```

### 5. Invoke the Deployed Agent

```python
import boto3
import json

agentcore_client = boto3.client('bedrock-agentcore', region_name='us-west-2')

response = agentcore_client.invoke_agent_runtime(
    agentRuntimeArn=launch_result.agent_arn,
    qualifier="DEFAULT",
    payload=json.dumps({
        "prompt": "Analyze TSLA stock for investment"
    })
)

# Parse and display results
from stock_response_parser import parse_bedrock_agentcore_stock_response
stock_analysis = parse_bedrock_agentcore_stock_response(response)
```

## Response Format

The agent returns a comprehensive analysis with three main sections:

### Step 1 - Stock Data Gathering
```
STOCK DATA GATHERING REPORT:
================================
Stock Symbol: AAPL
Company Name: Apple Inc.
Sector: Technology
Industry: Consumer Electronics

CURRENT MARKET DATA:
- Current Price: $229.31
- Market Cap: $3,403,051,958,272
- 52-Week High: $259.18
- 52-Week Low: $168.80
- YTD Return: 1.03%
- Volatility (Annualized): 32.22%

FINANCIAL METRICS:
- P/E Ratio: 34.80
- Forward P/E: 27.59
- Price-to-Book: 51.75
- Dividend Yield: 0.46%
- Revenue (TTM): $408,624,988,160
- Profit Margin: 24.30%
```

### Step 2 - Performance Analysis
```
STOCK PERFORMANCE ANALYSIS:
===============================
Stock: AAPL | Current Price: $239.78

TECHNICAL ANALYSIS:
- Price Trend: SLIGHT UPTREND
- YTD Performance: 8.33%

FUNDAMENTAL ANALYSIS:
- P/E Ratio: 36.44073
- Profit Margin: 24.30%
- Dividend Yield: 44.00%
- Beta: 1.109

KEY OBSERVATIONS:
• P/E ratio suggests potential overvaluation
• Excellent profit margins
• High dividend yield

ANALYST SUMMARY:
Based on technical and fundamental analysis, AAPL shows slight uptrend with medium volatility profile.
The analysis reflects current market conditions and financial performance metrics for informational purposes.

DISCLAIMER: This analysis is for informational purposes only and does not constitute investment advice.
```

### Step 3 - Report Generation
```
STOCK REPORT GENERATION:
===============================
Stock: AAPL (Apple Inc.)
Sector: Technology
Current Price: $239.78

REPORT SUMMARY:
- Technical Analysis: 8.33% YTD performance
- Report Type: Comprehensive stock analysis for informational purposes
- Generated: 2025-09-04 23:11:55

PDF report uploaded to S3: s3://surya-495365983931/2025/09/04/AAPL_Stock_Report_20250904_231155.pdf

REPORT CONTENTS:
• Executive Summary with key metrics
• Detailed market data and financial metrics
• Technical and fundamental analysis
• Risk assessment and observations
• Professional formatting for documentation

DISCLAIMER: This report is for informational and educational purposes only.
It does not constitute investment advice or recommendations.
```

## Configuration

### Environment Variables

The agent can be configured through environment variables:

- `SAGEMAKER_ENDPOINT_NAME`: Name of your SageMaker endpoint
- `S3_BUCKET_NAME`: S3 bucket for PDF report storage
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

### Investment Decision Logic

Modify the recommendation criteria in `make_investment_decision`:

- Technical score thresholds (based on YTD performance trends)
- Fundamental score criteria (P/E ratios, profit margins, dividend yields)
- Risk level assessments (volatility and beta analysis)

## Monitoring and Debugging

### CloudWatch Logs

Agent execution logs are available in CloudWatch:

```bash
aws logs tail "/aws/bedrock-agentcore/runtimes/your-agent-id-DEFAULT" --follow --region us-west-2
```

### Response Parsing

Use the provided parsing utilities to extract structured information:

```python
from stock_response_parser import parse_bedrock_agentcore_stock_response

# Parse and display formatted results
analysis_data = parse_bedrock_agentcore_stock_response(invoke_response)

# Access specific data programmatically
decision = analysis_data['final_decision']
rating = analysis_data['investment_rating']
pdf_path = analysis_data['pdf_status']
```

### Notebook-Friendly Parser

For Jupyter notebook usage:

```python
from notebook_stock_parser import parse_bedrock_agentcore_stock_response

# Clean output without emojis for notebooks
stock_analysis = parse_bedrock_agentcore_stock_response(invoke_response)
```

## Error Handling

The agent includes comprehensive error handling:

- **JSON parsing errors**: Falls back to plain text display
- **Tool execution errors**: Returns error messages with context
- **SageMaker endpoint errors**: Provides debugging information
- **yfinance API errors**: Handles missing or invalid stock symbols gracefully
- **S3 upload errors**: Reports PDF generation/upload failures with details
- **Missing data**: Handles incomplete stock information appropriately

## Security Considerations

- **IAM roles**: Use least-privilege permissions for production deployments
- **Data handling**: Ensure financial data is handled according to compliance requirements
- **Endpoint access**: Restrict SageMaker endpoint access to authorized services only
- **S3 bucket security**: Configure appropriate bucket policies and encryption
- **Logging**: Be mindful of sensitive data in CloudWatch logs

## Customization

### Adding New Analysis Tools

To add additional analysis capabilities:

1. Create a new function decorated with `@tool`
2. Add the tool to the `tools` list in `create_agent()`
3. Update the `SagemakerLLMWrapper` to handle the new tool
4. Redeploy the agent

### Modifying Analysis Logic

Update the analysis criteria in `analyze_stock_performance()`:

```python
# Custom analysis logic
if custom_criteria:
    fundamental_factors.append("Custom criteria met")
```

### Enhancing Data Sources

Replace simulated news with real APIs:

```python
# Example: Integrate with news API
import requests
news_response = requests.get(f"https://api.newsapi.org/v2/everything?q={symbol}")
recent_news = [article['title'] for article in news_response.json()['articles'][:5]]
```

### PDF Report Customization

Modify the PDF generation in `create_and_upload_stock_report_pdf()`:

```python
# Add charts, additional sections, or custom branding
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot

# Create price chart
chart = LinePlot()
# ... configure chart with historical data
story.append(chart)
```

## Troubleshooting

### Common Issues

1. **"Unsupported message type" error**: Ensure `AIMessage` is returned from tools
2. **SageMaker endpoint timeout**: Check endpoint status and increase timeout
3. **Permission denied**: Verify IAM role has all required permissions
4. **Tool not executing**: Check tool detection logic in `SagemakerLLMWrapper`
5. **S3 upload failures**: Verify bucket permissions and existence
6. **yfinance errors**: Check internet connectivity and stock symbol validity

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

- **Batch processing**: Process multiple stock symbols in a single request
- **Caching**: Cache stock data for repeated analysis within trading hours
- **Async processing**: Use async patterns for high-throughput scenarios
- **Model optimization**: Fine-tune model parameters for your use case
- **S3 optimization**: Use multipart uploads for large PDF files

## Contributing

When contributing to this project:

1. Test changes locally using `langgraph_stock_local.py`
2. Ensure all tools return proper data types
3. Update documentation for any new features
4. Test deployment to AgentCore Runtime
5. Verify response parsing works correctly
6. Test S3 integration and PDF generation

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.

## Support

For issues and questions:

1. Check CloudWatch logs for runtime errors
2. Verify SageMaker endpoint is healthy
3. Ensure IAM permissions are correctly configured
4. Test S3 bucket access and permissions
5. Validate stock symbols and yfinance connectivity
6. Test locally before deploying to AgentCore

## Sample Use Cases

### Portfolio Analysis
```python
# Analyze multiple stocks for portfolio documentation
stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
for symbol in stocks:
    result = langgraph_stock_sagemaker({"prompt": f"Analyze {symbol} stock"})
    # Process results for portfolio documentation
```

### Risk Assessment
```python
# Focus on risk analysis for existing holdings
result = langgraph_stock_sagemaker({
    "prompt": "Analyze NVDA stock"
})
```

### Sector Analysis
```python
# Compare stocks within the same sector
result = langgraph_stock_sagemaker({
    "prompt": "Analyze AAPL stock and compare with technology sector trends"
})
```

This comprehensive stock analysis agent provides institutional-quality research capabilities, combining real-time market data, advanced AI analysis, and professional report generation in a scalable, cloud-native architecture for educational and documentation purposes.
