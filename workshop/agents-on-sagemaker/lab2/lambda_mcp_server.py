
import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """Lambda MCP server for research report processing"""
    
    try:
        body = json.loads(event.get('body', '{}'))
        method = body.get('method')
        params = body.get('params', {})
    except:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON'})
        }
    
    if method == 'tools/list':
        return {
            'statusCode': 200,
            'body': json.dumps({
                "tools": [
                    {
                        "name": "save_research_report",
                        "description": "Save research report as structured document to S3",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Report content"},
                                "filename": {"type": "string", "description": "File name"},
                                "bucket": {"type": "string", "description": "S3 bucket name"},
                                "format": {"type": "string", "description": "Output format (markdown/json)", "default": "markdown"}
                            },
                            "required": ["content", "filename", "bucket"]
                        }
                    }
                ]
            })
        }
    
    elif method == 'tools/call':
        tool_name = params.get('name')
        args = params.get('arguments', {})
        
        if tool_name == 'save_research_report':
            try:
                result = save_structured_report(
                    args['content'],
                    args['filename'], 
                    args['bucket'],
                    args.get('format', 'markdown')
                )
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        "content": [{
                            "type": "text",
                            "text": result
                        }]
                    })
                }
            except Exception as e:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        "content": [{
                            "type": "text", 
                            "text": f"Error: {str(e)}"
                        }]
                    })
                }
    
    return {
        'statusCode': 400,
        'body': json.dumps({'error': 'Unknown method'})
    }

def save_structured_report(content, filename, bucket, format_type='markdown'):
    """Save structured research report to S3"""
    s3_client = boto3.client('s3')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    if format_type == 'json':
        # Structure as JSON
        structured_content = {
            "metadata": {
                "generated": timestamp,
                "type": "research_report",
                "format": "structured_json"
            },
            "content": content,
            "processing_info": {
                "processor": "AgentCore MCP Gateway",
                "lambda_function": "research-report-processor"
            }
        }
        
        body = json.dumps(structured_content, indent=2).encode('utf-8')
        content_type = 'application/json'
        
    else:
        # Structure as enhanced markdown
        structured_content = f"""---
title: Research Report
generated: {timestamp}
type: research_report
processor: AgentCore MCP Gateway
format: structured_markdown
---

{content}

---

## Processing Information

- **Generated**: {timestamp}
- **Processor**: AgentCore MCP Gateway
- **Lambda Function**: research-report-processor
- **Format**: Structured Markdown with YAML frontmatter

This report was automatically generated and processed through AWS Bedrock Agent Core with MCP Gateway integration.
"""
        
        body = structured_content.encode('utf-8')
        content_type = 'text/markdown'
    
    # Upload to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=filename,
        Body=body,
        ContentType=content_type,
        Metadata={
            'generated-by': 'agentcore-mcp-gateway',
            'report-type': 'research-analysis',
            'timestamp': timestamp
        }
    )
    
    return f"Structured report saved successfully to s3://{bucket}/{filename} (format: {format_type})"
