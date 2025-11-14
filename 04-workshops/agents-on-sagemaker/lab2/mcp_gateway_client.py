
import json
import boto3
from datetime import datetime

class MCPGatewayClient:
    """Client for AgentCore MCP Gateway integration"""
    
    def __init__(self, lambda_function_name='research-report-mcp-server'):
        self.lambda_client = boto3.client('lambda')
        self.function_name = lambda_function_name
    
    def call_mcp_tool(self, tool_name, arguments):
        """Call MCP tool via Lambda function"""
        payload = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                Payload=json.dumps({
                    'body': json.dumps(payload)
                })
            )
            
            result = json.loads(response['Payload'].read())
            return json.loads(result['body'])
            
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"MCP Gateway Error: {str(e)}"
                }]
            }
    
    def save_research_report(self, content, bucket_name, format_type='markdown'):
        """Save research report using MCP Gateway"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.{format_type}"
        
        result = self.call_mcp_tool(
            "save_research_report",
            {
                "content": content,
                "filename": filename,
                "bucket": bucket_name,
                "format": format_type
            }
        )
        
        return result
