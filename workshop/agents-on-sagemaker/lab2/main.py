from bedrock_agentcore.runtime import BedrockAgentCoreApp
import json
import boto3
from agent.tools import internet_search
import requests
from bs4 import BeautifulSoup
import io

app = BedrockAgentCoreApp()

# Get account info
account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name or 'us-west-2'

class LineIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])

def fetch_content_from_url(url):
    """Fetch actual content from Wikipedia URLs"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navbox', '.infobox']):
            element.decompose()
        
        # Get main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            text = content_div.get_text()
        else:
            text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Return first 2000 characters
        return text[:2000] if text else "No content extracted"
        
    except Exception as e:
        return f"Could not fetch content: {str(e)}"

def stream_response(endpoint_name, inputs, max_tokens=1000, temperature=0.3, top_p=0.8):
    """Use SageMaker streaming inference"""
    try:
        smr_client = boto3.client('sagemaker-runtime', region_name=region)
        
        body = {
          "messages": [
            {"role": "user", "content": [{"type": "text", "text": inputs}]}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        resp = smr_client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            Body=json.dumps(body),
            ContentType="application/json",
        )

        event_stream = resp["Body"]
        start_json = b"{"
        full_response = ""

        for line in LineIterator(event_stream):
            if line != b"" and start_json in line:
                data = json.loads(line[line.find(start_json):].decode("utf-8"))
                token_text = data['choices'][0]['delta'].get('content', '')
                full_response += token_text

        return full_response
        
    except Exception as e:
        return f"Model analysis error: {str(e)}"

def call_mcp_gateway(content, bucket):
    """Call MCP Gateway Lambda to save report"""
    lambda_client = boto3.client('lambda')
    
    payload = {
        "method": "tools/call",
        "params": {
            "name": "save_research_report",
            "arguments": {
                "content": content,
                "filename": f"research_report_{int(account_id)}.md",
                "bucket": bucket,
                "format": "markdown"
            }
        }
    }
    
    try:
        response = lambda_client.invoke(
            FunctionName='research-report-mcp-server',
            Payload=json.dumps({'body': json.dumps(payload)})
        )
        
        result = json.loads(response['Payload'].read())
        return json.loads(result['body'])
    except Exception as e:
        return {"content": [{"type": "text", "text": f"MCP Error: {str(e)}"}]}

@app.entrypoint
async def simple_qwen_research(payload):
    """Complete research pipeline: tools.py -> fetch content -> model analysis -> MCP report"""
    user_input = payload.get("prompt", "test query")
    
    # Step 1: Use tools.py to get search results
    search_results = internet_search(user_input, max_results=3)
    
    # Step 2: Fetch actual content from the URLs returned by tools.py
    enriched_results = []
    all_content = ""
    
    for result in search_results.get('results', []):
        url = result.get('url', '')
        title = result.get('title', '')
        source = result.get('source', '')
        
        print(f"Fetching content from: {url}")
        actual_content = fetch_content_from_url(url)
        
        enriched_results.append({
            'title': title,
            'source': source,
            'url': url,
            'content': actual_content
        })
        
        all_content += f"\n\nSource: {source} - {title}\n{actual_content}"
    
    # Step 3: Use SageMaker model to analyze the content
    if all_content.strip():
        prompt = f"""Analyze this research content about "{user_input}" and provide:

SUMMARY: [Write a 2-3 sentence summary of the key findings]
KEY POINTS:
• [First key finding]
• [Second key finding] 
• [Third key finding]
TRENDS: [Notable trends or developments mentioned]

Content to analyze:
{all_content[:3000]}"""

        model_analysis = stream_response("model-251113-1249", prompt)
    else:
        model_analysis = "SUMMARY: No content available for analysis. KEY POINTS: • No sources found TRENDS: Unable to determine trends."
    
    # Step 4: Parse model analysis
    summary = "Analysis completed"
    key_points = []
    trends = "Current developments noted"
    
    if "SUMMARY:" in model_analysis:
        parts = model_analysis.split("SUMMARY:")[1]
        if "KEY POINTS:" in parts:
            summary = parts.split("KEY POINTS:")[0].strip()
            key_section = parts.split("KEY POINTS:")[1]
            if "TRENDS:" in key_section:
                key_text = key_section.split("TRENDS:")[0]
                trends = key_section.split("TRENDS:")[1].strip()
            else:
                key_text = key_section
            
            key_points = [line.strip() for line in key_text.split('\n') if line.strip().startswith('•')]
    
    # Step 5: Generate comprehensive report
    sources_section = ""
    detailed_findings = ""
    
    for i, result in enumerate(enriched_results, 1):
        title = result.get('title', 'N/A')
        url = result.get('url', 'N/A')
        source = result.get('source', 'Unknown')
        content = result.get('content', 'No content available')
        
        sources_section += f"{i}. **{source}**: [{title}]({url})\n"
        
        # Show actual content excerpt
        if content and len(content) > 100:
            excerpt = content[:400] + "..." if len(content) > 400 else content
            detailed_findings += f"**{title}** ({source})\n{excerpt}\n\n"
    
    key_points_section = "\n".join(key_points) if key_points else "• Analysis completed with available sources"
    
    report = f"""# Research Report: {user_input}

## Executive Summary
{summary}

## Sources Analyzed (via tools.py)
{sources_section}

## Key Findings & Insights
{key_points_section}

## Trends & Developments
{trends}

## Detailed Source Analysis
{detailed_findings}

## Research Pipeline
- **Step 1**: tools.py internet search
- **Step 2**: Content extraction from URLs
- **Step 3**: SageMaker model analysis ({"model-251113-1249"})
- **Step 4**: MCP Gateway S3 storage

---
*Complete Agent Core tutorial: tools.py → content fetch → model analysis → MCP report*"""

    # Step 6: Use MCP Gateway to save report to S3
    mcp_result = call_mcp_gateway(report, "sagemaker-us-west-2-471112832125")
    storage_status = mcp_result.get('content', [{}])[0].get('text', 'MCP processing completed')
    
    final_report = report + f"\n\n## MCP Gateway Status\n{storage_status}"
    
    yield {
        "messages": [{
            "type": "assistant", 
            "content": final_report
        }]
    }

if __name__ == "__main__":
    app.run()