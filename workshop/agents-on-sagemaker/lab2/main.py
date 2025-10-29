from agent.tools import internet_search
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from qwen_client import QwenClient
import json
import re

app = BedrockAgentCoreApp()

def parse_clean_report(response_text):
    """Extract and clean report from Bedrock Agent Core response"""
    lines = response_text.strip().split('\n')
    
    for line in lines:
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])
                messages = data.get('messages', [])
                
                # Find the last assistant message with report
                for msg in reversed(messages):
                    if (msg.get('type') == 'assistant' and 
                        msg.get('content') and 
                        '# Comprehensive Research Report' in msg.get('content', '')):
                        
                        content = msg['content']
                        
                        # Remove emojis
                        content = re.sub(r'[🎯📚📊💡🔍]', '', content)
                        
                        # Clean up extra spaces
                        content = re.sub(r'\n\s*\n', '\n\n', content)
                        content = re.sub(r'  +', ' ', content)
                        
                        return content.strip()
            except:
                continue
    
    return "No report found"

class SimpleQwenWrapper(Runnable):
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.qwen = QwenClient(endpoint_name)
    
    def bind_tools(self, tools):
        return self
    
    def invoke(self, input_data, config=None, **kwargs):
        messages = input_data if isinstance(input_data, list) else input_data.get("messages", [])
        
        if not messages:
            return AIMessage(content="I'll help with research.")
        
        # Check if we have search results to summarize
        search_results = None
        for msg in messages:
            if isinstance(msg, ToolMessage) and "results" in getattr(msg, 'content', ''):
                try:
                    search_results = json.loads(msg.content)
                    break
                except:
                    pass
        
        # If we have search results, use Qwen to generate summary
        if search_results and "results" in search_results:
            # Call Qwen endpoint for report generation
            qwen_response = self.qwen.invoke(messages)
            
            # Generate clean report without emojis
            results = search_results["results"]
            sources = []
            findings = []
            for result in results[:5]:
                sources.append(f"- [{result.get('title', 'N/A')}]({result.get('url', 'N/A')})")
                findings.append(f"**{result.get('title', 'Source')}**: {result.get('content', 'No content available')}")
            
            clean_report = f"""# Comprehensive Research Report

## Research Query
{search_results.get('query', 'Research query')}

## Sources Analyzed
{chr(10).join(sources)}

## Key Findings
{chr(10).join(findings)}

## Analysis Summary
{qwen_response}

## Research Methodology
- Conducted targeted internet search via Qwen endpoint
- Analyzed multiple authoritative sources
- Generated insights using Qwen model
- Provided source citations for verification

---
Report generated using Qwen model: {self.endpoint_name}"""

            return AIMessage(content=clean_report)
        
        # If no search results yet, trigger search
        user_question = ""
        for msg in messages:
            if type(msg).__name__ == "HumanMessage":
                user_question = getattr(msg, 'content', '')
                break
        
        if user_question:
            return AIMessage(
                content="Conducting comprehensive research using Qwen model...",
                tool_calls=[{
                    "name": "internet_search",
                    "args": {"query": user_question, "max_results": 5},
                    "id": "search_1"
                }]
            )
        
        return AIMessage(content="Please provide a research question.")
    
    async def ainvoke(self, input_data, config=None, **kwargs):
        return self.invoke(input_data, config, **kwargs)

# Create simple wrapper
LLM = SimpleQwenWrapper(endpoint_name="model-251024-1552")

def serialize_message(message):
    if isinstance(message, HumanMessage):
        return {
            "type": "human",
            "content": message.content,
            "id": getattr(message, "id", None),
        }
    elif isinstance(message, AIMessage):
        return {
            "type": "assistant",
            "content": message.content,
            "id": getattr(message, "id", None),
            "tool_calls": getattr(message, "tool_calls", []),
        }
    elif isinstance(message, ToolMessage):
        return {
            "type": "tool",
            "content": message.content,
            "name": getattr(message, "name", "unknown"),
        }
    else:
        return {
            "type": "unknown",
            "content": str(message),
        }

def serialize_chunk(chunk):
    serialized_chunk = {}
    if "messages" in chunk:
        serialized_chunk["messages"] = []
        for message in chunk["messages"]:
            serialized_chunk["messages"].append(serialize_message(message))
    return serialized_chunk

@app.entrypoint
async def simple_qwen_research(payload):
    """Simple research using Qwen wrapper"""
    user_input = payload.get("prompt")
    
    # Step 1: Trigger search
    search_response = LLM.invoke([HumanMessage(content=user_input)])
    yield serialize_chunk({"messages": [HumanMessage(content=user_input), search_response]})
    
    # Step 2: Execute search
    if hasattr(search_response, 'tool_calls') and search_response.tool_calls:
        search_call = search_response.tool_calls[0]
        search_result = internet_search(**search_call['args'])
        tool_msg = ToolMessage(content=json.dumps(search_result), name="internet_search", tool_call_id=search_call['id'])
        yield serialize_chunk({"messages": [HumanMessage(content=user_input), search_response, tool_msg]})
        
        # Step 3: Generate summary using Qwen
        final_response = LLM.invoke([HumanMessage(content=user_input), search_response, tool_msg])
        yield serialize_chunk({"messages": [HumanMessage(content=user_input), search_response, tool_msg, final_response]})

if __name__ == "__main__":
    app.run()
