from agent.tools import internet_search
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from qwen_client import QwenClient
import json
import asyncio

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
LLM = SimpleQwenWrapper(endpoint_name="model-251029-2112")

async def simple_qwen_research(payload):
    """Simple research without deep agents - local version"""
    user_input = payload.get("prompt")
    
    # Step 1: Trigger search
    search_response = LLM.invoke([HumanMessage(content=user_input)])
    print("Step 1 - Search trigger:")
    print(f"Content: {search_response.content}")
    
    # Step 2: Execute search
    if hasattr(search_response, 'tool_calls') and search_response.tool_calls:
        search_call = search_response.tool_calls[0]
        search_result = internet_search(**search_call['args'])
        tool_msg = ToolMessage(content=json.dumps(search_result), name="internet_search", tool_call_id=search_call['id'])
        print("\nStep 2 - Tool execution completed")
        
        # Step 3: Generate summary
        final_response = LLM.invoke([HumanMessage(content=user_input), search_response, tool_msg])
        print("\nStep 3 - Final report:")
        print(final_response.content)

# Test locally
if __name__ == "__main__":
    query = "What is the latest in Quantum Computing today?"
    payload = {"prompt": query}
    asyncio.run(simple_qwen_research(payload))
