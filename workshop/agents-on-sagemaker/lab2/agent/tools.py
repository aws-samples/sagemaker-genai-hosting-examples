from typing import Literal
from tavily import TavilyClient

TAVILY_API_KEY = "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"


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=TAVILY_API_KEY)
    return tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


if __name__ == "__main__":
    internet_search(query="WHat is langsmith?")
