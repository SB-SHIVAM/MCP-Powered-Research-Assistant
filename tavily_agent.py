import os
from tavily import TavilyClient

async def tavily_agent(query: str) -> str:
    """Searches the web using Tavily."""
    try:
        # Fetching inside the function ensures dotenv has had time to load
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY is missing."

        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(query=query, search_depth="basic")
        
        results = [f"Title: {r['title']}\nContent: {r['content']}\n" for r in response.get('results', [])]
        return "\n".join(results) if results else "No web results found."
    except Exception as e:
        return f"Web search error: {str(e)}"