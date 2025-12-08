from tools.base_tool import BaseTool
import requests

class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo API"""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information and recent events"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    def execute(self, query: str) -> dict:
        """Execute web search"""
        print(f"🔍 Searching: {query}")
        
        try:
            # Using DuckDuckGo Instant Answer API (free, no key needed)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            # Extract relevant information
            abstract = data.get("AbstractText", "")
            related = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:3]]
            
            result = {
                "query": query,
                "abstract": abstract or "No direct answer found",
                "related_topics": related,
                "source": data.get("AbstractURL", "")
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Search failed. Using simulated data.",
                "query": query,
                "results": [f"Simulated result for: {query}"]
            }