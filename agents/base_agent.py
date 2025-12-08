from abc import ABC, abstractmethod
from typing import List, Dict, Any
from tools.base_tool import BaseTool

class BaseAgent(ABC):
    """Base class for AI agents"""
    
    def __init__(self, tools: List[BaseTool], config: Any):
        self.tools = {tool.name: tool for tool in tools}
        self.config = config
        self.conversation_history = []
    
    @abstractmethod
    def chat(self, message: str) -> str:
        """Send a message and get response"""
        pass
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        if tool_name in self.tools:
            return self.tools[tool_name].execute(**arguments)
        return {"error": f"Tool '{tool_name}' not found"}
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

