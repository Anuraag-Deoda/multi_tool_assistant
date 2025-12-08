from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    """Base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.openai_parameters
            }
        }
    
    @property
    def openai_parameters(self) -> Dict[str, Any]:
        """Tool parameters schema for OpenAI (default: same as Gemini)"""
        return self.parameters
    
    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert tool to Gemini function format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
