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
            "parameters": self._convert_to_gemini_schema(self.parameters)
        }

    def _convert_to_gemini_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert JSON schema types to uppercase for Gemini"""
        if not isinstance(schema, dict):
            return schema
            
        # Create a copy to avoid modifying original
        new_schema = schema.copy()
        
        # Handle 'type' field - convert to uppercase
        if "type" in new_schema and isinstance(new_schema["type"], str):
            new_schema["type"] = new_schema["type"].upper()
            
        # Remove additionalProperties as it's not supported in Gemini Schema
        if "additionalProperties" in new_schema:
            del new_schema["additionalProperties"]
            
        # Recurse into properties
        if "properties" in new_schema and isinstance(new_schema["properties"], dict):
            new_schema["properties"] = {
                k: self._convert_to_gemini_schema(v)
                for k, v in new_schema["properties"].items()
            }
            
        # Recurse into items (for arrays)
        if "items" in new_schema and isinstance(new_schema["items"], dict):
            new_schema["items"] = self._convert_to_gemini_schema(new_schema["items"])
            
        return new_schema
