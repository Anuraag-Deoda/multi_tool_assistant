from tools.base_tool import BaseTool
from typing import Dict

class FileManagerTool(BaseTool):
    """File management tool (create, read, append)"""
    
    def __init__(self):
        self.files = {}  # In-memory storage
    
    @property
    def name(self) -> str:
        return "file_operation"
    
    @property
    def description(self) -> str:
        return "Manage files: create, read, or append content"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                "operation": {
                    "type": "STRING",
                    "enum": ["create", "read", "append"],
                    "description": "Operation to perform"
                },
                "filename": {
                    "type": "STRING",
                    "description": "Name of the file"
                },
                "content": {
                    "type": "STRING",
                    "description": "Content to write/append (not needed for read)"
                }
            },
            "required": ["operation", "filename"]
        }
    
    def execute(self, operation: str, filename: str, content: str = "") -> dict:
        """Execute file operation"""
        print(f"📄 File operation: {operation} on {filename}")
        
        if operation == "create":
            self.files[filename] = content
            return {"success": True, "message": f"Created '{filename}'"}
        
        elif operation == "read":
            if filename in self.files:
                return {"success": True, "content": self.files[filename]}
            return {"success": False, "error": f"File '{filename}' not found"}
        
        elif operation == "append":
            if filename in self.files:
                self.files[filename] += "\n" + content
                return {"success": True, "message": f"Appended to '{filename}'"}
            return {"success": False, "error": f"File '{filename}' not found"}
        
        return {"success": False, "error": "Invalid operation"}
