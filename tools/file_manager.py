from tools.base_tool import BaseTool
from typing import Dict
import os

class FileManagerTool(BaseTool):
    """File management tool (create, read, append)"""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "file_operation"
    
    @property
    def description(self) -> str:
        return "Manage files: create, read, or append content"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "read", "append"],
                    "description": "Operation to perform"
                },
                "filename": {
                    "type": "string",
                    "description": "Name of the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write/append (not needed for read)"
                }
            },
            "required": ["operation", "filename"],
            "additionalProperties": False

        }
    
    def execute(self, operation: str, filename: str, content: str = "") -> dict:
        """Execute file operation"""
        print(f"📄 File operation: {operation} on {filename}")
        
        try:
            if operation == "create":
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"success": True, "message": f"Created '{filename}'"}
            
            elif operation == "read":
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        return {"success": True, "content": f.read()}
                return {"success": False, "error": f"File '{filename}' not found"}
            
            elif operation == "append":
                if os.path.exists(filename):
                    with open(filename, 'a', encoding='utf-8') as f:
                        f.write("\n" + content)
                    return {"success": True, "message": f"Appended to '{filename}'"}
                return {"success": False, "error": f"File '{filename}' not found"}
            
            return {"success": False, "error": "Invalid operation"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
