from tools.base_tool import BaseTool
from io import StringIO
import sys

class PythonExecutorTool(BaseTool):
    """Safe Python code execution tool"""
    
    @property
    def name(self) -> str:
        return "run_python"
    
    @property
    def description(self) -> str:
        return "Execute Python code safely. Useful for calculations, data processing, and algorithms."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"],
            "additionalProperties": False
        }
    
    def execute(self, code: str) -> dict:
        """Execute Python code in restricted environment"""
        print(f"🐍 Executing Python code")
        
        # Safe built-ins
        safe_builtins = {
            'print': print, 'len': len, 'range': range, 'sum': sum,
            'min': min, 'max': max, 'abs': abs, 'round': round,
            'sorted': sorted, 'enumerate': enumerate, 'zip': zip,
            'map': map, 'filter': filter, 'list': list, 'dict': dict,
            'set': set, 'tuple': tuple, 'str': str, 'int': int,
            'float': float, 'bool': bool
        }
        
        try:
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()
            
            # Execute
            exec(code, {"__builtins__": safe_builtins})
            
            # Restore stdout
            sys.stdout = old_stdout
            output = captured.getvalue()
            
            return {
                "success": True,
                "output": output if output else "Code executed (no output)",
                "code": code
            }
        except Exception as e:
            sys.stdout = old_stdout
            return {
                "success": False,
                "error": str(e),
                "code": code
            }
