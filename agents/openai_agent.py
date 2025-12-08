from agents.base_agent import BaseAgent
from openai import OpenAI
import json

class OpenAIAgent(BaseAgent):
    """OpenAI-powered agent with function calling"""
    
    def __init__(self, tools, config):
        super().__init__(tools, config)
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
    
    def chat(self, message: str) -> str:
        """Chat with OpenAI agent"""
        print(f"\n{'='*60}")
        print(f"User: {message}")
        print(f"{'='*60}\n")
        
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Prepare tools for OpenAI
        tools_formatted = [tool.to_openai_format() for tool in self.tools.values()]
        
        # Agentic loop
        iteration = 0
        while iteration < self.config.MAX_ITERATIONS:
            iteration += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools_formatted,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if function calling is needed
            if assistant_message.tool_calls:
                # Add assistant message with tool calls
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Calling tool: {function_name}")
                    print(f"📥 Arguments: {function_args}")
                    
                    # Execute tool
                    result = self.execute_tool(function_name, function_args)
                    
                    print(f"📤 Result: {result}\n")
                    
                    # Add tool result
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Continue loop to get final response
                continue
            
            else:
                # No more tool calls, return final answer
                final_response = assistant_message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                print(f"{'='*60}")
                print(f"Assistant: {final_response}")
                print(f"{'='*60}\n")
                
                return final_response
        
        return "Max iterations reached"