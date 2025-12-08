from agents.base_agent import BaseAgent
import google.generativeai as genai
import json

class GeminiAgent(BaseAgent):
    """Google Gemini-powered agent with function calling"""
    
    def __init__(self, tools, config):
        super().__init__(tools, config)
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Convert tools to Gemini format
        tools_formatted = [tool.to_gemini_format() for tool in self.tools.values()]
        
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            tools=tools_formatted
        )
        self.chat_session = self.model.start_chat(history=[])
    
    def chat(self, message: str) -> str:
        """Chat with Gemini agent"""
        print(f"\n{'='*60}")
        print(f"User: {message}")
        print(f"{'='*60}\n")
        
        # Agentic loop
        iteration = 0
        response = self.chat_session.send_message(message)
        
        while iteration < self.config.MAX_ITERATIONS:
            iteration += 1
            
            # Check for function calls
            if response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    function_name = fc.name
                    function_args = dict(fc.args)
                    
                    print(f"🔧 Calling tool: {function_name}")
                    print(f"📥 Arguments: {function_args}")
                    
                    # Execute tool
                    result = self.execute_tool(function_name, function_args)
                    
                    print(f"📤 Result: {result}\n")
                    
                    # Send result back to Gemini
                    response = self.chat_session.send_message(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=function_name,
                                    response={"result": result}
                                )
                            )]
                        )
                    )
                    continue
            
            # Extract final text response
            final_response = response.text
            
            print(f"{'='*60}")
            print(f"Assistant: {final_response}")
            print(f"{'='*60}\n")
            
            return final_response
        
        return "Max iterations reached"