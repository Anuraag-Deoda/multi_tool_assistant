# ============================================================================
# FILE: agents/gemini_agent.py (Updated with Memory Support)
# ============================================================================

from agents.base_agent import BaseAgent
from typing import List, Optional, Any
import json

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiAgent(BaseAgent):
    """Google Gemini-powered agent with function calling and memory"""
    
    def __init__(
        self,
        tools,
        config,
        conversation_memory=None,
        long_term_memory=None,
        vector_store=None
    ):
        super().__init__(
            tools, config,
            conversation_memory=conversation_memory,
            long_term_memory=long_term_memory,
            vector_store=vector_store
        )
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Convert tools to Gemini format
        tools_formatted = [tool.to_gemini_format() for tool in self.tools.values()]
        
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            tools=tools_formatted if tools_formatted else None,
            system_instruction=self.get_system_prompt()
        )
        self.chat_session = self.model.start_chat(history=[])
        
        # Set up summarizer function
        if self.conversation_memory:
            self.conversation_memory.summarizer_fn = self._summarize_text
    
    def _summarize_text(self, text: str) -> str:
        """Use Gemini to summarize text"""
        try:
            model = genai.GenerativeModel(model_name=self.config.GEMINI_MODEL)
            response = model.generate_content(
                f"Summarize the following conversation concisely, capturing key points and important information:\n\n{text}"
            )
            return response.text
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Gemini"""
        try:
            result = genai.embed_content(
                model=self.config.GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
    
    def chat(self, message: str) -> str:
        """Chat with Gemini agent"""
        print(f"\n{'='*60}")
        print(f"User: {message}")
        print(f"{'='*60}\n")
        
        # Add user message to memory
        self.add_message_to_memory("user", message)
        
        # Add memory context to message if available
        enhanced_message = message
        if self.long_term_memory:
            context = self.long_term_memory.get_relevant_context(message, max_tokens=300)
            if context:
                enhanced_message = f"{context}\n\nUser message: {message}"
        
        # Agentic loop
        iteration = 0
        response = self.chat_session.send_message(enhanced_message)
        
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
            
            # Add to memory
            self.add_message_to_memory("assistant", final_response)
            
            # Store learnings
            self._extract_and_store_learnings(message, final_response)
            
            print(f"{'='*60}")
            print(f"Assistant: {final_response}")
            print(f"{'='*60}\n")
            
            return final_response
        
        return "Max iterations reached"
    
    def _extract_and_store_learnings(self, user_message: str, assistant_response: str):
        """Extract important information from the conversation and store in memory"""
        if not self.long_term_memory:
            return
        
        # Check for user information patterns
        user_info_patterns = [
            "my name is", "i am", "i'm", "i like", "i prefer",
            "i work", "i live", "i'm from", "i have"
        ]
        
        message_lower = user_message.lower()
        for pattern in user_info_patterns:
            if pattern in message_lower:
                self.long_term_memory.remember(
                    content=user_message,
                    memory_type="user_info",
                    category="from_conversation",
                    importance=0.7
                )
                break