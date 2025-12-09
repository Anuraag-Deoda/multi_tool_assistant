# ============================================================================
# FILE: agents/openai_agent.py (Updated with Memory Support)
# ============================================================================

from agents.base_agent import BaseAgent
from openai import OpenAI
from typing import List, Optional, Any
import json


class OpenAIAgent(BaseAgent):
    """OpenAI-powered agent with function calling and memory"""
    
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
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        
        # Set up summarizer function for conversation memory
        if self.conversation_memory:
            self.conversation_memory.summarizer_fn = self._summarize_text
    
    def _summarize_text(self, text: str) -> str:
        """Use OpenAI to summarize text"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the following conversation concisely, capturing key points and important information."},
                    {"role": "user", "content": text}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.config.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
    
    def chat(self, message: str) -> str:
        """Chat with OpenAI agent"""
        print(f"\n{'='*60}")
        print(f"User: {message}")
        print(f"{'='*60}\n")
        
        # Add user message to memory
        self.add_message_to_memory("user", message)
        
        # Get system prompt with memory context
        system_prompt = self.get_system_prompt()
        
        # Build messages for API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation context
        context_messages = self.get_context_messages()
        messages.extend(context_messages)
        
        # Prepare tools for OpenAI
        tools_formatted = [tool.to_openai_format() for tool in self.tools.values()]
        
        # Agentic loop
        iteration = 0
        while iteration < self.config.MAX_ITERATIONS:
            iteration += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_formatted if tools_formatted else None,
                tool_choice="auto" if tools_formatted else None,
                temperature=self.config.TEMPERATURE
            )
            
            assistant_message = response.choices[0].message
            
            # Check if function calling is needed
            if assistant_message.tool_calls:
                # Add assistant message with tool calls
                messages.append({
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Continue loop to get final response
                continue
            
            else:
                # No more tool calls, return final answer
                final_response = assistant_message.content
                
                # Add to memory
                self.add_message_to_memory("assistant", final_response)
                
                # Store any learned information in long-term memory
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
                # Store as user info
                self.long_term_memory.remember(
                    content=user_message,
                    memory_type="user_info",
                    category="from_conversation",
                    importance=0.7
                )
                break