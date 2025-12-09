# ============================================================================
# FILE: agents/base_agent.py (Updated with Memory Support)
# ============================================================================

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tools.base_tool import BaseTool
from memory.conversation_memory import ConversationMemory
from memory.vector_store import VectorStore
from memory.long_term_memory import LongTermMemory


class BaseAgent(ABC):
    """Base class for AI agents with memory support"""
    
    def __init__(
        self,
        tools: List[BaseTool],
        config: Any,
        conversation_memory: Optional[ConversationMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.config = config
        
        # Memory components
        self.conversation_memory = conversation_memory
        self.long_term_memory = long_term_memory
        self.vector_store = vector_store
        
        # Legacy conversation history (for compatibility)
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
        if self.conversation_memory:
            self.conversation_memory.clear()
        self.conversation_history = []
    
    def get_system_prompt(self) -> str:
        """Get system prompt with memory context"""
        base_prompt = self.config.SYSTEM_PROMPT
        
        # Add memory context if available
        if self.long_term_memory and self.conversation_memory:
            # Get recent conversation topics for context
            recent_topics = []
            for msg in self.conversation_memory.messages[-5:]:
                if msg.role == "user":
                    recent_topics.append(msg.content[:100])
            
            if recent_topics:
                query = " ".join(recent_topics)
                memory_context = self.long_term_memory.get_relevant_context(query)
                if memory_context:
                    base_prompt += f"\n\n{memory_context}"
        
        return base_prompt
    
    def add_message_to_memory(self, role: str, content: str):
        """Add message to conversation memory"""
        if self.conversation_memory:
            self.conversation_memory.add_message(role, content)
        
        # Also add to legacy history for compatibility
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get messages for LLM context"""
        if self.conversation_memory:
            return self.conversation_memory.get_context_messages()
        return self.conversation_history
    
    def summarize_if_needed(self):
        """Trigger summarization if needed"""
        if self.conversation_memory:
            # The conversation memory handles this automatically
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {}
        
        if self.conversation_memory:
            stats["conversation"] = self.conversation_memory.get_stats()
        
        if self.long_term_memory:
            stats["long_term"] = self.long_term_memory.get_stats()
        
        if self.vector_store:
            stats["vector_store"] = self.vector_store.get_stats()
        
        return stats