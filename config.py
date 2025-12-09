import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for the assistant"""

    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Model settings
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    GEMINI_MODEL: str = "gemini-2.5-pro"
    
    # Agent settings
    DEFAULT_AGENT: str = "openai"  # or "gemini"
    MAX_ITERATIONS: int = 10
    TEMPERATURE: float = 0.7
    
    # Memory settings
    MEMORY_ENABLED: bool = True
    CONVERSATION_WINDOW_SIZE: int = 20  # Messages before summarization
    SUMMARY_MAX_TOKENS: int = 500
    MAX_CONTEXT_MESSAGES: int = 10  # Messages to keep in active context
    
    # Vector store settings
    VECTOR_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "assistant_memory"
    EMBEDDING_DIMENSION: int = 1536  # OpenAI embeddings
    TOP_K_RESULTS: int = 5
    
    # Long-term memory settings
    MEMORY_FILE_PATH: str = "./long_term_memory.json"
    USER_PROFILE_PATH: str = "./user_profile.json"
    
    # System prompts
    SYSTEM_PROMPT: str = """You are a helpful personal assistant with memory capabilities.
You can remember information from past conversations and use it to provide better assistance.
When you learn something important about the user or a topic, use the memory tools to store it.
When answering questions, check your memory for relevant information first.

You have access to:
- Conversation history and summaries
- Long-term memory for facts and preferences
- Semantic search for finding relevant past information

Be proactive about remembering important information and using past context."""
