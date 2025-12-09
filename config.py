import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Configuration for the assistant"""

    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Model settings
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GEMINI_MODEL: str = "gemini-2.5-pro"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"
    
    # ==========================================================================
    # Agent settings
    # ==========================================================================
    DEFAULT_AGENT: str = "openai"
    MAX_ITERATIONS: int = 10
    TEMPERATURE: float = 0.7
    
    # ==========================================================================
    # Memory settings
    # ==========================================================================
    MEMORY_ENABLED: bool = True
    CONVERSATION_WINDOW_SIZE: int = 20
    SUMMARY_MAX_TOKENS: int = 500
    MAX_CONTEXT_MESSAGES: int = 10
    
    # ==========================================================================
    # Vector store settings
    # ==========================================================================
    VECTOR_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "assistant_memory"
    EMBEDDING_DIMENSION: int = 1536
    TOP_K_RESULTS: int = 5
    
    # ==========================================================================
    # Long-term memory settings
    # ==========================================================================
    MEMORY_FILE_PATH: str = "./long_term_memory.json"
    USER_PROFILE_PATH: str = "./user_profile.json"
    
    # ==========================================================================
    # Planning & Reasoning settings (NEW)
    # ==========================================================================
    
    # Chain of Thought settings
    COT_ENABLED: bool = True
    COT_DEFAULT_STRATEGY: str = "structured"  # zero_shot, few_shot, structured, self_consistency
    COT_SELF_CONSISTENCY_PATHS: int = 3  # Number of paths for self-consistency
    COT_VERBOSE: bool = True
    
    # Planning settings
    PLANNING_ENABLED: bool = True
    PLAN_MAX_STEPS: int = 10
    PLAN_AUTO_EXECUTE: bool = False  # Auto-execute plans after creation
    PLAN_RETRY_FAILED_STEPS: int = 2
    
    # Tree of Thoughts settings
    TOT_ENABLED: bool = True
    TOT_DEFAULT_STRATEGY: str = "beam"  # bfs, dfs, beam, best_first
    TOT_MAX_DEPTH: int = 4
    TOT_BRANCHING_FACTOR: int = 3
    TOT_BEAM_WIDTH: int = 3
    TOT_MIN_SCORE_THRESHOLD: float = 3.0
    TOT_VERBOSE: bool = True
    
    # Reasoning mode settings
    REASONING_MODE: str = "auto"  # auto, always, never
    # auto: Use reasoning for complex queries
    # always: Always apply reasoning
    # never: Disable advanced reasoning
    
    # Complexity detection for auto mode
    COMPLEXITY_KEYWORDS: List[str] = None
    
    def __post_init__(self):
        if self.COMPLEXITY_KEYWORDS is None:
            self.COMPLEXITY_KEYWORDS = [
                "explain", "analyze", "compare", "evaluate", "design",
                "plan", "strategy", "complex", "difficult", "multi-step",
                "how to", "why", "what if", "pros and cons", "trade-offs",
                "best approach", "optimize", "solve", "debug", "investigate"
            ]
    
    # ==========================================================================
    # System prompts
    # ==========================================================================
    SYSTEM_PROMPT: str = """You are a helpful personal assistant with advanced reasoning capabilities.

You have access to:
1. **Memory**: Remember information from past conversations
2. **Tools**: Execute various tools for tasks like web search, weather, file operations
3. **Reasoning**: Apply step-by-step thinking for complex problems

For complex problems, you can:
- Think step-by-step using Chain of Thought reasoning
- Create multi-step plans for complex tasks
- Explore multiple solution paths using Tree of Thoughts

When to use advanced reasoning:
- Complex problems requiring multiple steps
- Questions with no obvious answer
- Tasks requiring planning or analysis
- Problems where multiple approaches exist

Be clear about your reasoning process when solving complex problems."""

    PLANNING_SYSTEM_PROMPT: str = """You are a strategic planning assistant.

When creating plans:
1. Break down goals into clear, actionable steps
2. Identify dependencies between steps
3. Consider potential obstacles and alternatives
4. Estimate effort for each step

When executing plans:
1. Execute steps in dependency order
2. Adapt to failures by replanning if needed
3. Track progress and provide updates
4. Synthesize results into a coherent output"""

    REASONING_SYSTEM_PROMPT: str = """You are a reasoning assistant that thinks carefully before answering.

Approach problems by:
1. Understanding what is being asked
2. Breaking down complex problems into parts
3. Considering multiple approaches
4. Evaluating each approach
5. Synthesizing the best solution

Show your reasoning clearly and explain your thought process."""