# ============================================================================
# FILE: agents/base_agent.py (Updated with Planning & Reasoning Support)
# ============================================================================

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from tools.base_tool import BaseTool
from memory.conversation_memory import ConversationMemory
from memory.vector_store import VectorStore
from memory.long_term_memory import LongTermMemory

# Planning imports
from planning.chain_of_thought import ChainOfThought, CoTStrategy, ReasoningChain
from planning.planner import Planner, Plan, PlanStatus
from planning.tree_of_thoughts import TreeOfThoughts, SearchStrategy


class BaseAgent(ABC):
    """Base class for AI agents with memory and reasoning support"""
    
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
        
        # Planning & Reasoning components (initialized later with LLM function)
        self.chain_of_thought: Optional[ChainOfThought] = None
        self.planner: Optional[Planner] = None
        self.tree_of_thoughts: Optional[TreeOfThoughts] = None
        
        # Reasoning state
        self.last_reasoning_chain: Optional[ReasoningChain] = None
        self.current_plan: Optional[Plan] = None
        self.reasoning_enabled = config.COT_ENABLED if hasattr(config, 'COT_ENABLED') else True
    
    def _initialize_reasoning_components(self, llm_fn: Callable[[str], str]):
        """Initialize reasoning components with LLM function"""
        
        # Initialize Chain of Thought
        if self.config.COT_ENABLED:
            strategy = CoTStrategy(self.config.COT_DEFAULT_STRATEGY)
            self.chain_of_thought = ChainOfThought(
                llm_fn=llm_fn,
                strategy=strategy,
                num_paths=self.config.COT_SELF_CONSISTENCY_PATHS,
                verbose=self.config.COT_VERBOSE
            )
            print("✅ Chain of Thought initialized")
        
        # Initialize Planner
        if self.config.PLANNING_ENABLED:
            self.planner = Planner(
                llm_fn=llm_fn,
                tool_executor=self.execute_tool,
                available_tools=self.tools,
                verbose=self.config.COT_VERBOSE
            )
            print("✅ Multi-step Planner initialized")
        
        # Initialize Tree of Thoughts
        if self.config.TOT_ENABLED:
            strategy = SearchStrategy(self.config.TOT_DEFAULT_STRATEGY)
            self.tree_of_thoughts = TreeOfThoughts(
                llm_fn=llm_fn,
                strategy=strategy,
                max_depth=self.config.TOT_MAX_DEPTH,
                branching_factor=self.config.TOT_BRANCHING_FACTOR,
                beam_width=self.config.TOT_BEAM_WIDTH,
                min_score_threshold=self.config.TOT_MIN_SCORE_THRESHOLD,
                verbose=self.config.TOT_VERBOSE
            )
            print("✅ Tree of Thoughts initialized")
    
    @abstractmethod
    def chat(self, message: str) -> str:
        """Send a message and get response"""
        pass
    
    @abstractmethod
    def _get_llm_response(self, prompt: str) -> str:
        """Get a simple LLM response (for reasoning components)"""
        pass
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        if tool_name in self.tools:
            return self.tools[tool_name].execute(**arguments)
        return {"error": f"Tool '{tool_name}' not found"}
    
    # ==========================================================================
    # Reasoning Methods
    # ==========================================================================
    
    def should_use_reasoning(self, message: str) -> bool:
        """Determine if advanced reasoning should be used for this message"""
        if not self.reasoning_enabled:
            return False
        
        mode = getattr(self.config, 'REASONING_MODE', 'auto')
        
        if mode == 'never':
            return False
        elif mode == 'always':
            return True
        else:  # auto
            return self._is_complex_query(message)
    
    def _is_complex_query(self, message: str) -> bool:
        """Detect if a query is complex enough to warrant reasoning"""
        message_lower = message.lower()
        
        # Check for complexity keywords
        keywords = getattr(self.config, 'COMPLEXITY_KEYWORDS', [])
        for keyword in keywords:
            if keyword in message_lower:
                return True
        
        # Check for question patterns that suggest complexity
        complex_patterns = [
            "how can i", "how do i", "how should i",
            "what is the best", "what are the",
            "why does", "why is", "why would",
            "compare", "difference between",
            "pros and cons", "advantages and disadvantages",
            "step by step", "steps to",
            "help me understand", "explain how",
            "what if", "suppose",
            "design", "create a plan", "strategy for"
        ]
        
        for pattern in complex_patterns:
            if pattern in message_lower:
                return True
        
        # Check message length (longer messages often more complex)
        if len(message.split()) > 30:
            return True
        
        # Check for multiple questions
        question_marks = message.count('?')
        if question_marks > 1:
            return True
        
        return False
    
    def think(self, query: str, context: str = "") -> ReasoningChain:
        """Apply Chain of Thought reasoning to a query"""
        if not self.chain_of_thought:
            raise RuntimeError("Chain of Thought not initialized")
        
        print("\n🧠 Applying Chain of Thought reasoning...")
        chain = self.chain_of_thought.reason(query, context)
        self.last_reasoning_chain = chain
        
        return chain
    
    def plan(self, goal: str, context: Dict[str, Any] = None) -> Plan:
        """Create a multi-step plan for a goal"""
        if not self.planner:
            raise RuntimeError("Planner not initialized")
        
        print("\n📋 Creating execution plan...")
        plan = self.planner.create_plan(goal, context)
        self.current_plan = plan
        
        return plan
    
    def execute_plan(self, plan: Plan = None) -> Dict[str, Any]:
        """Execute a plan"""
        if not self.planner:
            raise RuntimeError("Planner not initialized")
        
        plan = plan or self.current_plan
        if not plan:
            raise ValueError("No plan to execute")
        
        print("\n🚀 Executing plan...")
        return self.planner.execute_plan(plan)
    
    def explore(self, problem: str, context: str = "") -> Dict[str, Any]:
        """Use Tree of Thoughts to explore solution paths"""
        if not self.tree_of_thoughts:
            raise RuntimeError("Tree of Thoughts not initialized")
        
        print("\n🌳 Exploring solution paths with Tree of Thoughts...")
        return self.tree_of_thoughts.solve(problem, context)
    
    def get_enhanced_message(self, message: str) -> str:
        """Enhance a message with reasoning context if appropriate"""
        if not self.should_use_reasoning(message):
            return message
        
        # Add reasoning instruction
        enhanced = f"""Before answering, think through this step by step:
1. What is the core question or task?
2. What information do I have and need?
3. What are the logical steps to the answer?
4. What is my conclusion?

Question: {message}

Think step by step, then provide your answer:"""
        
        return enhanced
    
    def reason_and_respond(self, message: str) -> str:
        """Apply reasoning and generate response"""
        # First, apply Chain of Thought
        if self.chain_of_thought:
            chain = self.think(message)
            
            # If it's a planning task, create a plan
            if self._is_planning_task(message) and self.planner:
                plan = self.plan(message)
                
                # Auto-execute if configured
                if self.config.PLAN_AUTO_EXECUTE:
                    result = self.execute_plan(plan)
                    return result.get('final_output', chain.final_answer)
            
            # If it's a complex exploration task, use ToT
            if self._should_explore(message) and self.tree_of_thoughts:
                result = self.explore(message)
                return result.get('solution', chain.final_answer)
            
            return chain.final_answer
        
        # Fallback to enhanced message
        return self.get_enhanced_message(message)
    
    def _is_planning_task(self, message: str) -> bool:
        """Check if message is a planning task"""
        planning_indicators = [
            "create a plan", "make a plan", "plan for",
            "steps to", "how to achieve", "strategy for",
            "roadmap", "action plan", "todo list",
            "schedule", "organize", "project plan"
        ]
        
        message_lower = message.lower()
        return any(ind in message_lower for ind in planning_indicators)
    
    def _should_explore(self, message: str) -> bool:
        """Check if message warrants Tree of Thoughts exploration"""
        exploration_indicators = [
            "best approach", "best way", "optimal",
            "explore options", "alternatives",
            "different ways", "multiple solutions",
            "compare approaches", "trade-offs",
            "creative solution", "brainstorm"
        ]
        
        message_lower = message.lower()
        return any(ind in message_lower for ind in exploration_indicators)
    
    # ==========================================================================
    # Memory Methods
    # ==========================================================================
    
    def reset_conversation(self):
        """Reset conversation history"""
        if self.conversation_memory:
            self.conversation_memory.clear()
        self.conversation_history = []
        self.last_reasoning_chain = None
        self.current_plan = None
    
    def get_system_prompt(self) -> str:
        """Get system prompt with memory and reasoning context"""
        base_prompt = self.config.SYSTEM_PROMPT
        
        # Add memory context if available
        if self.long_term_memory and self.conversation_memory:
            recent_topics = []
            for msg in self.conversation_memory.messages[-5:]:
                if msg.role == "user":
                    recent_topics.append(msg.content[:100])
            
            if recent_topics:
                query = " ".join(recent_topics)
                memory_context = self.long_term_memory.get_relevant_context(query)
                if memory_context:
                    base_prompt += f"\n\n{memory_context}"
        
        # Add reasoning context if there's an active reasoning chain
        if self.last_reasoning_chain:
            reasoning_summary = f"\n\nRecent reasoning: {self.last_reasoning_chain.final_answer[:200]}..."
            base_prompt += reasoning_summary
        
        # Add plan context if there's an active plan
        if self.current_plan and self.current_plan.status == PlanStatus.EXECUTING:
            progress = self.current_plan.get_progress()
            plan_context = f"\n\nActive plan: {self.current_plan.goal} ({progress['progress_pct']:.0f}% complete)"
            base_prompt += plan_context
        
        return base_prompt
    
    def add_message_to_memory(self, role: str, content: str):
        """Add message to conversation memory"""
        if self.conversation_memory:
            self.conversation_memory.add_message(role, content)
        
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
            pass  # Handled automatically
    
    # ==========================================================================
    # Stats and Diagnostics
    # ==========================================================================
    
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
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        stats = {
            "reasoning_enabled": self.reasoning_enabled,
            "cot_available": self.chain_of_thought is not None,
            "planner_available": self.planner is not None,
            "tot_available": self.tree_of_thoughts is not None
        }
        
        if self.chain_of_thought:
            stats["cot_history_count"] = len(self.chain_of_thought.reasoning_history)
        
        if self.planner:
            stats["plans_count"] = len(self.planner.plans)
            if self.current_plan:
                stats["current_plan"] = {
                    "goal": self.current_plan.goal,
                    "status": self.current_plan.status.value,
                    "progress": self.current_plan.get_progress()
                }
        
        if self.tree_of_thoughts:
            stats["trees_count"] = len(self.tree_of_thoughts.trees)
        
        return stats
    
    def get_full_stats(self) -> Dict[str, Any]:
        """Get all statistics"""
        return {
            "memory": self.get_memory_stats(),
            "reasoning": self.get_reasoning_stats(),
            "tools": list(self.tools.keys())
        }