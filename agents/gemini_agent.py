# ============================================================================
# FILE: agents/gemini_agent.py (Updated with Planning & Reasoning Support)
# ============================================================================

from agents.base_agent import BaseAgent
from typing import List, Dict, Any, Optional
import json

from planning.chain_of_thought import ChainOfThought, CoTStrategy
from planning.planner import Planner
from planning.tree_of_thoughts import TreeOfThoughts, SearchStrategy

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiAgent(BaseAgent):
    """Google Gemini-powered agent with function calling, memory, and reasoning"""
    
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
        
        # Store model name for later use
        self.model_name = config.GEMINI_MODEL
        
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
        
        # Initialize reasoning components with LLM function
        self._initialize_reasoning_components(self._get_llm_response)
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get a simple LLM response (for reasoning components)"""
        try:
            # Use a separate model instance for reasoning to avoid chat history issues
            reasoning_model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction="You are a helpful reasoning assistant. Think carefully and provide clear, structured responses."
            )
            response = reasoning_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting response: {str(e)}"
    
    def _summarize_text(self, text: str) -> str:
        """Use Gemini to summarize text"""
        try:
            model = genai.GenerativeModel(model_name=self.model_name)
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
    
    def _refresh_chat_session(self):
        """Refresh the chat session with updated system prompt"""
        tools_formatted = [tool.to_gemini_format() for tool in self.tools.values()]
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=tools_formatted if tools_formatted else None,
            system_instruction=self.get_system_prompt()
        )
        self.chat_session = self.model.start_chat(history=[])
    
    def chat(self, message: str) -> str:
        """Chat with Gemini agent with optional reasoning"""
        print(f"\n{'='*60}")
        print(f"User: {message}")
        print(f"{'='*60}\n")
        
        # Add user message to memory
        self.add_message_to_memory("user", message)
        
        # Check if we should use advanced reasoning
        use_reasoning = self.should_use_reasoning(message)
        
        if use_reasoning:
            print("🧠 Complex query detected - applying advanced reasoning...")
            return self._chat_with_reasoning(message)
        else:
            return self._chat_standard(message)
    
    def _chat_with_reasoning(self, message: str) -> str:
        """Chat with advanced reasoning applied"""
        
        # Determine reasoning approach
        if self._is_planning_task(message):
            return self._chat_with_planning(message)
        elif self._should_explore(message):
            return self._chat_with_exploration(message)
        else:
            return self._chat_with_cot(message)
    
    def _chat_with_cot(self, message: str) -> str:
        """Chat with Chain of Thought reasoning"""
        print("📝 Using Chain of Thought reasoning...")
        
        # Get memory context
        memory_context = ""
        if self.long_term_memory:
            memory_context = self.long_term_memory.get_relevant_context(message, max_tokens=300)
        
        # Apply Chain of Thought
        if self.chain_of_thought:
            reasoning_chain = self.chain_of_thought.reason(message, memory_context)
            
            # Generate final response with tools if needed
            enhanced_prompt = f"""Based on the following reasoning, provide a helpful response.

Original Question: {message}

Reasoning Process:
{chr(10).join([f"Step {s.step_number}: {s.thought}" for s in reasoning_chain.steps])}

Preliminary Conclusion: {reasoning_chain.final_answer}

Now provide the final answer. If you need additional information, you can use the available tools. Otherwise, provide the response based on the reasoning above."""
            
            return self._execute_with_tools(enhanced_prompt, message)
        else:
            return self._chat_standard(message)
    
    def _chat_with_planning(self, message: str) -> str:
        """Chat with multi-step planning"""
        print("📋 Creating and executing plan...")
        
        if not self.planner:
            return self._chat_with_cot(message)
        
        # Get context
        context = {}
        if self.long_term_memory:
            user_info = self.long_term_memory.get_user_info()
            if user_info:
                context["user_info"] = user_info
        
        # Create plan
        plan = self.planner.create_plan(message, context)
        self.current_plan = plan
        
        print(f"\n📋 Plan created with {len(plan.steps)} steps:")
        for step in plan.steps:
            print(f"   • {step.description}")
        
        # Execute plan
        result = self.planner.execute_plan(plan)
        
        final_output = result.get('final_output', '')
        
        # Add to memory
        self.add_message_to_memory("assistant", final_output)
        
        # Store learnings
        self._extract_and_store_learnings(message, final_output)
        
        print(f"\n{'='*60}")
        print(f"Assistant: {final_output}")
        print(f"{'='*60}\n")
        
        return final_output
    
    def _chat_with_exploration(self, message: str) -> str:
        """Chat with Tree of Thoughts exploration"""
        print("🌳 Exploring solution paths...")
        
        if not self.tree_of_thoughts:
            return self._chat_with_cot(message)
        
        # Get context
        context = ""
        if self.long_term_memory:
            context = self.long_term_memory.get_relevant_context(message, max_tokens=300)
        
        # Explore with ToT
        result = self.tree_of_thoughts.solve(message, context)
        
        solution = result.get('solution', '')
        
        # Enhance with tools if needed
        enhanced_prompt = f"""Based on the following exploration of solution paths, provide a final helpful response.

Question: {message}

Best Solution Found:
{solution}

Reasoning Path:
{chr(10).join(result.get('reasoning_path', [])[:5])}

Provide the final answer, using any tools if additional information is needed:"""
        
        final_response = self._execute_with_tools(enhanced_prompt, message)
        
        return final_response
    
    def _chat_standard(self, message: str) -> str:
        """Standard chat without advanced reasoning"""
        return self._execute_with_tools(message, message)
    
    def _execute_with_tools(self, prompt: str, original_message: str) -> str:
        """Execute prompt with tool calling support"""
        
        # Add memory context to message if available
        enhanced_message = prompt
        if self.long_term_memory:
            context = self.long_term_memory.get_relevant_context(original_message, max_tokens=300)
            if context and context not in prompt:
                enhanced_message = f"{context}\n\nUser request: {prompt}"
        
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
                    
                    print(f"📤 Result: {str(result)[:200]}...\n")
                    
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
            self._extract_and_store_learnings(original_message, final_response)
            
            print(f"\n{'='*60}")
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
            "i work", "i live", "i'm from", "i have", "my favorite",
            "i always", "i never", "i usually", "i need", "i want"
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
    
    # ==========================================================================
    # Reasoning Command Methods
    # ==========================================================================
    
    def think_about(self, query: str, strategy: str = None) -> str:
        """Explicitly apply Chain of Thought reasoning"""
        if strategy and self.chain_of_thought:
            self.chain_of_thought.strategy = CoTStrategy(strategy)
        
        chain = self.think(query)
        
        return chain.format_for_display()
    
    def create_plan_for(self, goal: str) -> str:
        """Explicitly create a plan for a goal"""
        plan = self.plan(goal)
        
        return plan.format_for_display()
    
    def explore_solutions(self, problem: str, strategy: str = None) -> str:
        """Explicitly explore solutions using Tree of Thoughts"""
        if strategy and self.tree_of_thoughts:
            self.tree_of_thoughts.strategy = SearchStrategy(strategy)
        
        result = self.explore(problem)
        
        output = [
            "🌳 Tree of Thoughts Exploration",
            "=" * 50,
            f"Problem: {problem}",
            "",
            "Best Solution:",
            result.get('solution', 'No solution found'),
            "",
            "Reasoning Path:",
        ]
        
        for i, step in enumerate(result.get('reasoning_path', [])[:5], 1):
            output.append(f"  {i}. {step}")
        
        output.append("")
        output.append(f"Confidence Score: {result.get('score', 0):.2f}")
        output.append(f"Nodes Explored: {result.get('stats', {}).get('nodes_created', 0)}")
        
        return "\n".join(output)
    
    def set_reasoning_mode(self, mode: str):
        """Set the reasoning mode (auto/always/never)"""
        if mode in ['auto', 'always', 'never']:
            self.config.REASONING_MODE = mode
            self.reasoning_enabled = mode != 'never'
            print(f"✅ Reasoning mode set to: {mode}")
        else:
            print(f"❌ Invalid mode. Use: auto, always, or never")
    
    def reset_conversation(self):
        """Reset conversation and refresh chat session"""
        super().reset_conversation()
        self._refresh_chat_session()