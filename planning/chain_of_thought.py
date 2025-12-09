# ============================================================================
# FILE: planning/chain_of_thought.py
# ============================================================================
"""
Chain of Thought (CoT) Implementation

Enables step-by-step reasoning before providing answers.
Supports multiple CoT strategies:
- Zero-shot CoT: "Let's think step by step"
- Few-shot CoT: Examples of reasoning chains
- Self-consistency: Multiple reasoning paths with voting
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re


class CoTStrategy(Enum):
    """Chain of Thought strategies"""
    ZERO_SHOT = "zero_shot"           # Simple "think step by step"
    FEW_SHOT = "few_shot"             # With examples
    SELF_CONSISTENCY = "self_consistency"  # Multiple paths + voting
    STRUCTURED = "structured"          # Structured reasoning format


@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    step_number: int
    thought: str
    conclusion: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "conclusion": self.conclusion,
            "confidence": self.confidence
        }


@dataclass
class ReasoningChain:
    """Complete chain of reasoning"""
    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    total_confidence: float = 1.0
    strategy_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "confidence": self.total_confidence,
            "strategy": self.strategy_used,
            "timestamp": self.timestamp
        }
    
    def format_for_display(self) -> str:
        """Format reasoning chain for display"""
        lines = ["🧠 Reasoning Chain:", "-" * 40]
        
        for step in self.steps:
            lines.append(f"  Step {step.step_number}: {step.thought}")
            if step.conclusion:
                lines.append(f"    → {step.conclusion}")
        
        lines.append("-" * 40)
        lines.append(f"📍 Final Answer: {self.final_answer}")
        lines.append(f"📊 Confidence: {self.total_confidence:.2%}")
        
        return "\n".join(lines)


class ChainOfThought:
    """
    Chain of Thought reasoning engine.
    
    Enhances LLM responses with explicit step-by-step reasoning.
    """
    
    # CoT prompts for different strategies
    ZERO_SHOT_PROMPT = """Think through this step-by-step:
1. First, understand what is being asked
2. Break down the problem into smaller parts
3. Solve each part systematically
4. Combine the results for the final answer

Question: {query}

Let's think step by step:"""

    STRUCTURED_PROMPT = """Analyze this problem using structured reasoning:

**Problem:** {query}

**Step 1 - Understanding:**
What is the core question or task?

**Step 2 - Information Gathering:**
What information do I have? What do I need?

**Step 3 - Analysis:**
How do the pieces connect? What patterns do I see?

**Step 4 - Solution:**
Based on my analysis, the answer is...

**Step 5 - Verification:**
Let me verify this makes sense...

Now provide your structured analysis:"""

    FEW_SHOT_EXAMPLES = [
        {
            "query": "If a train travels 120 miles in 2 hours, how long to travel 300 miles?",
            "reasoning": [
                "First, I need to find the speed of the train.",
                "Speed = Distance / Time = 120 miles / 2 hours = 60 mph",
                "Now I need to find time for 300 miles at 60 mph",
                "Time = Distance / Speed = 300 miles / 60 mph = 5 hours"
            ],
            "answer": "5 hours"
        },
        {
            "query": "What day comes two days after the day before yesterday if today is Wednesday?",
            "reasoning": [
                "Today is Wednesday",
                "The day before yesterday was Monday",
                "Two days after Monday is Wednesday",
                "So the answer is Wednesday"
            ],
            "answer": "Wednesday"
        }
    ]

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        strategy: CoTStrategy = CoTStrategy.STRUCTURED,
        num_paths: int = 3,  # For self-consistency
        verbose: bool = True
    ):
        """
        Initialize Chain of Thought engine.
        
        Args:
            llm_fn: Function to call LLM (takes prompt, returns response)
            strategy: CoT strategy to use
            num_paths: Number of reasoning paths for self-consistency
            verbose: Whether to print reasoning steps
        """
        self.llm_fn = llm_fn
        self.strategy = strategy
        self.num_paths = num_paths
        self.verbose = verbose
        self.reasoning_history: List[ReasoningChain] = []
    
    def reason(self, query: str, context: str = "") -> ReasoningChain:
        """
        Apply Chain of Thought reasoning to a query.
        
        Args:
            query: The question or task
            context: Optional additional context
            
        Returns:
            ReasoningChain with steps and final answer
        """
        if self.verbose:
            print(f"\n🧠 Applying {self.strategy.value} Chain of Thought...")
        
        if self.strategy == CoTStrategy.ZERO_SHOT:
            chain = self._zero_shot_cot(query, context)
        elif self.strategy == CoTStrategy.FEW_SHOT:
            chain = self._few_shot_cot(query, context)
        elif self.strategy == CoTStrategy.SELF_CONSISTENCY:
            chain = self._self_consistency_cot(query, context)
        else:  # STRUCTURED
            chain = self._structured_cot(query, context)
        
        self.reasoning_history.append(chain)
        
        if self.verbose:
            print(chain.format_for_display())
        
        return chain
    
    def _zero_shot_cot(self, query: str, context: str) -> ReasoningChain:
        """Zero-shot Chain of Thought"""
        prompt = self.ZERO_SHOT_PROMPT.format(query=query)
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        response = self.llm_fn(prompt)
        
        # Parse the response into steps
        steps = self._parse_reasoning_steps(response)
        final_answer = self._extract_final_answer(response)
        
        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=final_answer,
            strategy_used="zero_shot"
        )
    
    def _few_shot_cot(self, query: str, context: str) -> ReasoningChain:
        """Few-shot Chain of Thought with examples"""
        # Build prompt with examples
        examples_text = "\n\n".join([
            f"Example:\nQuestion: {ex['query']}\n" +
            "Reasoning:\n" + "\n".join([f"- {r}" for r in ex['reasoning']]) +
            f"\nAnswer: {ex['answer']}"
            for ex in self.FEW_SHOT_EXAMPLES
        ])
        
        prompt = f"""Here are some examples of step-by-step reasoning:

{examples_text}

Now solve this problem with the same approach:

Question: {query}
{f'Context: {context}' if context else ''}

Reasoning:"""
        
        response = self.llm_fn(prompt)
        
        steps = self._parse_reasoning_steps(response)
        final_answer = self._extract_final_answer(response)
        
        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=final_answer,
            strategy_used="few_shot"
        )
    
    def _structured_cot(self, query: str, context: str) -> ReasoningChain:
        """Structured Chain of Thought with explicit sections"""
        prompt = self.STRUCTURED_PROMPT.format(query=query)
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        response = self.llm_fn(prompt)
        
        # Parse structured response
        steps = self._parse_structured_response(response)
        final_answer = self._extract_final_answer(response)
        
        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=final_answer,
            strategy_used="structured"
        )
    
    def _self_consistency_cot(self, query: str, context: str) -> ReasoningChain:
        """Self-consistency: Generate multiple paths and vote"""
        all_chains = []
        all_answers = []
        
        if self.verbose:
            print(f"  Generating {self.num_paths} reasoning paths...")
        
        for i in range(self.num_paths):
            prompt = f"""Reasoning path {i+1}/{self.num_paths}:

Question: {query}
{f'Context: {context}' if context else ''}

Think through this problem step by step, then provide your final answer.
Be creative and consider different angles.

Reasoning:"""
            
            response = self.llm_fn(prompt)
            steps = self._parse_reasoning_steps(response)
            answer = self._extract_final_answer(response)
            
            all_chains.append(steps)
            all_answers.append(answer)
            
            if self.verbose:
                print(f"    Path {i+1}: {answer[:50]}...")
        
        # Vote on the final answer
        final_answer, confidence = self._vote_on_answers(all_answers)
        
        # Combine all reasoning steps
        combined_steps = []
        for i, chain in enumerate(all_chains):
            combined_steps.append(ReasoningStep(
                step_number=i+1,
                thought=f"Path {i+1}: " + (chain[0].thought if chain else "No reasoning"),
                conclusion=all_answers[i]
            ))
        
        return ReasoningChain(
            query=query,
            steps=combined_steps,
            final_answer=final_answer,
            total_confidence=confidence,
            strategy_used="self_consistency"
        )
    
    def _parse_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Parse LLM response into reasoning steps"""
        steps = []
        
        # Try to find numbered steps
        step_patterns = [
            r'(?:Step\s*)?(\d+)[.:]\s*(.+?)(?=(?:Step\s*)?\d+[.:]|$)',
            r'[-•]\s*(.+?)(?=[-•]|$)',
            r'(\d+)\)\s*(.+?)(?=\d+\)|$)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for i, match in enumerate(matches):
                    if isinstance(match, tuple):
                        thought = match[-1].strip()
                    else:
                        thought = match.strip()
                    
                    if thought and len(thought) > 10:
                        steps.append(ReasoningStep(
                            step_number=i + 1,
                            thought=thought[:500]
                        ))
                break
        
        # Fallback: split by sentences
        if not steps:
            sentences = response.split('.')
            for i, sent in enumerate(sentences[:5]):
                if sent.strip() and len(sent.strip()) > 20:
                    steps.append(ReasoningStep(
                        step_number=i + 1,
                        thought=sent.strip()
                    ))
        
        return steps
    
    def _parse_structured_response(self, response: str) -> List[ReasoningStep]:
        """Parse structured response with sections"""
        steps = []
        
        # Look for section headers
        sections = [
            "Understanding", "Information", "Analysis", 
            "Solution", "Verification", "Conclusion"
        ]
        
        for i, section in enumerate(sections):
            pattern = rf'\*?\*?{section}[:\*]*\s*(.+?)(?=\*?\*?(?:{"|".join(sections)})|$)'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                if content:
                    steps.append(ReasoningStep(
                        step_number=i + 1,
                        thought=f"{section}: {content[:300]}"
                    ))
        
        # Fallback to regular parsing
        if not steps:
            steps = self._parse_reasoning_steps(response)
        
        return steps
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from response"""
        # Look for explicit answer markers
        patterns = [
            r'(?:Final\s+)?[Aa]nswer[:\s]+(.+?)(?:\n|$)',
            r'[Cc]onclusion[:\s]+(.+?)(?:\n|$)',
            r'[Tt]herefore[,:\s]+(.+?)(?:\n|$)',
            r'[Ss]o[,:\s]+the\s+answer\s+is[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        
        # Fallback: return last meaningful sentence
        sentences = response.split('.')
        for sent in reversed(sentences):
            if sent.strip() and len(sent.strip()) > 20:
                return sent.strip()
        
        return response[-500:] if len(response) > 500 else response
    
    def _vote_on_answers(self, answers: List[str]) -> tuple:
        """Vote on multiple answers to find consensus"""
        if not answers:
            return "", 0.0
        
        # Simple voting: find most common answer (normalized)
        normalized = [a.lower().strip() for a in answers]
        
        # Count occurrences
        counts = {}
        for ans in normalized:
            # Use first 100 chars for comparison
            key = ans[:100]
            counts[key] = counts.get(key, 0) + 1
        
        # Find winner
        winner_key = max(counts, key=counts.get)
        winner_count = counts[winner_key]
        confidence = winner_count / len(answers)
        
        # Return original (non-normalized) version
        for ans in answers:
            if ans.lower().strip()[:100] == winner_key:
                return ans, confidence
        
        return answers[0], confidence
    
    def get_enhanced_prompt(self, query: str, context: str = "") -> str:
        """
        Get a prompt enhanced with CoT instructions.
        Useful for integrating with existing chat flow.
        """
        cot_instruction = """
Before answering, think through this step by step:
1. What is the core question or task?
2. What information and context do I have?
3. What logical steps lead to the answer?
4. What is my conclusion?

Show your reasoning, then provide the answer.
"""
        
        enhanced = f"{cot_instruction}\n\n"
        if context:
            enhanced += f"Context: {context}\n\n"
        enhanced += f"Question: {query}"
        
        return enhanced
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get reasoning history"""
        return [chain.to_dict() for chain in self.reasoning_history]
    
    def clear_history(self):
        """Clear reasoning history"""
        self.reasoning_history = []