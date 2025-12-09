# ============================================================================
# FILE: tools/planning_tool.py
# ============================================================================

from tools.base_tool import BaseTool
from typing import Dict, Any, Optional
from planning.chain_of_thought import ChainOfThought, CoTStrategy
from planning.planner import Planner, Plan
from planning.tree_of_thoughts import TreeOfThoughts, SearchStrategy


class ReasoningTool(BaseTool):
    """Tool for advanced reasoning operations"""
    
    def __init__(
        self,
        cot_engine: ChainOfThought = None,
        planner: Planner = None,
        tot_engine: TreeOfThoughts = None
    ):
        self.cot = cot_engine
        self.planner = planner
        self.tot = tot_engine
    
    @property
    def name(self) -> str:
        return "reasoning"
    
    @property
    def description(self) -> str:
        return """Advanced reasoning and planning tool. Use this for:
        
        - 'think': Apply Chain of Thought reasoning to think through a problem step-by-step
        - 'plan': Create a multi-step plan for complex tasks
        - 'explore': Use Tree of Thoughts to explore multiple solution paths
        - 'evaluate': Evaluate an idea or solution
        
        This tool helps with complex problems that require careful reasoning."""
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["think", "plan", "explore", "evaluate"],
                    "description": "The reasoning operation to perform"
                },
                "query": {
                    "type": "string",
                    "description": "The problem or question to reason about"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the reasoning"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["zero_shot", "few_shot", "structured", "self_consistency", 
                            "bfs", "dfs", "beam", "best_first"],
                    "description": "Strategy for reasoning (optional)"
                },
                "depth": {
                    "type": "integer",
                    "description": "Max depth for tree exploration (default 3)"
                }
            },
            "required": ["operation", "query"],
            "additionalProperties": False
        }
    
    def execute(
        self,
        operation: str,
        query: str,
        context: str = "",
        strategy: str = None,
        depth: int = 3
    ) -> Dict[str, Any]:
        """Execute reasoning operation"""
        
        print(f"🧠 Reasoning operation: {operation}")
        
        if operation == "think":
            if not self.cot:
                return {"success": False, "error": "Chain of Thought not configured"}
            
            # Set strategy if provided
            if strategy and strategy in ["zero_shot", "few_shot", "structured", "self_consistency"]:
                self.cot.strategy = CoTStrategy(strategy)
            
            result = self.cot.reason(query, context)
            
            return {
                "success": True,
                "operation": "chain_of_thought",
                "reasoning_steps": [s.to_dict() for s in result.steps],
                "conclusion": result.final_answer,
                "confidence": result.total_confidence
            }
        
        elif operation == "plan":
            if not self.planner:
                return {"success": False, "error": "Planner not configured"}
            
            plan = self.planner.create_plan(
                goal=query,
                context={"user_context": context} if context else None
            )
            
            return {
                "success": True,
                "operation": "planning",
                "plan_id": plan.plan_id,
                "steps": [s.to_dict() for s in plan.steps],
                "status": plan.status.value
            }
        
        elif operation == "explore":
            if not self.tot:
                return {"success": False, "error": "Tree of Thoughts not configured"}
            
            # Set strategy if provided
            if strategy and strategy in ["bfs", "dfs", "beam", "best_first"]:
                self.tot.strategy = SearchStrategy(strategy)
            
            self.tot.max_depth = depth
            
            result = self.tot.solve(query, context)
            
            return {
                "success": True,
                "operation": "tree_of_thoughts",
                "solution": result["solution"],
                "reasoning_path": result["reasoning_path"],
                "score": result["score"],
                "exploration_stats": result["stats"]
            }
        
        elif operation == "evaluate":
            if not self.cot:
                return {"success": False, "error": "Reasoning engine not configured"}
            
            # Use CoT for evaluation
            eval_query = f"Evaluate this idea/solution and provide a score from 1-10 with reasoning: {query}"
            result = self.cot.reason(eval_query, context)
            
            return {
                "success": True,
                "operation": "evaluation",
                "evaluation": result.final_answer,
                "reasoning": [s.thought for s in result.steps]
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }


class PlanExecutorTool(BaseTool):
    """Tool for executing and managing plans"""
    
    def __init__(self, planner: Planner):
        self.planner = planner
    
    @property
    def name(self) -> str:
        return "plan_executor"
    
    @property
    def description(self) -> str:
        return """Execute and manage multi-step plans.
        
        Operations:
        - 'execute': Execute a plan by ID or the current plan
        - 'status': Get status of a plan
        - 'list': List all plans
        - 'pause': Pause plan execution
        - 'resume': Resume paused plan
        """
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["execute", "status", "list", "pause", "resume"],
                    "description": "The operation to perform"
                },
                "plan_id": {
                    "type": "string",
                    "description": "Plan ID (optional, uses current plan if not specified)"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    
    def execute(
        self,
        operation: str,
        plan_id: str = None
    ) -> Dict[str, Any]:
        """Execute plan operation"""
        
        print(f"📋 Plan operation: {operation}")
        
        if operation == "execute":
            plan = self.planner.get_plan(plan_id) if plan_id else self.planner.current_plan
            
            if not plan:
                return {"success": False, "error": "No plan found"}
            
            result = self.planner.execute_plan(plan)
            
            return {
                "success": True,
                "status": result["status"],
                "progress": result["progress"],
                "final_output": result["final_output"]
            }
        
        elif operation == "status":
            plan = self.planner.get_plan(plan_id) if plan_id else self.planner.current_plan
            
            if not plan:
                return {"success": False, "error": "No plan found"}
            
            return {
                "success": True,
                "plan_id": plan.plan_id,
                "goal": plan.goal,
                "status": plan.status.value,
                "progress": plan.get_progress(),
                "steps": [{"id": s.step_id, "status": s.status.value} for s in plan.steps]
            }
        
        elif operation == "list":
            plans = self.planner.list_plans()
            
            return {
                "success": True,
                "plans": plans
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }