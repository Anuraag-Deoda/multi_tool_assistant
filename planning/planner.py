# ============================================================================
# FILE: planning/planner.py
# ============================================================================
"""
Multi-Step Planning Implementation

Creates and executes structured plans for complex tasks.
Supports:
- Plan generation with steps and dependencies
- Dynamic plan adjustment
- Plan execution with progress tracking
- Rollback and error handling
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re


class StepStatus(Enum):
    """Status of a plan step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(Enum):
    """Status of the overall plan"""
    DRAFT = "draft"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PlanStep:
    """Represents a single step in a plan"""
    step_id: str
    description: str
    action: str  # The action to take (tool call or response)
    expected_output: str
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite steps
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    estimated_time: Optional[int] = None  # seconds
    actual_time: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "expected_output": self.expected_output,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        data['status'] = StepStatus(data.get('status', 'pending'))
        return cls(**data)


@dataclass
class Plan:
    """Represents a complete execution plan"""
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)  # step_id -> result
    final_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "context": self.context,
            "results": self.results,
            "final_output": self.final_output
        }
    
    def get_next_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (dependencies met)"""
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        
        ready = []
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                if all(dep in completed_ids for dep in step.dependencies):
                    ready.append(step)
        
        return ready
    
    def get_progress(self) -> Dict[str, int]:
        """Get plan execution progress"""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        pending = sum(1 for s in self.steps if s.status == StepStatus.PENDING)
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "progress_pct": (completed / total * 100) if total > 0 else 0
        }
    
    def format_for_display(self) -> str:
        """Format plan for display"""
        lines = [
            f"📋 Plan: {self.goal}",
            f"   Status: {self.status.value}",
            f"   Steps: {len(self.steps)}",
            "-" * 50
        ]
        
        for step in self.steps:
            status_emoji = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }.get(step.status, "❓")
            
            lines.append(f"  {status_emoji} [{step.step_id}] {step.description}")
            if step.dependencies:
                lines.append(f"      ↳ Depends on: {', '.join(step.dependencies)}")
            if step.result:
                lines.append(f"      → Result: {step.result[:100]}...")
        
        progress = self.get_progress()
        lines.append("-" * 50)
        lines.append(f"Progress: {progress['progress_pct']:.1f}% ({progress['completed']}/{progress['total']})")
        
        return "\n".join(lines)


class Planner:
    """
    Multi-step planning engine.
    
    Creates structured plans from goals and executes them step by step.
    """
    
    PLANNING_PROMPT = """Create a detailed execution plan for the following goal.

Goal: {goal}

Available Tools:
{tools_description}

Context:
{context}

Create a step-by-step plan with:
1. Clear, atomic steps
2. Dependencies between steps (what must complete before each step)
3. The tool to use for each step (or "respond" for generating text)
4. Expected output of each step

Format your response as JSON:
{{
    "steps": [
        {{
            "step_id": "step_1",
            "description": "Description of what this step does",
            "action": "What action to take",
            "tool_name": "tool_name or null",
            "tool_args": {{}},
            "dependencies": [],
            "expected_output": "What this step should produce"
        }},
        ...
    ],
    "reasoning": "Why this plan will achieve the goal"
}}

Create the plan:"""

    REPLAN_PROMPT = """The current plan needs adjustment due to the following:

Original Goal: {goal}
Issue: {issue}
Completed Steps: {completed_steps}
Failed Step: {failed_step}
Error: {error}

Create an updated plan that:
1. Accounts for what's already completed
2. Addresses the failure
3. Still achieves the original goal

{format_instructions}"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        tool_executor: Optional[Callable[[str, Dict], Any]] = None,
        available_tools: Dict[str, Any] = None,
        verbose: bool = True
    ):
        """
        Initialize the Planner.
        
        Args:
            llm_fn: Function to call LLM
            tool_executor: Function to execute tools (tool_name, args) -> result
            available_tools: Dictionary of available tools
            verbose: Whether to print planning progress
        """
        self.llm_fn = llm_fn
        self.tool_executor = tool_executor
        self.available_tools = available_tools or {}
        self.verbose = verbose
        self.plans: Dict[str, Plan] = {}
        self.current_plan: Optional[Plan] = None
    
    def create_plan(
        self,
        goal: str,
        context: Dict[str, Any] = None
    ) -> Plan:
        """
        Create a plan for achieving a goal.
        
        Args:
            goal: The goal to achieve
            context: Additional context for planning
            
        Returns:
            Plan object with steps
        """
        if self.verbose:
            print(f"\n📋 Creating plan for: {goal}")
        
        # Build tools description
        tools_desc = self._format_tools_description()
        
        # Build context string
        context_str = json.dumps(context or {}, indent=2)
        
        # Generate plan using LLM
        prompt = self.PLANNING_PROMPT.format(
            goal=goal,
            tools_description=tools_desc,
            context=context_str
        )
        
        response = self.llm_fn(prompt)
        
        # Parse plan from response
        plan = self._parse_plan_response(response, goal)
        plan.context = context or {}
        
        # Validate plan
        self._validate_plan(plan)
        
        # Store plan
        self.plans[plan.plan_id] = plan
        self.current_plan = plan
        
        if self.verbose:
            print(plan.format_for_display())
        
        return plan
    
    def execute_plan(
        self,
        plan: Plan = None,
        step_callback: Optional[Callable[[PlanStep], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan step by step.
        
        Args:
            plan: Plan to execute (uses current_plan if None)
            step_callback: Optional callback after each step
            
        Returns:
            Execution results
        """
        plan = plan or self.current_plan
        
        if not plan:
            return {"error": "No plan to execute"}
        
        if self.verbose:
            print(f"\n🚀 Executing plan: {plan.goal}")
        
        plan.status = PlanStatus.EXECUTING
        
        while True:
            # Get next steps that can be executed
            next_steps = plan.get_next_steps()
            
            if not next_steps:
                # Check if plan is complete or stuck
                progress = plan.get_progress()
                if progress['pending'] == 0:
                    plan.status = PlanStatus.COMPLETED
                    break
                elif progress['failed'] > 0:
                    plan.status = PlanStatus.FAILED
                    break
                else:
                    # No progress possible - circular dependency?
                    plan.status = PlanStatus.FAILED
                    break
            
            # Execute ready steps (could be parallelized)
            for step in next_steps:
                success = self._execute_step(plan, step)
                
                if step_callback:
                    step_callback(step)
                
                if not success and step.retry_count >= step.max_retries:
                    # Step failed - try to replan
                    if self.verbose:
                        print(f"⚠️ Step {step.step_id} failed. Attempting replan...")
                    
                    new_plan = self._replan(plan, step)
                    if new_plan:
                        plan = new_plan
                        self.current_plan = plan
                    else:
                        plan.status = PlanStatus.FAILED
                        break
        
        # Generate final output
        if plan.status == PlanStatus.COMPLETED:
            plan.final_output = self._generate_final_output(plan)
        
        if self.verbose:
            print(f"\n✅ Plan execution {'completed' if plan.status == PlanStatus.COMPLETED else 'failed'}")
            print(plan.format_for_display())
        
        return {
            "status": plan.status.value,
            "results": plan.results,
            "final_output": plan.final_output,
            "progress": plan.get_progress()
        }
    
    def _execute_step(self, plan: Plan, step: PlanStep) -> bool:
        """Execute a single plan step"""
        if self.verbose:
            print(f"  🔄 Executing: [{step.step_id}] {step.description}")
        
        step.status = StepStatus.IN_PROGRESS
        
        try:
            if step.tool_name and step.tool_name != "respond":
                # Execute tool
                if self.tool_executor:
                    result = self.tool_executor(step.tool_name, step.tool_args)
                else:
                    result = {"error": "No tool executor configured"}
                
                step.result = json.dumps(result) if isinstance(result, dict) else str(result)
            
            else:
                # Generate response using LLM
                context = self._build_step_context(plan, step)
                prompt = f"""Based on the plan and previous results, complete this step:

Plan Goal: {plan.goal}
Current Step: {step.description}
Action: {step.action}
Previous Results: {context}

Provide the output for this step:"""
                
                step.result = self.llm_fn(prompt)
            
            step.status = StepStatus.COMPLETED
            plan.results[step.step_id] = step.result
            
            if self.verbose:
                print(f"    ✅ Completed: {step.result[:100]}...")
            
            return True
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.retry_count += 1
            
            if self.verbose:
                print(f"    ❌ Failed: {e}")
            
            return False
    
    def _replan(self, plan: Plan, failed_step: PlanStep) -> Optional[Plan]:
        """Create a new plan after a step failure"""
        completed = [s for s in plan.steps if s.status == StepStatus.COMPLETED]
        completed_desc = "\n".join([f"- {s.description}: {s.result[:100]}" for s in completed])
        
        prompt = self.REPLAN_PROMPT.format(
            goal=plan.goal,
            issue="Step execution failed",
            completed_steps=completed_desc or "None",
            failed_step=failed_step.description,
            error=failed_step.error,
            format_instructions="Create a new plan in JSON format as before."
        )
        
        try:
            response = self.llm_fn(prompt)
            new_plan = self._parse_plan_response(response, plan.goal)
            
            # Copy completed results
            new_plan.results = plan.results.copy()
            new_plan.context = plan.context
            
            return new_plan
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Replan failed: {e}")
            return None
    
    def _parse_plan_response(self, response: str, goal: str) -> Plan:
        """Parse LLM response into Plan object"""
        import hashlib
        
        plan_id = hashlib.md5(f"{goal}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                steps = []
                
                for i, step_data in enumerate(data.get('steps', [])):
                    step = PlanStep(
                        step_id=step_data.get('step_id', f'step_{i+1}'),
                        description=step_data.get('description', f'Step {i+1}'),
                        action=step_data.get('action', ''),
                        expected_output=step_data.get('expected_output', ''),
                        dependencies=step_data.get('dependencies', []),
                        tool_name=step_data.get('tool_name'),
                        tool_args=step_data.get('tool_args', {})
                    )
                    steps.append(step)
                
                return Plan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=steps,
                    status=PlanStatus.READY
                )
                
            except json.JSONDecodeError:
                pass
        
        # Fallback: parse as simple steps
        return self._parse_simple_plan(response, goal, plan_id)
    
    def _parse_simple_plan(self, response: str, goal: str, plan_id: str) -> Plan:
        """Parse response as simple numbered steps"""
        steps = []
        
        # Look for numbered items
        pattern = r'(?:Step\s*)?(\d+)[.:]\s*(.+?)(?=(?:Step\s*)?\d+[.:]|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for i, (num, content) in enumerate(matches):
            step = PlanStep(
                step_id=f"step_{i+1}",
                description=content.strip()[:200],
                action=content.strip()[:200],
                expected_output="Completed step",
                dependencies=[f"step_{i}"] if i > 0 else []
            )
            steps.append(step)
        
        if not steps:
            # Create single step
            steps.append(PlanStep(
                step_id="step_1",
                description=goal,
                action=goal,
                expected_output="Goal achieved"
            ))
        
        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            status=PlanStatus.READY
        )
    
    def _validate_plan(self, plan: Plan):
        """Validate plan structure"""
        step_ids = {s.step_id for s in plan.steps}
        
        for step in plan.steps:
            # Check dependencies exist
            for dep in step.dependencies:
                if dep not in step_ids:
                    if self.verbose:
                        print(f"⚠️ Unknown dependency {dep} in step {step.step_id}")
                    step.dependencies.remove(dep)
            
            # Check for self-dependency
            if step.step_id in step.dependencies:
                step.dependencies.remove(step.step_id)
    
    def _build_step_context(self, plan: Plan, step: PlanStep) -> str:
        """Build context from previous step results"""
        context_parts = []
        
        for dep_id in step.dependencies:
            if dep_id in plan.results:
                result = plan.results[dep_id]
                context_parts.append(f"{dep_id}: {result[:200]}")
        
        return "\n".join(context_parts) or "No previous results"
    
    def _generate_final_output(self, plan: Plan) -> str:
        """Generate final output from all step results"""
        results_summary = "\n".join([
            f"- {step.description}: {plan.results.get(step.step_id, 'N/A')[:200]}"
            for step in plan.steps
            if step.status == StepStatus.COMPLETED
        ])
        
        prompt = f"""Based on the completed plan, provide a final summary:

Goal: {plan.goal}

Completed Steps:
{results_summary}

Provide a comprehensive final output that addresses the original goal:"""
        
        return self.llm_fn(prompt)
    
    def _format_tools_description(self) -> str:
        """Format available tools for planning prompt"""
        if not self.available_tools:
            return "No tools available. Use 'respond' for text generation."
        
        lines = []
        for name, tool in self.available_tools.items():
            desc = getattr(tool, 'description', 'No description')
            lines.append(f"- {name}: {desc[:100]}")
        
        lines.append("- respond: Generate text response (no tool)")
        
        return "\n".join(lines)
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID"""
        return self.plans.get(plan_id)
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans"""
        return [
            {
                "plan_id": p.plan_id,
                "goal": p.goal,
                "status": p.status.value,
                "progress": p.get_progress()
            }
            for p in self.plans.values()
        ]