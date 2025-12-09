# ============================================================================
# FILE: planning/tree_of_thoughts.py
# ============================================================================
"""
Tree of Thoughts (ToT) Implementation

Explores multiple solution paths in a tree structure.
Supports:
- BFS (Breadth-First Search) exploration
- DFS (Depth-First Search) exploration
- Beam search with pruning
- Evaluation and backtracking
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
import heapq


class SearchStrategy(Enum):
    """Tree search strategies"""
    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    BEAM = "beam"  # Beam search with pruning
    BEST_FIRST = "best_first"  # Best-first search


class NodeStatus(Enum):
    """Status of a thought node"""
    ACTIVE = "active"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    SOLUTION = "solution"
    DEAD_END = "dead_end"


@dataclass
class ThoughtNode:
    """Represents a node in the thought tree"""
    node_id: str
    thought: str
    depth: int
    score: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.ACTIVE
    evaluation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "thought": self.thought,
            "depth": self.depth,
            "score": self.score,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "status": self.status.value,
            "evaluation": self.evaluation
        }
    
    def __lt__(self, other):
        """For heap comparison (higher score = higher priority)"""
        return self.score > other.score


@dataclass
class ThoughtTree:
    """Complete thought tree structure"""
    tree_id: str
    problem: str
    root_id: str
    nodes: Dict[str, ThoughtNode] = field(default_factory=dict)
    solution_path: List[str] = field(default_factory=list)
    best_solution: Optional[str] = None
    best_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    stats: Dict[str, int] = field(default_factory=lambda: {
        "nodes_created": 0,
        "nodes_expanded": 0,
        "nodes_pruned": 0,
        "max_depth": 0
    })
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        return self.nodes.get(node_id)
    
    def add_node(self, node: ThoughtNode):
        self.nodes[node.node_id] = node
        self.stats["nodes_created"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], node.depth)
    
    def get_path_to_node(self, node_id: str) -> List[ThoughtNode]:
        """Get the path from root to a node"""
        path = []
        current_id = node_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if node:
                path.append(node)
                current_id = node.parent_id
            else:
                break
        
        return list(reversed(path))
    
    def get_leaves(self) -> List[ThoughtNode]:
        """Get all leaf nodes"""
        return [
            node for node in self.nodes.values()
            if not node.children_ids and node.status == NodeStatus.ACTIVE
        ]
    
    def format_tree(self, node_id: str = None, indent: int = 0) -> str:
        """Format tree for display"""
        node_id = node_id or self.root_id
        node = self.nodes.get(node_id)
        
        if not node:
            return ""
        
        status_emoji = {
            NodeStatus.ACTIVE: "🔵",
            NodeStatus.EXPANDED: "🟢",
            NodeStatus.PRUNED: "🔴",
            NodeStatus.SOLUTION: "⭐",
            NodeStatus.DEAD_END: "⚫"
        }.get(node.status, "❓")
        
        lines = [
            "  " * indent + f"{status_emoji} [{node.score:.2f}] {node.thought[:60]}..."
        ]
        
        for child_id in node.children_ids:
            lines.append(self.format_tree(child_id, indent + 1))
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "problem": self.problem,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "solution_path": self.solution_path,
            "best_solution": self.best_solution,
            "best_score": self.best_score,
            "stats": self.stats
        }


class TreeOfThoughts:
    """
    Tree of Thoughts reasoning engine.
    
    Explores multiple solution paths and evaluates them to find
    the best solution to complex problems.
    """
    
    THOUGHT_GENERATION_PROMPT = """Given the problem and current thought process, generate {num_thoughts} different next steps or approaches.

Problem: {problem}

Current Path:
{current_path}

Generate {num_thoughts} distinct next thoughts or approaches. Each should be a different way to continue solving the problem.
Be creative and consider various angles.

Format each thought on a new line starting with a number:
1. [First approach]
2. [Second approach]
...

Generate the thoughts:"""

    EVALUATION_PROMPT = """Evaluate the following thought/approach for solving the problem.

Problem: {problem}

Thought Path:
{thought_path}

Current Thought to Evaluate:
{thought}

Evaluate this thought on a scale of 1-10 based on:
- Relevance to the problem (does it address the core issue?)
- Feasibility (is this approach practical?)
- Progress (does it move toward a solution?)
- Completeness (does it fully address what's needed?)

Provide your evaluation as:
Score: [1-10]
Reasoning: [Brief explanation]
Is Solution: [yes/no] (Is this a complete solution to the problem?)

Evaluate:"""

    SOLUTION_SYNTHESIS_PROMPT = """Based on the explored thought paths, synthesize the best solution.

Problem: {problem}

Best Thought Path:
{best_path}

Additional Explored Paths:
{other_paths}

Synthesize a comprehensive solution that incorporates the best insights:"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        strategy: SearchStrategy = SearchStrategy.BEAM,
        max_depth: int = 5,
        branching_factor: int = 3,
        beam_width: int = 3,
        min_score_threshold: float = 3.0,
        verbose: bool = True
    ):
        """
        Initialize Tree of Thoughts engine.
        
        Args:
            llm_fn: Function to call LLM
            strategy: Search strategy to use
            max_depth: Maximum tree depth
            branching_factor: Number of thoughts to generate per node
            beam_width: Number of paths to keep in beam search
            min_score_threshold: Minimum score to continue exploring
            verbose: Whether to print progress
        """
        self.llm_fn = llm_fn
        self.strategy = strategy
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width
        self.min_score_threshold = min_score_threshold
        self.verbose = verbose
        
        self.trees: Dict[str, ThoughtTree] = {}
        self.current_tree: Optional[ThoughtTree] = None
    
    def solve(self, problem: str, context: str = "") -> Dict[str, Any]:
        """
        Solve a problem using Tree of Thoughts.
        
        Args:
            problem: The problem to solve
            context: Additional context
            
        Returns:
            Solution with path and explanation
        """
        if self.verbose:
            print(f"\n🌳 Tree of Thoughts: Solving...")
            print(f"   Strategy: {self.strategy.value}")
            print(f"   Max Depth: {self.max_depth}")
            print(f"   Branching Factor: {self.branching_factor}")
        
        # Create tree
        tree = self._create_tree(problem, context)
        self.current_tree = tree
        
        # Explore based on strategy
        if self.strategy == SearchStrategy.BFS:
            self._bfs_explore(tree)
        elif self.strategy == SearchStrategy.DFS:
            self._dfs_explore(tree, tree.root_id)
        elif self.strategy == SearchStrategy.BEAM:
            self._beam_search(tree)
        else:  # BEST_FIRST
            self._best_first_search(tree)
        
        # Get best solution
        solution = self._extract_best_solution(tree)
        
        if self.verbose:
            print(f"\n📊 Exploration Stats:")
            print(f"   Nodes Created: {tree.stats['nodes_created']}")
            print(f"   Nodes Expanded: {tree.stats['nodes_expanded']}")
            print(f"   Nodes Pruned: {tree.stats['nodes_pruned']}")
            print(f"   Max Depth Reached: {tree.stats['max_depth']}")
            print(f"\n🌳 Tree Structure:")
            print(tree.format_tree())
        
        return solution
    
    def _create_tree(self, problem: str, context: str) -> ThoughtTree:
        """Create initial tree with root node"""
        import hashlib
        
        tree_id = hashlib.md5(f"{problem}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        root_id = "root"
        
        # Create root node with initial understanding
        root_thought = f"Problem: {problem}"
        if context:
            root_thought += f"\nContext: {context}"
        
        root_node = ThoughtNode(
            node_id=root_id,
            thought=root_thought,
            depth=0,
            score=5.0  # Neutral starting score
        )
        
        tree = ThoughtTree(
            tree_id=tree_id,
            problem=problem,
            root_id=root_id
        )
        tree.add_node(root_node)
        
        self.trees[tree_id] = tree
        
        return tree
    
    def _generate_thoughts(
        self,
        tree: ThoughtTree,
        parent_node: ThoughtNode
    ) -> List[ThoughtNode]:
        """Generate child thoughts for a node"""
        # Build current path
        path = tree.get_path_to_node(parent_node.node_id)
        path_text = "\n".join([f"Step {i+1}: {n.thought}" for i, n in enumerate(path)])
        
        prompt = self.THOUGHT_GENERATION_PROMPT.format(
            problem=tree.problem,
            current_path=path_text,
            num_thoughts=self.branching_factor
        )
        
        response = self.llm_fn(prompt)
        
        # Parse thoughts from response
        thoughts = self._parse_thoughts(response)
        
        # Create child nodes
        children = []
        for i, thought in enumerate(thoughts[:self.branching_factor]):
            node_id = f"{parent_node.node_id}_{i+1}"
            
            child = ThoughtNode(
                node_id=node_id,
                thought=thought,
                depth=parent_node.depth + 1,
                parent_id=parent_node.node_id
            )
            
            children.append(child)
            tree.add_node(child)
            parent_node.children_ids.append(node_id)
        
        parent_node.status = NodeStatus.EXPANDED
        tree.stats["nodes_expanded"] += 1
        
        return children
    
    def _evaluate_thought(
        self,
        tree: ThoughtTree,
        node: ThoughtNode
    ) -> Tuple[float, bool]:
        """Evaluate a thought node. Returns (score, is_solution)"""
        path = tree.get_path_to_node(node.node_id)
        path_text = "\n".join([f"Step {i+1}: {n.thought}" for i, n in enumerate(path[:-1])])
        
        prompt = self.EVALUATION_PROMPT.format(
            problem=tree.problem,
            thought_path=path_text or "Starting point",
            thought=node.thought
        )
        
        response = self.llm_fn(prompt)
        
        # Parse score and solution status
        score = self._parse_score(response)
        is_solution = self._parse_is_solution(response)
        
        node.score = score
        node.evaluation = response
        
        if is_solution:
            node.status = NodeStatus.SOLUTION
        elif score < self.min_score_threshold:
            node.status = NodeStatus.DEAD_END
        
        return score, is_solution
    
    def _bfs_explore(self, tree: ThoughtTree):
        """Breadth-first search exploration"""
        queue = [tree.root_id]
        
        while queue:
            node_id = queue.pop(0)
            node = tree.get_node(node_id)
            
            if not node or node.depth >= self.max_depth:
                continue
            
            if node.status in [NodeStatus.PRUNED, NodeStatus.DEAD_END]:
                continue
            
            if self.verbose:
                print(f"  Exploring: {node.thought[:50]}... (depth {node.depth})")
            
            # Generate children
            children = self._generate_thoughts(tree, node)
            
            # Evaluate and add to queue
            for child in children:
                score, is_solution = self._evaluate_thought(tree, child)
                
                if is_solution:
                    tree.solution_path = [n.node_id for n in tree.get_path_to_node(child.node_id)]
                    return
                
                if child.status == NodeStatus.ACTIVE:
                    queue.append(child.node_id)
    
    def _dfs_explore(self, tree: ThoughtTree, node_id: str) -> bool:
        """Depth-first search exploration. Returns True if solution found."""
        node = tree.get_node(node_id)
        
        if not node:
            return False
        
        if node.depth >= self.max_depth:
            return False
        
        if node.status in [NodeStatus.PRUNED, NodeStatus.DEAD_END]:
            return False
        
        if self.verbose:
            print(f"  {'  ' * node.depth}Exploring: {node.thought[:40]}...")
        
        # Evaluate if not root
        if node.depth > 0:
            score, is_solution = self._evaluate_thought(tree, node)
            
            if is_solution:
                tree.solution_path = [n.node_id for n in tree.get_path_to_node(node_id)]
                return True
            
            if node.status == NodeStatus.DEAD_END:
                return False
        
        # Generate and explore children
        children = self._generate_thoughts(tree, node)
        
        # Sort by score if we've evaluated
        children_with_scores = []
        for child in children:
            score, _ = self._evaluate_thought(tree, child)
            children_with_scores.append((score, child))
        
        children_with_scores.sort(reverse=True)
        
        for _, child in children_with_scores:
            if child.status == NodeStatus.SOLUTION:
                tree.solution_path = [n.node_id for n in tree.get_path_to_node(child.node_id)]
                return True
            
            if self._dfs_explore(tree, child.node_id):
                return True
        
        return False
    
    def _beam_search(self, tree: ThoughtTree):
        """Beam search - keep top k paths at each level"""
        current_beam = [tree.get_node(tree.root_id)]
        
        for depth in range(self.max_depth):
            if self.verbose:
                print(f"  Beam search depth {depth}, beam size: {len(current_beam)}")
            
            all_children = []
            
            for node in current_beam:
                if node.status in [NodeStatus.PRUNED, NodeStatus.DEAD_END, NodeStatus.SOLUTION]:
                    continue
                
                # Generate children
                children = self._generate_thoughts(tree, node)
                
                # Evaluate each child
                for child in children:
                    score, is_solution = self._evaluate_thought(tree, child)
                    
                    if is_solution:
                        tree.solution_path = [n.node_id for n in tree.get_path_to_node(child.node_id)]
                        tree.best_solution = child.thought
                        tree.best_score = score
                        return
                    
                    if child.status == NodeStatus.ACTIVE:
                        all_children.append(child)
            
            if not all_children:
                break
            
            # Keep top k children for next beam
            all_children.sort(key=lambda x: x.score, reverse=True)
            current_beam = all_children[:self.beam_width]
            
            # Prune others
            for child in all_children[self.beam_width:]:
                child.status = NodeStatus.PRUNED
                tree.stats["nodes_pruned"] += 1
        
        # If no solution found, take best path
        best_node = max(tree.nodes.values(), key=lambda x: x.score)
        tree.solution_path = [n.node_id for n in tree.get_path_to_node(best_node.node_id)]
        tree.best_score = best_node.score
    
    def _best_first_search(self, tree: ThoughtTree):
        """Best-first search using priority queue"""
        # Priority queue: (negative score for max-heap, node_id)
        heap = [(-5.0, tree.root_id)]  # Start with root
        
        while heap:
            neg_score, node_id = heapq.heappop(heap)
            node = tree.get_node(node_id)
            
            if not node or node.depth >= self.max_depth:
                continue
            
            if node.status in [NodeStatus.PRUNED, NodeStatus.DEAD_END]:
                continue
            
            if self.verbose:
                print(f"  Best-first: [{-neg_score:.2f}] {node.thought[:40]}...")
            
            # Check if already a solution
            if node.status == NodeStatus.SOLUTION:
                tree.solution_path = [n.node_id for n in tree.get_path_to_node(node_id)]
                return
            
            # Generate and evaluate children
            children = self._generate_thoughts(tree, node)
            
            for child in children:
                score, is_solution = self._evaluate_thought(tree, child)
                
                if is_solution:
                    tree.solution_path = [n.node_id for n in tree.get_path_to_node(child.node_id)]
                    tree.best_solution = child.thought
                    tree.best_score = score
                    return
                
                if child.status == NodeStatus.ACTIVE:
                    heapq.heappush(heap, (-score, child.node_id))
        
        # Take best found
        best_node = max(tree.nodes.values(), key=lambda x: x.score)
        tree.solution_path = [n.node_id for n in tree.get_path_to_node(best_node.node_id)]
        tree.best_score = best_node.score
    
    def _extract_best_solution(self, tree: ThoughtTree) -> Dict[str, Any]:
        """Extract and synthesize the best solution"""
        if not tree.solution_path:
            # Find best scoring node
            best_node = max(tree.nodes.values(), key=lambda x: x.score)
            tree.solution_path = [n.node_id for n in tree.get_path_to_node(best_node.node_id)]
        
        # Get the solution path
        path_nodes = [tree.get_node(nid) for nid in tree.solution_path]
        best_path = "\n".join([f"Step {i+1}: {n.thought}" for i, n in enumerate(path_nodes) if n])
        
        # Get other explored paths for comparison
        all_paths = []
        for node in tree.nodes.values():
            if node.status in [NodeStatus.EXPANDED, NodeStatus.SOLUTION] and node.node_id not in tree.solution_path:
                all_paths.append(f"- {node.thought[:100]}... (score: {node.score:.1f})")
        
        other_paths = "\n".join(all_paths[:5]) if all_paths else "No other significant paths explored"
        
        # Synthesize final solution
        prompt = self.SOLUTION_SYNTHESIS_PROMPT.format(
            problem=tree.problem,
            best_path=best_path,
            other_paths=other_paths
        )
        
        final_solution = self.llm_fn(prompt)
        tree.best_solution = final_solution
        
        return {
            "problem": tree.problem,
            "solution": final_solution,
            "reasoning_path": [n.thought for n in path_nodes if n],
            "score": tree.best_score,
            "stats": tree.stats,
            "tree_id": tree.tree_id
        }
    
    def _parse_thoughts(self, response: str) -> List[str]:
        """Parse generated thoughts from response"""
        thoughts = []
        
        # Look for numbered items
        pattern = r'(?:^|\n)\s*(\d+)[.):\s]+(.+?)(?=(?:\n\s*\d+[.):\s])|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for _, thought in matches:
            thought = thought.strip()
            if thought and len(thought) > 10:
                thoughts.append(thought)
        
        # Fallback: split by newlines
        if not thoughts:
            for line in response.split('\n'):
                line = line.strip()
                if line and len(line) > 20:
                    # Remove leading numbers/bullets
                    line = re.sub(r'^[\d\-\*\•\.]+\s*', '', line)
                    if line:
                        thoughts.append(line)
        
        return thoughts[:self.branching_factor]
    
    def _parse_score(self, response: str) -> float:
        """Parse score from evaluation response"""
        patterns = [
            r'[Ss]core[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s+out\s+of\s+10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
        
        # Default score based on sentiment
        positive_words = ['good', 'great', 'excellent', 'promising', 'correct', 'right']
        negative_words = ['bad', 'wrong', 'incorrect', 'poor', 'weak', 'fail']
        
        response_lower = response.lower()
        pos_count = sum(1 for w in positive_words if w in response_lower)
        neg_count = sum(1 for w in negative_words if w in response_lower)
        
        return 5.0 + pos_count - neg_count
    
    def _parse_is_solution(self, response: str) -> bool:
        """Parse whether thought is a complete solution"""
        patterns = [
            r'[Ii]s\s+[Ss]olution[:\s]+(yes|true)',
            r'complete\s+solution',
            r'solves?\s+the\s+problem',
            r'final\s+answer'
        ]
        
        response_lower = response.lower()
        for pattern in patterns:
            if re.search(pattern, response_lower):
                return True
        
        return False
    
    def get_tree(self, tree_id: str = None) -> Optional[ThoughtTree]:
        """Get a thought tree"""
        if tree_id:
            return self.trees.get(tree_id)
        return self.current_tree
    
    def visualize_tree(self, tree: ThoughtTree = None) -> str:
        """Get a visualization of the tree"""
        tree = tree or self.current_tree
        if not tree:
            return "No tree to visualize"
        
        return tree.format_tree()