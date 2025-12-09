# ============================================================================
# FILE: main.py (Updated with Planning & Reasoning Support)
# ============================================================================

import sys
from typing import Optional

from config import Config
from agents.openai_agent import OpenAIAgent
from agents.gemini_agent import GeminiAgent

# Tools
from tools.web_search import WebSearchTool
from tools.weather import WeatherTool
from tools.python_executor import PythonExecutorTool
from tools.file_manager import FileManagerTool
from tools.memory_tool import MemoryTool, KnowledgeBaseTool
from tools.planning_tool import ReasoningTool, PlanExecutorTool

# Memory
from memory.conversation_memory import ConversationMemory
from memory.vector_store import VectorStore
from memory.long_term_memory import LongTermMemory


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🤖 Multi-Tool Personal Assistant v3.0                       ║
║                                                                  ║
║     Features:                                                    ║
║     ├─ 🧠 Chain of Thought Reasoning                            ║
║     ├─ 📋 Multi-Step Planning                                   ║
║     ├─ 🌳 Tree of Thoughts Exploration                          ║
║     ├─ 💾 Long-term Memory                                      ║
║     ├─ 🔍 Semantic Search (RAG)                                 ║
║     └─ 🔧 Tool Integration                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def initialize_memory_system(config: Config, agent_type: str = "openai"):
    """Initialize the memory system components"""
    print("\n💾 Initializing Memory System...")
    
    # Create vector store
    vector_store = VectorStore(
        config=config,
        collection_name=f"{config.COLLECTION_NAME}_{agent_type}"
    )
    
    # Create long-term memory with vector store
    long_term_memory = LongTermMemory(
        config=config,
        vector_store=vector_store
    )
    
    # Create conversation memory
    conversation_memory = ConversationMemory(
        config=config,
        storage_path=f"./conversations_{agent_type}"
    )
    
    print("   ✅ Memory system initialized")
    
    return conversation_memory, long_term_memory, vector_store


def create_tools(config: Config, long_term_memory=None, vector_store=None):
    """Create all tools including memory and reasoning tools"""
    tools = [
        WebSearchTool(),
        WeatherTool(config.OPENWEATHER_API_KEY),
        PythonExecutorTool(),
        FileManagerTool()
    ]
    
    # Add memory tools if memory system is available
    if long_term_memory and vector_store:
        tools.append(MemoryTool(long_term_memory, vector_store))
        tools.append(KnowledgeBaseTool(vector_store))
    
    return tools


def add_reasoning_tools(tools: list, agent) -> list:
    """Add reasoning tools after agent is initialized"""
    # Create reasoning tool with agent's reasoning components
    reasoning_tool = ReasoningTool(
        cot_engine=agent.chain_of_thought,
        planner=agent.planner,
        tot_engine=agent.tree_of_thoughts
    )
    tools.append(reasoning_tool)
    
    # Add plan executor tool if planner is available
    if agent.planner:
        tools.append(PlanExecutorTool(agent.planner))
    
    # Update agent's tools
    agent.tools[reasoning_tool.name] = reasoning_tool
    if agent.planner:
        agent.tools["plan_executor"] = PlanExecutorTool(agent.planner)
    
    return tools


def print_stats(agent):
    """Print comprehensive statistics"""
    stats = agent.get_full_stats()
    
    print("\n" + "=" * 60)
    print("📊 ASSISTANT STATISTICS")
    print("=" * 60)
    
    # Memory stats
    if "memory" in stats:
        print("\n💾 MEMORY:")
        mem = stats["memory"]
        
        if "conversation" in mem:
            conv = mem["conversation"]
            print(f"   Conversation:")
            print(f"     • Active messages: {conv.get('active_messages', 0)}")
            print(f"     • Total messages: {conv.get('total_messages', 0)}")
            print(f"     • Summaries: {conv.get('summaries_count', 0)}")
        
        if "long_term" in mem:
            lt = mem["long_term"]
            print(f"   Long-term Memory:")
            print(f"     • Total memories: {lt.get('total_memories', 0)}")
            print(f"     • Categories: {', '.join(lt.get('categories', []))}")
        
        if "vector_store" in mem:
            vs = mem["vector_store"]
            print(f"   Vector Store:")
            print(f"     • Documents: {vs.get('total_documents', 0)}")
    
    # Reasoning stats
    if "reasoning" in stats:
        print("\n🧠 REASONING:")
        reas = stats["reasoning"]
        print(f"   • Reasoning enabled: {reas.get('reasoning_enabled', False)}")
        print(f"   • Chain of Thought: {'✅' if reas.get('cot_available') else '❌'}")
        print(f"   • Planner: {'✅' if reas.get('planner_available') else '❌'}")
        print(f"   • Tree of Thoughts: {'✅' if reas.get('tot_available') else '❌'}")
        
        if reas.get('cot_history_count', 0) > 0:
            print(f"   • CoT reasoning chains: {reas['cot_history_count']}")
        
        if reas.get('plans_count', 0) > 0:
            print(f"   • Plans created: {reas['plans_count']}")
        
        if "current_plan" in reas:
            plan = reas["current_plan"]
            print(f"   • Active plan: {plan['goal'][:40]}...")
            print(f"     Status: {plan['status']}")
            print(f"     Progress: {plan['progress']['progress_pct']:.1f}%")
    
    # Tools
    if "tools" in stats:
        print(f"\n🔧 TOOLS: {', '.join(stats['tools'])}")
    
    print("\n" + "=" * 60)


def print_help():
    """Print help information"""
    help_text = """
╔══════════════════════════════════════════════════════════════════╗
║                        HELP & COMMANDS                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  GENERAL COMMANDS:                                               ║
║  ─────────────────                                               ║
║  /help          - Show this help message                         ║
║  /stats         - Show memory and reasoning statistics           ║
║  /clear         - Clear conversation history                     ║
║  /exit or exit  - Exit the assistant                            ║
║                                                                  ║
║  MEMORY COMMANDS:                                                ║
║  ────────────────                                                ║
║  /memories      - List stored memories                           ║
║  /forget        - Clear all memories (use with caution)          ║
║                                                                  ║
║  REASONING COMMANDS:                                             ║
║  ──────────────────                                              ║
║  /think <query>     - Apply Chain of Thought reasoning           ║
║  /plan <goal>       - Create a multi-step plan                   ║
║  /explore <problem> - Explore solutions with Tree of Thoughts    ║
║  /execute           - Execute the current plan                   ║
║                                                                  ║
║  REASONING MODES:                                                ║
║  ────────────────                                                ║
║  /mode auto     - Auto-detect when to use reasoning (default)    ║
║  /mode always   - Always use advanced reasoning                  ║
║  /mode never    - Disable advanced reasoning                     ║
║                                                                  ║
║  REASONING STRATEGIES:                                           ║
║  ─────────────────────                                           ║
║  /cot zero_shot      - Use zero-shot CoT                         ║
║  /cot few_shot       - Use few-shot CoT with examples            ║
║  /cot structured     - Use structured reasoning (default)        ║
║  /cot self_consistency - Use self-consistency (multiple paths)   ║
║                                                                  ║
║  /tot bfs        - Tree of Thoughts: Breadth-first search        ║
║  /tot dfs        - Tree of Thoughts: Depth-first search          ║
║  /tot beam       - Tree of Thoughts: Beam search (default)       ║
║  /tot best_first - Tree of Thoughts: Best-first search           ║
║                                                                  ║
║  EXAMPLE QUERIES:                                                ║
║  ────────────────                                                ║
║  • "My name is Alex and I'm a developer"                         ║
║  • "What do you remember about me?"                              ║
║  • "Explain how neural networks learn"                           ║
║  • "Create a plan to learn Python in 30 days"                    ║
║  • "What's the best approach to optimize a slow database?"       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def handle_command(agent, command: str) -> Optional[str]:
    """Handle special commands. Returns response or None to continue."""
    
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    # General commands
    if cmd in ['/exit', 'exit', 'quit', 'bye']:
        if agent.conversation_memory:
            agent.conversation_memory.save_conversation()
        print("\n💾 Conversation saved. Goodbye! 👋\n")
        sys.exit(0)
    
    if cmd == '/help':
        print_help()
        return ""
    
    if cmd == '/stats':
        print_stats(agent)
        return ""
    
    if cmd == '/clear':
        agent.reset_conversation()
        print("🗑️ Conversation cleared")
        return ""
    
    # Memory commands
    if cmd == '/memories':
        if agent.long_term_memory:
            memories = agent.long_term_memory.recall(limit=10)
            print("\n📝 Stored Memories:")
            if memories:
                for m in memories:
                    print(f"  [{m.memory_type}] {m.content[:80]}...")
            else:
                print("  No memories stored yet.")
            print()
        return ""
    
    if cmd == '/forget':
        confirm = input("⚠️ This will clear all memories. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            if agent.long_term_memory:
                agent.long_term_memory.memories.clear()
                agent.long_term_memory.categories.clear()
                agent.long_term_memory._save_memories()
            if agent.vector_store:
                agent.vector_store.clear()
            print("🗑️ All memories cleared")
        else:
            print("❌ Operation cancelled")
        return ""
    
    # Reasoning commands
    if cmd == '/think':
        if not args:
            print("❌ Usage: /think <query>")
            return ""
        result = agent.think_about(args)
        print(result)
        return ""
    
    if cmd == '/plan':
        if not args:
            print("❌ Usage: /plan <goal>")
            return ""
        result = agent.create_plan_for(args)
        print(result)
        return ""
    
    if cmd == '/explore':
        if not args:
            print("❌ Usage: /explore <problem>")
            return ""
        result = agent.explore_solutions(args)
        print(result)
        return ""
    
    if cmd == '/execute':
        if not agent.current_plan:
            print("❌ No plan to execute. Create one with /plan first.")
            return ""
        result = agent.execute_plan()
        print(f"\n📋 Plan Execution Result:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Progress: {result.get('progress', {}).get('progress_pct', 0):.1f}%")
        if result.get('final_output'):
            print(f"\n   Output:\n   {result['final_output']}")
        return ""
    
    # Reasoning mode commands
    if cmd == '/mode':
        if args in ['auto', 'always', 'never']:
            agent.set_reasoning_mode(args)
        else:
            print("❌ Usage: /mode <auto|always|never>")
            print(f"   Current mode: {agent.config.REASONING_MODE}")
        return ""
    
    # CoT strategy commands
    if cmd == '/cot':
        strategies = ['zero_shot', 'few_shot', 'structured', 'self_consistency']
        if args in strategies:
            if agent.chain_of_thought:
                from planning.chain_of_thought import CoTStrategy
                agent.chain_of_thought.strategy = CoTStrategy(args)
                print(f"✅ CoT strategy set to: {args}")
            else:
                print("❌ Chain of Thought not available")
        else:
            print(f"❌ Usage: /cot <{'/'.join(strategies)}>")
        return ""
    
    # ToT strategy commands
    if cmd == '/tot':
        strategies = ['bfs', 'dfs', 'beam', 'best_first']
        if args in strategies:
            if agent.tree_of_thoughts:
                from planning.tree_of_thoughts import SearchStrategy
                agent.tree_of_thoughts.strategy = SearchStrategy(args)
                print(f"✅ ToT strategy set to: {args}")
            else:
                print("❌ Tree of Thoughts not available")
        else:
            print(f"❌ Usage: /tot <{'/'.join(strategies)}>")
        return ""
    
    # Not a command
    return None


def demo_reasoning_features(agent):
    """Demonstrate reasoning features"""
    print("\n" + "=" * 60)
    print("🎯 REASONING FEATURES DEMO")
    print("=" * 60)
    
    # Demo 1: Chain of Thought
    print("\n" + "-" * 60)
    print("1️⃣  CHAIN OF THOUGHT REASONING")
    print("-" * 60)
    query1 = "If a train leaves at 2pm traveling at 60mph, and another train leaves at 3pm traveling at 80mph, when will the second train catch up?"
    print(f"\nQuery: {query1}\n")
    result = agent.think_about(query1)
    print(result)
    
    # Demo 2: Multi-Step Planning
    print("\n" + "-" * 60)
    print("2️⃣  MULTI-STEP PLANNING")
    print("-" * 60)
    goal = "Create a simple Python web scraper to extract headlines from a news website"
    print(f"\nGoal: {goal}\n")
    result = agent.create_plan_for(goal)
    print(result)
    
    # Demo 3: Tree of Thoughts
    print("\n" + "-" * 60)
    print("3️⃣  TREE OF THOUGHTS EXPLORATION")
    print("-" * 60)
    problem = "What is the best approach to learn machine learning for a beginner programmer?"
    print(f"\nProblem: {problem}\n")
    result = agent.explore_solutions(problem)
    print(result)
    
    # Print final stats
    print_stats(agent)
    
    print("\n✅ Demo complete! Try these features yourself with /think, /plan, and /explore commands.\n")


def demo_memory_features(agent):
    """Demonstrate memory features"""
    print("\n" + "=" * 60)
    print("🎯 MEMORY FEATURES DEMO")
    print("=" * 60)
    
    # Example 1: Remember user information
    print("\n1️⃣  Testing: Remembering user information")
    agent.chat("My name is Alex and I'm a software developer from San Francisco. I love hiking and coffee.")
    
    # Example 2: Recall information
    print("\n2️⃣  Testing: Recalling information")
    agent.chat("What do you remember about me?")
    
    # Example 3: Set preferences
    print("\n3️⃣  Testing: Setting preferences")
    agent.chat("Remember that I prefer Python over JavaScript and I like to receive concise answers.")
    
    # Example 4: Use preferences
    print("\n4️⃣  Testing: Using preferences")
    agent.chat("Write me a simple hello world program")
    
    # Print stats
    print_stats(agent)


def select_agent() -> str:
    """Let user select which agent to use"""
    print("\n🤖 Select AI Agent:")
    print("─" * 40)
    print("  1. OpenAI GPT-4")
    print("  2. Google Gemini")
    print("  3. Demo: Reasoning Features (OpenAI)")
    print("  4. Demo: Reasoning Features (Gemini)")
    print("  5. Demo: Memory Features (OpenAI)")
    print("  6. Demo: Memory Features (Gemini)")
    print("─" * 40)
    
    while True:
        choice = input("\nEnter choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print("❌ Invalid choice. Please enter 1-6.")


def create_agent(choice: str, config: Config):
    """Create the appropriate agent based on user choice"""
    
    agent_type = "openai" if choice in ["1", "3", "5"] else "gemini"
    
    # Initialize memory system
    conversation_memory, long_term_memory, vector_store = initialize_memory_system(
        config, agent_type
    )
    
    # Create tools
    tools = create_tools(config, long_term_memory, vector_store)
    
    print(f"\n🔧 Initializing {agent_type.upper()} Agent...")
    
    # Create agent
    if agent_type == "openai":
        agent = OpenAIAgent(
            tools=tools,
            config=config,
            conversation_memory=conversation_memory,
            long_term_memory=long_term_memory,
            vector_store=vector_store
        )
    else:
        agent = GeminiAgent(
            tools=tools,
            config=config,
            conversation_memory=conversation_memory,
            long_term_memory=long_term_memory,
            vector_store=vector_store
        )
    
    # Add reasoning tools
    add_reasoning_tools(tools, agent)
    
    print(f"   ✅ {agent_type.upper()} Agent initialized with reasoning capabilities")
    
    return agent, choice


def main():
    """Main entry point"""
    
    # Print banner
    print_banner()
    
    # Load configuration
    config = Config()
    
    # Select and create agent
    choice = select_agent()
    agent, choice = create_agent(choice, config)
    
    # Run demos if selected
    if choice in ["3", "4"]:
        demo_reasoning_features(agent)
        return
    
    if choice in ["5", "6"]:
        demo_memory_features(agent)
        return
    
    # Print help hint
    print("\n💡 Type /help for available commands\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.startswith('/') or user_input.lower() in ['exit', 'quit', 'bye']:
                result = handle_command(agent, user_input)
                if result is not None:
                    continue
            
            # Regular chat
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            if agent.conversation_memory:
                agent.conversation_memory.save_conversation()
            print("\n\n💾 Conversation saved. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()