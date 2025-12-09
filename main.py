# ============================================================================
# FILE: main.py (Updated with Memory Support)
# ============================================================================

from config import Config
from agents.openai_agent import OpenAIAgent
from agents.gemini_agent import GeminiAgent
from tools.web_search import WebSearchTool
from tools.weather import WeatherTool
from tools.python_executor import PythonExecutorTool
from tools.file_manager import FileManagerTool
from tools.memory_tool import MemoryTool, KnowledgeBaseTool
from memory.conversation_memory import ConversationMemory
from memory.vector_store import VectorStore
from memory.long_term_memory import LongTermMemory


def initialize_memory_system(config: Config, agent_type: str = "openai"):
    """Initialize the memory system components"""
    print("\n🧠 Initializing Memory System...")
    
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
    
    print("✅ Memory system initialized\n")
    
    return conversation_memory, long_term_memory, vector_store


def create_tools(config: Config, long_term_memory=None, vector_store=None):
    """Create all tools including memory tools"""
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


def print_memory_stats(agent):
    """Print memory statistics"""
    stats = agent.get_memory_stats()
    
    print("\n📊 Memory Statistics:")
    print("-" * 40)
    
    if "conversation" in stats:
        conv = stats["conversation"]
        print(f"  Conversation:")
        print(f"    - Active messages: {conv.get('active_messages', 0)}")
        print(f"    - Total messages: {conv.get('total_messages', 0)}")
        print(f"    - Summaries: {conv.get('summaries_count', 0)}")
    
    if "long_term" in stats:
        lt = stats["long_term"]
        print(f"  Long-term Memory:")
        print(f"    - Total memories: {lt.get('total_memories', 0)}")
        print(f"    - Categories: {', '.join(lt.get('categories', []))}")
    
    if "vector_store" in stats:
        vs = stats["vector_store"]
        print(f"  Vector Store:")
        print(f"    - Documents: {vs.get('total_documents', 0)}")
    
    print("-" * 40)


def demo_memory_features(agent):
    """Demonstrate memory features"""
    print("\n🎯 Memory Features Demo")
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
    agent.chat("Write me a simple web server")
    
    # Print stats
    print_memory_stats(agent)


def main():
    """Main entry point"""
    
    # Load configuration
    config = Config()
    
    print("🤖 Multi-Tool Personal Assistant with Memory")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✅ Conversation History & Summarization")
    print("  ✅ Long-term Memory (facts, preferences)")
    print("  ✅ Vector Database (semantic search)")
    print("  ✅ RAG (Retrieval Augmented Generation)")
    print()
    
    # Choose agent
    print("Choose your AI agent:")
    print("1. OpenAI (GPT-4)")
    print("2. Google Gemini")
    print("3. Demo memory features (OpenAI)")
    print("4. Demo memory features (Gemini)")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    agent_type = "openai" if choice in ["1", "3"] else "gemini"
    
    # Initialize memory system
    conversation_memory, long_term_memory, vector_store = initialize_memory_system(
        config, agent_type
    )
    
    # Create tools with memory tools
    tools = create_tools(config, long_term_memory, vector_store)
    
    # Create agent
    if choice in ["1", "3"]:
        agent = OpenAIAgent(
            tools=tools,
            config=config,
            conversation_memory=conversation_memory,
            long_term_memory=long_term_memory,
            vector_store=vector_store
        )
        print(f"\n✅ Using OpenAI GPT-4 with Memory\n")
    else:
        agent = GeminiAgent(
            tools=tools,
            config=config,
            conversation_memory=conversation_memory,
            long_term_memory=long_term_memory,
            vector_store=vector_store
        )
        print(f"\n✅ Using Google Gemini with Memory\n")
    
    # Run demo if selected
    if choice in ["3", "4"]:
        demo_memory_features(agent)
        return
    
    # Example queries
    print("Example queries:")
    print("  - 'My name is [name]' - The assistant will remember this")
    print("  - 'What do you know about me?' - Recall stored information")
    print("  - 'Remember that I prefer...' - Store preferences")
    print("  - 'What's the weather?' - Use tools with memory context")
    print("  - '/stats' - Show memory statistics")
    print("  - '/clear' - Clear conversation history")
    print("  - '/memories' - Show stored memories")
    print()
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # Save conversation before exiting
                if conversation_memory:
                    conversation_memory.save_conversation()
                print("💾 Conversation saved. Goodbye! 👋")
                break
            
            if user_input.lower() == '/stats':
                print_memory_stats(agent)
                continue
            
            if user_input.lower() == '/clear':
                agent.reset_conversation()
                print("🗑️ Conversation cleared")
                continue
            
            if user_input.lower() == '/memories':
                if long_term_memory:
                    memories = long_term_memory.recall(limit=10)
                    print("\n📝 Stored Memories:")
                    for m in memories:
                        print(f"  [{m.memory_type}] {m.content[:80]}...")
                    print()
                continue
            
            if user_input.lower() == '/help':
                print("\nCommands:")
                print("  /stats    - Show memory statistics")
                print("  /clear    - Clear conversation history")
                print("  /memories - Show stored memories")
                print("  /help     - Show this help")
                print("  exit      - Exit the assistant")
                print()
                continue
            
            # Regular chat
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            if conversation_memory:
                conversation_memory.save_conversation()
            print("\n💾 Conversation saved. Goodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()