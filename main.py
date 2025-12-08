from config import Config
from agents.openai_agent import OpenAIAgent
from agents.gemini_agent import GeminiAgent
from tools.web_search import WebSearchTool
from tools.weather import WeatherTool
from tools.python_executor import PythonExecutorTool
from tools.file_manager import FileManagerTool

def main():
    """Main entry point"""
    
    # Load configuration
    config = Config()
    
    # Initialize tools
    tools = [
        WebSearchTool(),
        WeatherTool(config.OPENWEATHER_API_KEY),
        PythonExecutorTool(),
        FileManagerTool()
    ]
    
    # Choose agent
    print("🤖 Multi-Tool Personal Assistant")
    print("=" * 60)
    print("Choose your AI agent:")
    print("1. OpenAI (GPT-4)")
    print("2. Google Gemini")
    print("3. Both (compare responses)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        agent = OpenAIAgent(tools, config)
        print("\n✅ Using OpenAI GPT-4\n")
    elif choice == "2":
        agent = GeminiAgent(tools, config)
        print("\n✅ Using Google Gemini\n")
    else:
        print("\n✅ Using both agents\n")
        openai_agent = OpenAIAgent(tools, config)
        gemini_agent = GeminiAgent(tools, config)
    
    # Example queries
    print("Example queries:")
    print("  - 'What's the weather in Tokyo?'")
    print("  - 'Calculate the fibonacci sequence up to 10'")
    print("  - 'Search for latest AI news'")
    print("  - 'Create a file called notes.txt with hello world'")
    print("  - 'Read notes.txt'")
    print()
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if choice in ["1", "2"]:
                agent.chat(user_input)
            else:
                print("\n--- OpenAI Response ---")
                openai_agent.chat(user_input)
                print("\n--- Gemini Response ---")
                gemini_agent.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()