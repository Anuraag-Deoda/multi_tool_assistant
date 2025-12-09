# 🤖 Multi-Tool Personal Assistant

A **modular agentic AI system** that teaches the fundamentals of building intelligent agents with tool-calling capabilities. Supports both **OpenAI GPT-4** and **Google Gemini** with a clean, extensible architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Available Tools](#available-tools)
- [Code Walkthrough](#code-walkthrough)
- [Agentic Flow](#agentic-flow)
- [Extending the Project](#extending-the-project)
- [Learning Outcomes](#learning-outcomes)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## 🎯 Overview

This project implements a **tool-using agent** (also called function-calling agent) that can:

- 🔍 Search the web for current information
- 🌤️ Get weather data for any location
- 🐍 Execute Python code safely
- 📄 Create, read, and append files

**Key Features:**
- Supports multiple AI providers (OpenAI & Gemini)
- Modular, extensible architecture
- Real agentic loops (agent can chain multiple tools)
- Clean separation of concerns
- Educational code with extensive comments

---

## 🏗️ Architecture

### High-Level Design

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │ Query
       ↓
┌─────────────────┐
│   Main.py       │ ← Entry point
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Agent Layer    │ ← OpenAI or Gemini
│  - OpenAIAgent  │
│  - GeminiAgent  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Tool Layer    │ ← Modular tools
│  - WebSearch    │
│  - Weather      │
│  - Python       │
│  - FileManager  │
└─────────────────┘
```

### Design Patterns Used

1. **Abstract Factory Pattern**: `BaseAgent` and `BaseTool` define interfaces
2. **Strategy Pattern**: Swap between OpenAI and Gemini
3. **Command Pattern**: Tools encapsulate actions
4. **Template Method**: Agentic loop structure

---

## 🔄 How It Works

### The Agentic Loop (ReAct Pattern)

```
1. User sends query → "What's the weather in Tokyo?"

2. Agent THINKS:
   "I need weather data. I should use get_weather tool."

3. Agent ACTS:
   Calls: get_weather(location="Tokyo")

4. Tool EXECUTES:
   Returns: {"temperature": 15, "condition": "Cloudy"}

5. Agent THINKS again:
   "Now I have the data. Let me format a response."

6. Agent RESPONDS:
   "The weather in Tokyo is currently 15°C and Cloudy."
```

### Function Calling Flow

```python
# Step 1: User message
"Calculate fibonacci up to 10"

# Step 2: LLM decides to use tool
{
  "function": "run_python",
  "arguments": {
    "code": "def fib(n): ..."
  }
}

# Step 3: Your code executes the tool
result = python_executor.execute(code="def fib(n): ...")

# Step 4: Send result back to LLM
"Tool result: [0, 1, 1, 2, 3, 5, 8]"

# Step 5: LLM formats final answer
"The Fibonacci sequence up to 10 is: 0, 1, 1, 2, 3, 5, 8"
```

---

## 📁 Project Structure

```
multi_tool_assistant/
│
├── main.py                      # 🚀 Entry point - Run this!
│   ├── Initializes tools
│   ├── Creates agent (OpenAI/Gemini)
│   └── Interactive CLI loop
│
├── config.py                    # ⚙️ Configuration management
│   ├── API keys
│   ├── Model names
│   └── Settings (temperature, max_iterations)
│
├── agents/                      # 🧠 AI Agent implementations
│   ├── __init__.py
│   ├── base_agent.py           # Abstract base class
│   │   ├── Defines agent interface
│   │   ├── Tool execution logic
│   │   └── Conversation history
│   │
│   ├── openai_agent.py         # OpenAI GPT-4 implementation
│   │   ├── Uses OpenAI SDK
│   │   ├── Handles function calling
│   │   └── Manages conversation flow
│   │
│   └── gemini_agent.py         # Google Gemini implementation
│       ├── Uses Google GenAI SDK
│       ├── Handles function calling
│       └── Chat session management
│
├── tools/                       # 🔧 Tool implementations
│   ├── __init__.py
│   ├── base_tool.py            # Abstract tool interface
│   │   ├── Defines tool contract
│   │   ├── to_openai_format()
│   │   └── to_gemini_format()
│   │
│   ├── web_search.py           # 🔍 Web search capability
│   │   ├── DuckDuckGo API integration
│   │   └── Returns search results
│   │
│   ├── weather.py              # 🌤️ Weather information
│   │   ├── OpenWeatherMap API
│   │   └── Current conditions
│   │
│   ├── python_executor.py      # 🐍 Safe code execution
│   │   ├── Restricted namespace
│   │   ├── Output capture
│   │   └── Error handling
│   │
│   └── file_manager.py         # 📄 File operations
│       ├── Create files
│       ├── Read files
│       └── Append to files
│
└── utils/                       # 🛠️ Utility functions
    ├── __init__.py
    └── helpers.py              # Helper functions (future)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- API keys for OpenAI and/or Google Gemini

### Step 1: Clone/Create Project

```bash
mkdir multi_tool_assistant
cd multi_tool_assistant
```

### Step 2: Install Dependencies

```bash
pip install openai google-generativeai requests
```

**Dependency breakdown:**
- `openai` - OpenAI API client
- `google-generativeai` - Google Gemini API client  
- `requests` - HTTP requests for weather/search APIs

### Step 3: Set Environment Variables

**On macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-proj-..."
export GEMINI_API_KEY="AIza..."
export OPENWEATHER_API_KEY="your-key"  # Optional
```

**On Windows:**
```cmd
set OPENAI_API_KEY=sk-proj-...
set GEMINI_API_KEY=AIza...
set OPENWEATHER_API_KEY=your-key
```

**Or create a `.env` file:**
```bash
# .env
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIza...
OPENWEATHER_API_KEY=your-key
```

Then load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 4: Create the Files

Copy all the code sections into their respective files following the project structure above.

---

## 💻 Usage

### Basic Usage

```bash
python main.py
```

**Interactive Menu:**
```
🤖 Multi-Tool Personal Assistant
============================================================
Choose your AI agent:
1. OpenAI (GPT-4)
2. Google Gemini
3. Both (compare responses)

Enter choice (1/2/3): 1
```

### Example Queries

#### 1. Weather Query
```
You: What's the weather in Paris?

🌤️ Fetching weather for: Paris
🔧 Calling tool: get_weather
📥 Arguments: {'location': 'Paris'}
📤 Result: {'temperature': 18, 'condition': 'Partly Cloudy', ...}