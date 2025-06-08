# Multi-Protocol LangGraph Research Agents

This demo showcases LangGraph-powered research agents that support multiple protocols (Google A2A, AGNTCY ACP, MCP) with real API integration.

## Features

- **LangGraph ReACT Architecture**: Each agent uses LangGraph's StateGraph for reasoning
- **Multi-Protocol Support**: 
  - Google A2A (Agent-to-Agent) protocol
  - AGNTCY ACP (Agent Connect Protocol)
  - MCP (Model Context Protocol)
- **Real API Integration**: 
  - GitHub API for repository research
  - arXiv API for academic paper research
  - LLM APIs for synthesis and summarization
- **Multi-LLM Support**: Automatically tries OpenAI → Anthropic → Google Gemini
- **Interactive CLI**: Chat with agents using `@agent` mentions

## Prerequisites

1. **Python 3.10+** required
2. **Install dependencies**:
   ```bash
   # Install with all required dependencies
   uv pip install -e ".[langgraph,interop]"
   ```

3. **Configure API Keys** in your `.env` file:
   ```bash
   # At least one LLM API key is required:
   OPENAI_API_KEY=your-openai-key
   ANTHROPIC_API_KEY=your-anthropic-key  
   GOOGLE_API_KEY=your-google-key

   # Optional: GitHub token for higher rate limits
   GITHUB_TOKEN=your-github-token
   ```

## Running the Demo

### Interactive Mode
```bash
uv run python examples/a2a_langgraph_demo.py
```

### Test Mode
```bash
uv run python examples/a2a_langgraph_demo.py --test
```

## Available Agents

### @github - GitHub Research Agent
- **Search**: `@github search for langchain repositories`
- **Analyze**: `@github analyze agntcy/acp-sdk`
- **Fetch Files**: `@github get README from pytorch/pytorch`

### @arxiv - arXiv Research Agent  
- **Search**: `@arxiv find papers on transformer architectures`
- **Fetch**: `@arxiv get details for paper 1706.03762`
- **Summarize**: `@arxiv summarize 1706.03762`

### @synthesis - Synthesis Agent
- **Compare**: `@synthesis compare PyTorch and TensorFlow`
- **Analyze Trends**: `@synthesis what are the trends in ML frameworks?`
- **Synthesize**: Combines results from other agents with full context

## Architecture

```
┌─────────────────────────────────────────────┐
│          Interactive CLI (@mentions)         │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────┴─────────────────────┐
│            LangGraph Agents                │
├─────────────┬─────────────┬───────────────┤
│   GitHub    │    arXiv    │   Synthesis   │
│   Agent     │    Agent    │    Agent      │
├─────────────┼─────────────┼───────────────┤
│ • Search    │ • Search    │ • Synthesize  │
│ • Analyze   │ • Fetch     │ • Compare     │
│ • Fetch     │ • Summarize │ • Trends      │
└─────────────┴─────────────┴───────────────┘
                      │
┌─────────────────────┴─────────────────────┐
│          Protocol Adapters                 │
├─────────────┬─────────────┬───────────────┤
│    A2A      │     ACP     │     MCP       │
│   Server    │  Endpoints  │    Tools      │
└─────────────┴─────────────┴───────────────┘
```

## Agent Cards

Each agent exposes a unified agent card at `/.well-known/agent.json` containing:
- Basic metadata (id, name, description, version)
- Protocol endpoints (A2A, ACP, MCP)
- OASF capability definitions
- Available tools and resources

Type `cards` in the interactive mode to see agent details.

## Example Interactions

```bash
> @github search for popular Python web frameworks
🤖 GitHub agent thinking...

🔍 **GitHub Research Results**

Found 5 repositories matching "popular Python web frameworks":
• django/django (⭐ 70,234) - The Web framework for perfectionists with deadlines
• pallets/flask (⭐ 62,456) - The Python micro framework for building web applications
• encode/django-rest-framework (⭐ 25,678) - Web APIs for Django
• tiangolo/fastapi (⭐ 58,123) - FastAPI framework, high performance, easy to learn
• tornadoweb/tornado (⭐ 21,234) - Python web framework and asynchronous networking library

📊 **Tools used:** github_search

🧠 **Reasoning steps:**
• Processing message: search for popular Python web frameworks
• Calling tool: github_search

⏱️ Response time: 2.34s
```

## Troubleshooting

1. **No LLM API keys**: The demo requires at least one LLM API key. It will try them in order: OpenAI → Anthropic → Google.

2. **Import errors**: Make sure you've installed the required dependencies:
   ```bash
   uv pip install -e ".[langgraph,interop]"
   ```

3. **GitHub rate limits**: Add a `GITHUB_TOKEN` to your `.env` file for higher rate limits.

4. **Protocol warnings**: If you see warnings about A2A/ACP/MCP not being available, the agents will still work but without full protocol support.

## Development

The code is organized as:
- `agents/base_langgraph_agent.py` - Base class with multi-protocol support
- `agents/langgraph_github_agent.py` - GitHub research implementation
- `agents/langgraph_arxiv_agent.py` - arXiv research implementation
- `agents/langgraph_synthesis_agent.py` - Synthesis and analysis
- `a2a_langgraph_demo.py` - Main CLI driver

Each agent:
1. Inherits from `MultiProtocolLangGraphAgent`
2. Implements `get_tools()` to define LangChain tools
3. Implements `get_capabilities()` for OASF definitions
4. Uses LangGraph StateGraph for ReACT reasoning
5. Exposes unified agent cards for all protocols 