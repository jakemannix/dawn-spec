# DAWN-Spec: LangGraph Multi-Agent Development Guide

This document provides comprehensive development guidelines for the **LangGraph multi-agent research system** with multi-protocol support and real API integrations.

## 🚀 LangGraph Multi-Agent System Overview

### Interactive Research CLI
- **Natural language interface** with specialized agents: `@github`, `@arxiv`, synthesis
- **Real LLM integration** using LangGraph ReACT patterns with multi-provider fallback
- **Context-aware conversations** with reasoning trace capture
- **Multi-protocol support**: A2A, ACP, MCP, OASF compatible agents

### Key Implementation Features
- **Real API integrations**: GitHub API, arXiv API (not mocks)
- **LangGraph StateGraph**: Proper ReACT reasoning loops with tool integration
- **Multi-LLM fallback**: OpenAI → Anthropic → Google Gemini
- **Protocol adapters**: Unified agent cards for cross-protocol discovery
- **Context management**: Intelligent context passing between agents

## 🏗️ Development Environment Setup

### Virtual Environment & Dependencies
```bash
# Always ensure .venv virtual environment is active
uv venv && source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate              # Windows

# Install LangGraph agent dependencies
uv sync --extra langgraph

# For all protocol integrations
uv sync --extra interop
```

### Environment Configuration
```bash
# Copy and configure environment
cp template.env .env

# Required for LangGraph agents (at least one):
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key  
GEMINI_API_KEY=AIza-your-gemini-key

# Optional for enhanced GitHub API limits
GITHUB_TOKEN=ghp_your-github-token

# Optional LLM configuration overrides
DEFAULT_MODEL=gpt-4o-mini        # Override default model
TEMPERATURE=0.7                  # Override default temperature
MAX_TOKENS=2000                  # Override default max tokens
```

### Running Commands
**Always use `uv run python ...` when running Python scripts/tests**

```bash
# Start interactive LangGraph research system
uv run python examples/a2a_langgraph_demo.py

# Run individual agent tests
uv run python -c "
from examples.agents import LangGraphGitHubAgent
import asyncio
async def test():
    agent = LangGraphGitHubAgent('test')
    result = await agent.process_request('search for python ML libraries')
    print(result.response_text)
asyncio.run(test())
"

# Run protocol integration demos
uv run python examples/a2a_interop_demo.py
uv run python examples/mcp_interop_demo.py

# Run tests
uv run python -m pytest tests/
```

## 🏛️ LangGraph Agent Architecture

### Three-Layer Framework
1. **🔧 MCP Tools**: LLM-integrated tools for local interaction
2. **🏷️ OASF Capabilities**: Schema validation and semantic tagging  
3. **📡 A2A Skills**: Network-callable functions for distributed systems

### Agent Structure
```python
# Base multi-protocol LangGraph agent
class MultiProtocolLangGraphAgent:
    """Base class for LangGraph agents with multi-protocol support"""
    
    def __init__(self, agent_id: str, name: str):
        self.llm = self._initialize_llm()      # Multi-LLM fallback
        self.graph = self._create_langgraph()  # StateGraph with ReACT
        self.protocols = {                     # Protocol adapters
            'a2a': A2AAdapter(self),
            'acp': ACPAdapter(self), 
            'mcp': MCPAdapter(self),
            'oasf': OASFAdapter(self)
        }
    
    def _create_langgraph(self) -> StateGraph:
        """Create LangGraph with ReACT pattern"""
        # Agent node: LLM reasoning
        # Tool node: Tool execution
        # Conditional routing based on agent decisions
    
    async def process_request(self, query: str) -> AgentResponse:
        """Process request through LangGraph reasoning"""
        # Returns response with reasoning_trace for debugging
```

### Specialized Agent Implementations
- **`LangGraphGitHubAgent`**: Repository search, file analysis, codebase exploration
- **`LangGraphArxivAgent`**: Academic paper search, research trend analysis
- **`LangGraphSynthesisAgent`**: Cross-source research synthesis and insights

### Protocol Integration
```python
# Each agent supports multiple discovery protocols
agent = LangGraphGitHubAgent("github-research")

# A2A peer-to-peer communication
await agent.start_a2a_server(port=8001)

# ACP REST endpoint serving  
await agent.start_acp_server(port=8002)

# MCP tool-based composition
await agent.start_mcp_server()

# Unified agent card for discovery
agent_card = agent.generate_agent_card()  # Compatible with all protocols
```

## 🛠️ Development Guidelines

### Dependency Management
- **Add dependencies to `pyproject.toml`**, not separate requirements files
- Use optional dependency groups:
  - `langgraph`: LangGraph agent implementation
  - `interop`: A2A, MCP, OASF interoperability  
  - `dev`: Development tools (pytest, black, etc.)

### Code Structure
```
examples/
├── a2a_langgraph_demo.py          # Main interactive CLI
├── agents/                        # LangGraph agent implementations
│   ├── base_langgraph_agent.py    # Base multi-protocol agent
│   ├── langgraph_github_agent.py  # GitHub research agent
│   ├── langgraph_arxiv_agent.py   # arXiv research agent
│   └── langgraph_synthesis_agent.py # Research synthesis agent
├── protocols/                     # Protocol adapter implementations
│   ├── a2a_adapter.py            # A2A protocol integration
│   ├── acp_adapter.py            # ACP protocol integration
│   └── mcp_adapter.py            # MCP protocol integration
└── README_LANGGRAPH.md           # LangGraph-specific documentation
```

### Import Guidelines
- **Use absolute imports**: `from src.module import Class`
- **Add `src/` to path** for examples: `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))`
- **Import LangGraph agents**: `from examples.agents import LangGraphGitHubAgent`

### Code Style & Testing
```bash
# Format code
uv run ruff format .
uv run black .

# Type checking and linting
uv run ruff check .

# Run tests with proper imports
uv run python -m pytest tests/
uv run python -m pytest tests/test_langgraph_agents.py -v
```

### Error Handling & Logging
```python
# Multi-LLM fallback with graceful degradation
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    await llm.ainvoke([HumanMessage(content="test")])
except Exception as e:
    try:
        llm = ChatAnthropic(model="claude-3-haiku-20240307")
        await llm.ainvoke([HumanMessage(content="test")])
    except Exception as e2:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Comprehensive API error handling
try:
    result = await github_api.search_repositories(query)
except GithubException as e:
    if e.status == 403:  # Rate limit
        return "GitHub API rate limit reached. Please try again later."
    elif e.status == 401:  # Auth error
        return "GitHub API authentication failed. Check GITHUB_TOKEN."
    else:
        return f"GitHub API error: {e.data.get('message', str(e))}"
```

## 🔧 Agent Development Patterns

### LangGraph ReACT Implementation
```python
def _create_langgraph(self) -> StateGraph:
    """Create ReACT pattern LangGraph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", self._agent_node)      # LLM reasoning
    workflow.add_node("tools", self._tool_node)       # Tool execution
    
    # Define routing logic
    workflow.add_conditional_edges(
        "agent",
        self._should_continue,
        {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile()

async def _agent_node(self, state: AgentState) -> Dict[str, Any]:
    """LLM reasoning with tool availability"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", self.system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_tools_agent(self.llm, self.tools, prompt)
    response = await agent.ainvoke(state)
    return {"messages": [response]}
```

### Tool Implementation
```python
class GitHubSearchTool(BaseTool):
    """LangChain-compatible GitHub search tool"""
    name: str = "github_search"
    description: str = "Search GitHub repositories"
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__()
        self.github = Github(github_token) if github_token else Github()
    
    def _run(self, query: str, language: str = "", sort: str = "stars") -> str:
        """Synchronous tool execution"""
        return asyncio.run(self._arun(query, language, sort))
    
    async def _arun(self, query: str, language: str = "", sort: str = "stars") -> str:
        """Asynchronous tool execution with proper error handling"""
        try:
            repos = self.github.search_repositories(
                query=f"{query} language:{language}" if language else query,
                sort=sort,
                order="desc"
            )
            # Process and return results
        except GithubException as e:
            return f"GitHub search error: {e}"
```

### Context Management
```python
class ContextManager:
    """Manages conversation context between agents"""
    
    def __init__(self):
        self.full_history = []      # Complete conversation
        self.agent_contexts = {}    # Per-agent filtered context
    
    def add_message(self, message: Dict[str, Any]):
        """Add message to full history"""
        self.full_history.append(message)
    
    def get_context_for_agent(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get filtered context for specific agent"""
        if agent_type == "synthesis":
            return self.full_history  # Synthesis gets full context
        else:
            # Research agents get filtered context
            return [msg for msg in self.full_history 
                   if self._is_relevant_for_agent(msg, agent_type)]
```

---

### DAWN/AGNTCY Specification Implementation

This section outlines the original design goals and technical architecture for implementing Cisco's DAWN (Distributed Agents in a Worldwide Network) and AGNTCY specification.

#### Legacy Architecture Overview
The original project implemented a **three-layer agent architecture**:

1. **A2A Skills (Wire Protocol)**: External functions callable over network (`invoice.pay`)
2. **OASF Capabilities (Schema/Validation)**: Semantic tags + JSON schemas (`finance.payment`)  
3. **MCP Tools (LLM Integration)**: JSON objects for local LLM interaction

#### Pluggable Agent System
- `src/agent_core.py`: Core orchestration and abstractions
- `src/implementations/`: Different agent backends (text matching, LangGraph, etc.)  
- `src/agent_factory.py`: Factory for creating and configuring agents
- `config/agent_config.yaml`: Configuration for different implementations

## Project Overview

This repository contains an experimental implementation of the DAWN architecture and AGNTCY framework, which define protocols and standards for enabling interoperability between AI agents. The implementation focuses on:

1. Agent Connect Protocol (ACP) - REST-based API for agent configuration and invocation
2. Agent Gateway Protocol (AGP) - gRPC-based protocol for efficient real-time agent communication
3. Open Agent Schema Framework (OASF) - Standardized data model for describing agent capabilities
4. DAWN architecture - Implementation of Principal and Gateway Agents for distributed collaboration

## NEW: A2A + MCP + OASF Interoperability (Phase 1)

As of the latest release, this repository includes a comprehensive interoperability prototype that integrates:

1. **A2A (Agent-to-Agent) Protocol** - Google's official A2A SDK for peer-to-peer agent communication
2. **MCP (Model Context Protocol)** - Anthropic's official MCP SDK for centralized tool-based agent interactions  
3. **Enhanced OASF** - Strongly-typed business logic schemas with JSON Schema validation

### Dual Communication Paradigms

The implementation supports two distinct multi-agent interaction patterns:

#### 1. Peer-to-Peer (A2A Protocol)
- Both agents maintain independent planning and reasoning
- Agents delegate tasks to each other while preserving autonomy
- Uses Google's official `a2a-sdk` for standardized communication

#### 2. Centralized Intelligence (MCP Protocol)  
- One agent uses another's capabilities as discoverable tools
- Planning remains centralized while execution is distributed
- Uses Anthropic's official `mcp` SDK for tool discovery and invocation

### Quick Start - Interoperability Features

**Test A2A peer-to-peer communication:**
```bash
# Terminal 1: Start A2A server
uv run python examples/a2a_interop_demo.py

# Terminal 2: Run A2A client workflow
uv run python examples/a2a_interop_demo.py client
```

**Test MCP centralized intelligence:**
```bash
# Terminal 1: Start MCP server
uv run python examples/mcp_interop_demo.py

# Terminal 2: Run MCP client workflow  
uv run python examples/mcp_interop_demo.py client
```

**Test schema validation:**
```bash
uv run python -c "from src.schemas import schema_validator; print([s['schema_type'] for s in schema_validator.list_available_schemas()])"
```

## Core Interfaces

The following interfaces define the key components of our implementation:

### 1. Agent Interface

```python
class IAgent:
    """Interface that all agents must implement."""
    
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        pass
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        pass
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs and configuration."""
        pass
        
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        pass
```

### 2. Principal Agent Interface

```python
class IPrincipalAgent(IAgent):
    """Interface for the orchestration agent in DAWN architecture."""
    
    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down a complex task into subtasks."""
        pass
        
    def discover_agents(self, capability_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find agents with capabilities matching requirements."""
        pass
        
    def create_execution_plan(self, subtasks: List[Dict[str, Any]], available_agents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Create a plan for executing subtasks with selected agents."""
        pass
        
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the plan and return aggregated results."""
        pass
        
    def handle_error(self, error: Dict[str, Any], execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle errors during plan execution."""
        pass
```

### 3. Gateway Agent Interface

```python
class IGatewayAgent(IAgent):
    """Interface for registry/gateway agent in DAWN architecture."""
    
    def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """Register an agent in the registry."""
        pass
        
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        pass
        
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        pass
        
    def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List agents matching optional filters."""
        pass
        
    def find_agents_by_capability(self, capability_type: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find agents with a specific capability type and parameters."""
        pass
        
    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Validate an agent's capabilities and accessibility."""
        pass
```

### 4. Message Interface

```python
class IMessage:
    """Interface for standardized message format."""
    
    def get_id(self) -> str:
        """Get unique message identifier."""
        pass
        
    def get_sender_id(self) -> str:
        """Get sender agent identifier."""
        pass
        
    def get_recipient_id(self) -> str:
        """Get recipient agent identifier."""
        pass
        
    def get_content(self) -> Any:
        """Get message content."""
        pass
        
    def get_conversation_id(self) -> str:
        """Get conversation identifier for related messages."""
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get message metadata."""
        pass
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        pass
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IMessage':
        """Create message from dictionary representation."""
        pass
```

### 5. Task Interface

```python
class ITask:
    """Interface for defining units of work."""
    
    def get_id(self) -> str:
        """Get unique task identifier."""
        pass
        
    def get_description(self) -> str:
        """Get task description."""
        pass
        
    def get_inputs(self) -> Dict[str, Any]:
        """Get task inputs."""
        pass
        
    def get_output_schema(self) -> Dict[str, Any]:
        """Get schema for expected task output."""
        pass
        
    def get_required_capabilities(self) -> List[Dict[str, Any]]:
        """Get capabilities required for task execution."""
        pass
        
    def get_dependencies(self) -> List[str]:
        """Get IDs of dependent tasks."""
        pass
        
    def get_status(self) -> str:
        """Get task status."""
        pass
        
    def set_status(self, status: str) -> None:
        """Update task status."""
        pass
        
    def set_result(self, result: Dict[str, Any]) -> None:
        """Set task result."""
        pass
        
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get task result."""
        pass
```

## Repository Structure

```
dawn-spec/
├── docs/               # Documentation files
│   └── design.md       # Design document outlining AGNTCY architecture
├── src/                # Source code
│   ├── agent.py        # Agent and Capability implementation
│   ├── registry.py     # Agent registry for discovery
│   └── acp_server.py   # HTTP server for ACP implementation
├── examples/           # Example usage
│   ├── basic_agent_interaction.py       # Basic example
│   └── acp_agent_example.py             # ACP server example
├── tests/              # Test files
├── README.md           # Project overview
└── CLAUDE.md           # This file - design goals and guidelines
```

## Design Goals

1. **Interoperability**: Create agents that can communicate with each other regardless of implementation details.
2. **Standards Compliance**: Follow the AGNTCY specifications for ACP and AGP protocols.
3. **Modularity**: Design components that can be used independently or together.
4. **Extensibility**: Make it easy to extend the implementation for specific use cases.
5. **Security**: Implement proper authentication, authorization, and encryption.

## Implementation Phases

1. **Phase 1** (Completed): Basic agent framework with ACP implementation
   - Agent, Capability, and Message classes
   - Registry for agent discovery (Gateway Agent functionality)
   - REST API server implementation for ACP

2. **Phase 1.5** (Completed): A2A + MCP + OASF Interoperability Prototype
   - **A2A Integration**: Google's official `a2a-sdk` for peer-to-peer agent communication
   - **MCP Integration**: Anthropic's official `mcp` SDK for centralized tool-based interactions
   - **Enhanced OASF**: Strongly-typed business logic schemas with JSON validation
   - **Dual Paradigms**: Support for both peer-to-peer and centralized intelligence patterns
   - **Production-Ready**: Built on official SDKs with comprehensive demos and testing

3. **Phase 2** (Planned): DAWN Architecture Components
   - Principal Agent implementation with planning capabilities
   - Gateway Agent with full resource registry
   - Context management for maintaining task state
   - Implementation of DAWN's architectural layers

4. **Phase 3** (Planned): AGP implementation
   - gRPC-based communication
   - Support for various message patterns
   - Security features

5. **Phase 4** (Future): Advanced features
   - Agent orchestration with ReAct-based reasoning
   - Complex workflows across distributed agents
   - Security, safety, and compliance mechanisms
   - Integration with other AI agent ecosystems

## Development Guidelines

- **Python Code Style**: Follow PEP 8 style guide
- **Documentation**: Document all classes and functions with docstrings
- **Type Annotations**: Use type hints for all function parameters and return values
- **Testing**: Write unit tests for all components
- **Error Handling**: Implement proper error handling and validation

## References

- [AGNTCY GitHub Organization](https://github.com/agntcy)
- [Agent Connect Protocol Specification](https://github.com/agntcy/acp-spec)
- [Agent Gateway Protocol Specification](https://github.com/agntcy/agp-spec)
- [Agent Gateway Protocol Implementation](https://github.com/agntcy/agp)
- [Cisco DAWN Initiative](https://outshift.cisco.com/blog/agntcy-internet-of-agents-is-on-github)