# DAWN Architecture Implementation

This repository contains a complete implementation of Cisco's "DAWN" (Distributed Agents in a Worldwide Network) specification for agent interaction and interoperability, following the AGNTCY framework protocols.

## Overview

The DAWN architecture provides a framework for building distributed agent systems where specialized agents can cooperate to solve complex tasks. This implementation demonstrates:

- **Principal Agents**: Orchestration agents that decompose tasks and coordinate other agents
- **Gateway Agents**: Agents that enable discovery and registration of other agents
- **Specialized Agents**: Agents with specific capabilities (GitHub search, arXiv access, web search)
- **Standardized Protocols**: Implementation of ACP (Agent Connect Protocol) and AGP (Agent Gateway Protocol)

Key features:

- Interface-based design for modularity and extensibility
- Multiple LLM integrations (OpenAI, Anthropic Claude, Google Gemini)
- Real API integrations (not just mock implementations)
- Capability-based invocation
- Command-line interface (CLI) for easy interaction

## 🚀 NEW: A2A + MCP + OASF Interoperability Prototype

This repository now includes a comprehensive interoperability layer that bridges multiple agent communication protocols:

### Dual Communication Paradigms

#### 🤝 **A2A (Agent-to-Agent)** - Peer-to-Peer Communication
- Agents maintain independent planning and reasoning
- Built with Google's official `a2a-sdk`
- Enables true multi-agent collaboration where both agents are intelligent

#### 🔧 **MCP (Model Context Protocol)** - Centralized Intelligence
- One agent uses others as discoverable tools
- Built with Anthropic's official `mcp` SDK
- Enables tool-based agent composition with centralized planning

#### 📋 **Enhanced OASF** - Strongly-Typed Business Logic
- JSON Schema validation for all payloads
- Pre-built schemas: OrderProcessing, DataAnalysis, DocumentProcessing, ResearchQuery
- Safe payload evolution and comprehensive error reporting

### 🎯 Quick Start - Try the Interoperability Features

**A2A Peer-to-Peer Demo:**
```bash
# Terminal 1: Start text processing agent as A2A server
uv run python examples/a2a_interop_demo.py

# Terminal 2: Run reasoning agent that delegates to the first agent
uv run python examples/a2a_interop_demo.py client
```

**MCP Centralized Intelligence Demo:**
```bash
# Terminal 1: Start analytics agent as MCP server
uv run python examples/mcp_interop_demo.py

# Terminal 2: Run orchestrator that uses analytics as tools
uv run python examples/mcp_interop_demo.py client
```

**Schema Validation:**
```bash
# List available business logic schemas
uv run python -c "from src.schemas import schema_validator; print([s['schema_type'] for s in schema_validator.list_available_schemas()])"

# Test order processing schema validation
uv run python -c "from src.schemas import schema_validator; print(schema_validator.validate_payload('order_processing', {'order_id': 'ORD-ABC12345', 'customer_id': 'CUST-XYZ67890', 'customer_info': {'name': 'Test', 'email': 'test@example.com'}, 'items': [{'product_id': 'PROD-DEF11111', 'product_name': 'Widget', 'quantity': 1, 'unit_price': 10.0, 'total_price': 10.0}], 'total_amount': 10.0, 'currency': 'USD', 'order_status': 'pending'}))"
```

**Create Interop-Capable Agents:**
```bash
# Create an agent that supports both A2A and MCP protocols
uv run python -c "
from src.agent import MCPCapableAgent, Capability
agent = MCPCapableAgent('InteropAgent', 'Supports both A2A and MCP')
cap = Capability('test', 'Test Capability', 'A test capability', business_logic_schema='data_analysis')
agent.add_capability(cap)
print(f'Agent: {agent.name}')
print(f'A2A capable: {hasattr(agent, \"start_a2a_server\")}')
print(f'MCP capable: {hasattr(agent, \"start_mcp_server\")}')
print(f'Schema validation: {cap.business_logic_schema}')
"
```

## Architecture

The DAWN architecture follows this structure:

```
+-------------------------------------+
|        DAWN Architecture           |
+-------------------------------------+

+----------------+        +----------------+
|                |        |                |
| Principal Agent|<------>| Gateway Agent  |
|                |        |                |
+----------------+        +----------------+
        ^                        ^
        |                        |
        v                        v
+----------------+        +----------------+
|                |        |                |
| GitHub Agent   |        | Web Search     |
|                |        | Agent          |
+----------------+        +----------------+
        ^                        ^
        |                        |
        v                        v
+----------------+        +----------------+
|                |        |                |
| arXiv Agent    |        | Other          |
|                |        | Specialized    |
+----------------+        | Agents         |
                          +----------------+
```

The implementation follows these core principles:

1. **Modularity**: All components implement standardized interfaces
2. **Standardization**: Common message formats and protocols
3. **Extensibility**: Easy to add new agent types and capabilities
4. **Interoperability**: Agents can seamlessly work together
5. **Discoverability**: Agents can discover and utilize capabilities of other agents

## Components

### Core Interfaces

- `IAgent`: Base interface for all agents
- `IPrincipalAgent`: Interface for orchestration agents
- `IGatewayAgent`: Interface for gateway agents
- `IMessage`: Interface for standardized message format
- `ITask`: Interface for units of work

### Agent Implementations

- `OpenAI-based agents`: Implementation using OpenAI API (GPT-3.5, GPT-4)
- `Anthropic-based agents`: Implementation using Claude models
- `Gemini-based agents`: Implementation using Google Gemini models

### Protocol Implementations

- `Agent Connect Protocol (ACP)`: For direct agent-to-agent communication
- `Agent Gateway Protocol (AGP)`: For agent registration and discovery

### Demos

- **Research Agent**: Answers questions using GitHub, arXiv, and web search
- **Weather Email**: A simple demo that retrieves weather information and generates an email
- **Anthropic/Gemini/OpenAI demos**: Showcase model-specific capabilities

## Getting Started

### Prerequisites

- Python 3.10+ with pip or [uv](https://github.com/astral-sh/uv) (updated for A2A/MCP compatibility)
- API keys for the services you want to use (OpenAI, Anthropic, Google Gemini)

### Setup with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
# Install uv if you don't have it yet
pip install uv

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies (core + interoperability)
uv sync --extra interop

# Or install only core dependencies
uv sync

# Set up environment variables
cp template.env .env
# Edit .env with your API keys
```

### Alternative Setup with pip

If you prefer using standard pip:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies with interoperability features
pip install -e ".[interop]"

# Or install only core dependencies
pip install -e .
```

### Environment Variables

This project uses dotenv to manage environment variables. To set up:

1. Copy the template environment file:
   ```bash
   cp template.env .env
   ```

2. Edit `.env` with your actual API keys:
   ```
   # OpenAI API Configuration
   OPENAI_API_KEY=sk-your-actual-key
   
   # Anthropic API Configuration
   ANTHROPIC_API_KEY=sk-ant-your-actual-key
   
   # Gemini API Configuration
   GEMINI_API_KEY=AIza-your-actual-key
   
   # GitHub API Configuration (optional, but recommended)
   GITHUB_TOKEN=ghp_your-actual-token
   ```

3. Check your configuration:
   ```bash
   python -m src.config
   ```

Note: The `.env` file is gitignored to prevent accidental commits of sensitive information.

## Running Demos

### Command Line Interface (CLI)

The project includes a comprehensive CLI for interacting with the DAWN implementation:

```bash
# Show help
python dawn_cli.py --help

# Show configuration
python dawn_cli.py config

# Show agent information
python dawn_cli.py agent-info anthropic

# Invoke an agent capability
python dawn_cli.py invoke anthropic text-generation --prompt "Write a poem about AI agents"

# Run a demo
python dawn_cli.py run-demo research
```

### Available Demos

#### Weather Email Demo (No API keys required)

A simple demo using mock implementations:

```bash
python examples/weather_email_demo.py
# or
python dawn_cli.py run-demo weather-email
```

#### Research Agent Demo (Requires API keys)

A complete DAWN architecture with GitHub, arXiv, and web search:

```bash
python examples/research_agent_demo.py
# or
python dawn_cli.py run-demo research
```

Try these sample queries:
- "What are the key features of the AGNTCY framework?"
- "Find recent papers about multimodal LLMs and summarize them"
- "Compare React and ReAct frameworks for agents"

#### LLM-specific Agent Demos

Each demo showcases a specific LLM implementation:

```bash
# OpenAI agent demo (requires OpenAI API key)
python examples/openai_agent_demo.py
# or
python dawn_cli.py run-demo openai

# Anthropic agent demo (requires Anthropic API key)
python examples/anthropic_agent_demo.py
# or
python dawn_cli.py run-demo anthropic

# Gemini agent demo (requires Gemini API key)
python examples/gemini_agent_demo.py
# or
python dawn_cli.py run-demo gemini
```

#### AGP Gateway Demo

Demonstrates agent registration and discovery:

```bash
python examples/agp_gateway_demo.py
# or
python dawn_cli.py run-demo agp-gateway
```

### Minimal Testing with Limited API Quota

If you have limited API quota (especially for Gemini), you can run minimal tests:

```bash
# Test Gemini with minimal token usage
python test_gemini.py

# Run OpenAI diagnostics mode
python examples/openai_agent_demo.py diagnostics

# Run Anthropic demo with limited tokens
python dawn_cli.py invoke anthropic text-generation --prompt "Hello" --config '{"max_tokens": 10}'
```

## Implementation Details

### Agent Structure

Each agent implementation follows this pattern:

1. **Initialization**: Set up the agent with model parameters and capabilities
2. **Capability Definition**: Define what the agent can do
3. **Capability Invocation**: Logic to invoke specific capabilities
4. **Response Handling**: Processing the model's responses

### Project Organization

- `src/`: Core implementation
  - `interfaces.py`: Core interface definitions
  - `agent.py`: Base agent implementation
  - `openai_agent.py`, `anthropic_agent.py`, `gemini_agent.py`: Model-specific implementations
  - `agp_gateway.py`: Gateway agent implementation
  - `config.py`: Configuration management
  - `registry.py`: Agent registry for discovery

- `examples/`: Demonstration scripts
- `tests/`: Unit tests
- `proto/`: Protocol definitions in Protocol Buffers
- `docs/`: Additional documentation

## Testing

See the [Testing Guide](docs/testing_guide.md) for detailed instructions on testing the implementation with varying levels of API access.

### Minimal Testing with Limited API Quota

If you have limited API quota (especially for Gemini), you can run minimal tests:

```bash
# Test with no API keys required
python examples/weather_email_demo.py

# Test Gemini with minimal token usage
python test_gemini.py

# Run OpenAI diagnostics mode
python examples/openai_agent_demo.py diagnostics

# Run Anthropic demo with limited tokens
python dawn_cli.py invoke anthropic text-generation --prompt "Hello" --config '{"max_tokens": 10}'
```

## Troubleshooting

### Gemini API Quota Issues

Gemini has strict quota limits for free tier users. If you encounter quota errors:

1. Set a low `max_tokens` value (e.g., 50) when invoking Gemini agents
2. Implement exponential backoff by waiting before retrying (the implementation includes automatic retry logic)
3. Consider using a different model as your primary agent (`PRINCIPAL_AGENT_TYPE=openai` in .env)

### OpenAI API Health Check Fails

If the OpenAI health check fails:

1. Verify your API key is correct in `.env`
2. Check your API subscription status
3. Run the diagnostics mode: `python examples/openai_agent_demo.py diagnostics`
4. Try a simple text completion to verify access: `python dawn_cli.py invoke openai text-generation --prompt "Hello"`

### Anthropic API Issues

For Anthropic API issues:

1. Ensure you're using the correct API key format (starting with `sk-ant-`)
2. Verify your API key hasn't expired
3. Check your quota usage in the Anthropic console

### Protocol Buffer Version Conflicts

If you encounter protocol buffer errors:

1. Verify you're using protobuf 3.19.0 or higher
2. If using Streamlit alongside this project, upgrade to Streamlit 1.45.1+ and protobuf 5.29.4+
3. Regenerate the protocol buffer code: `python generate_protos.py`

## Advanced Usage

### Creating Custom Agents

To create a new agent:

1. Implement the `IAgent` interface
2. Define the agent's capabilities
3. Implement the invocation logic
4. Register the agent with the registry

Example:

```python
from src.interfaces import IAgent

class MyCustomAgent(IAgent):
    def __init__(self, name, description):
        self._id = f"custom-agent-{uuid.uuid4()}"
        self._name = name
        self._description = description
        self._capabilities = [
            {
                "id": "my-capability",
                "name": "My Custom Capability",
                "description": "Does something awesome"
            }
        ]

    def get_info(self):
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }

    def get_capabilities(self):
        return self._capabilities

    def invoke(self, capability_id, inputs, config=None):
        if capability_id == "my-capability":
            # Implement the capability
            return {"result": "Something awesome"}
        return {"error": "Unknown capability"}
```

### Extending Protocols

To extend the protocols:

1. Modify the `.proto` files in the `proto/` directory
2. Regenerate the protocol code: `python generate_protos.py`
3. Update the agent implementations to use the new protocol features

## Comparison with Other Agent Frameworks

This implementation differs from other agent frameworks in several ways:

- **Focus on Interoperability**: Built around standardized interfaces and protocols
- **Multiple Model Support**: Works with OpenAI, Anthropic, and Google models
- **Real API Integrations**: Uses actual APIs rather than simulations
- **Standardized Protocols**: Implements ACP and AGP specifications
- **Lightweight**: Minimal dependencies and focused implementation

## References and Resources

- [DAWN Paper](https://arxiv.org/html/2410.22339v2)
- [AGNTCY GitHub Organization](https://github.com/agntcy)
- [Agent Connect Protocol Specification](https://github.com/agntcy/acp-spec)
- [Agent Gateway Protocol Specification](https://github.com/agntcy/agp-spec)

## Project Documentation

For more detailed information about the project:

- See `IMPLEMENTATION_SUMMARY.md` for a technical overview
- Check `docs/design.md` for architectural details
- Read `docs/demo_plan.md` for the demo implementation plan

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.