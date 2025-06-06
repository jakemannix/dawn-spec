# DAWN/AGNTCY Specification Implementation

This document outlines the design goals, repository structure, and development guidelines for implementing and experimenting with Cisco's DAWN (Distributed Agents in a Worldwide Network) and AGNTCY specification for agent interaction.

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