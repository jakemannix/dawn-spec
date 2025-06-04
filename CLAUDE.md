# DAWN/AGNTCY Specification Implementation

This document outlines the design goals, repository structure, and development guidelines for implementing and experimenting with Cisco's DAWN (Distributed Agents in a Worldwide Network) and AGNTCY specification for agent interaction.

## Project Overview

This repository contains an experimental implementation of the DAWN architecture and AGNTCY framework, which define protocols and standards for enabling interoperability between AI agents. The implementation focuses on:

1. Agent Connect Protocol (ACP) - REST-based API for agent configuration and invocation
2. Agent Gateway Protocol (AGP) - gRPC-based protocol for efficient real-time agent communication
3. Open Agent Schema Framework (OASF) - Standardized data model for describing agent capabilities
4. DAWN architecture - Implementation of Principal and Gateway Agents for distributed collaboration

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

1. **Phase 1** (Current): Basic agent framework with ACP implementation
   - Agent, Capability, and Message classes
   - Registry for agent discovery (Gateway Agent functionality)
   - REST API server implementation for ACP

2. **Phase 2** (Planned): DAWN Architecture Components
   - Principal Agent implementation with planning capabilities
   - Gateway Agent with full resource registry
   - Context management for maintaining task state
   - Implementation of DAWN's architectural layers

3. **Phase 3** (Planned): AGP implementation
   - gRPC-based communication
   - Support for various message patterns
   - Security features

4. **Phase 4** (Future): Advanced features
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