# Dawn Spec Codebase Refactoring Plan

## Overview

This document outlines a comprehensive plan to refactor the Dawn Spec codebase to achieve better separation of concerns, cleaner architecture, and more maintainable code structure.

## Goals

1. **Move core functionality from `examples/` to `src/`**
2. **Make LangGraph a core dependency**
3. **Implement dependency injection for protocol adapters**
4. **Create separate builds for different protocols (A2A, ACP, MCP)**
5. **Establish a demo runner base class for protocol demonstrations**
6. **Simplify examples to actual usage demonstrations**
7. **Add comprehensive structured logging with different verbosity levels**

## Current Issues

- **Too much implementation in examples**: The `examples/agents/` directory contains full agent implementations that should be in the library
- **Conditional imports everywhere**: Protocol availability checks scattered throughout the code
- **No clear separation**: Core agent functionality mixed with protocol-specific code
- **Duplicate implementations**: Multiple agent implementations across different files
- **Complex examples**: Demo files are too complex and contain implementation details
- **Limited observability**: No structured logging for debugging agent interactions

## Proposed Architecture

### Directory Structure

```
dawn-spec/
├── src/
│   └── dawn/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent.py              # Base agent classes
│       │   ├── langgraph_agent.py    # LangGraph base implementation
│       │   ├── state.py              # Agent state management
│       │   ├── tools.py              # Tool interfaces
│       │   └── capabilities.py       # Capability definitions
│       │
│       ├── protocols/
│       │   ├── __init__.py
│       │   ├── base.py               # Protocol adapter interface
│       │   ├── a2a/
│       │   │   ├── __init__.py
│       │   │   ├── adapter.py        # A2A protocol adapter
│       │   │   └── server.py         # A2A server implementation
│       │   ├── acp/
│       │   │   ├── __init__.py
│       │   │   ├── adapter.py        # ACP protocol adapter
│       │   │   └── client.py         # ACP client implementation
│       │   └── mcp/
│       │       ├── __init__.py
│       │       ├── adapter.py        # MCP protocol adapter
│       │       └── session.py        # MCP session management
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── github.py             # GitHub research agent
│       │   ├── arxiv.py              # arXiv research agent
│       │   ├── synthesis.py          # Synthesis agent
│       │   └── registry.py           # Agent registry
│       │
│       ├── runners/
│       │   ├── __init__.py
│       │   ├── base.py               # Base demo runner
│       │   ├── a2a_runner.py         # A2A-specific runner
│       │   ├── acp_runner.py         # ACP-specific runner
│       │   ├── mcp_runner.py         # MCP-specific runner
│       │   └── multi_runner.py       # Multi-protocol runner
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py            # Structured logging system
│           └── config.py
│
├── examples/
│   ├── basic/
│   │   ├── simple_agent.py           # Minimal agent example
│   │   ├── tool_usage.py             # Tool usage example
│   │   └── multi_agent.py            # Multi-agent example
│   │
│   ├── protocols/
│   │   ├── a2a_demo.py               # A2A protocol demo
│   │   ├── acp_demo.py               # ACP protocol demo
│   │   ├── mcp_demo.py               # MCP protocol demo
│   │   └── multi_protocol_demo.py    # All protocols demo
│   │
│   └── advanced/
│       ├── custom_agent.py           # Custom agent implementation
│       ├── protocol_adapter.py       # Custom protocol adapter
│       └── logging_demo.py           # Logging system demo
│
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

## Implementation Plan

### Phase 1: Core Library Structure (Week 1)

1. **Create base package structure**
   ```python
   # src/dawn/__init__.py
   from dawn.core import Agent, LangGraphAgent
   from dawn.protocols import ProtocolAdapter
   from dawn.agents import registry
   from dawn.utils.logging import get_logger
   
   __all__ = ['Agent', 'LangGraphAgent', 'ProtocolAdapter', 'registry', 'get_logger']
   ```

2. **Move and refactor base agent classes**
   - Move `examples/agents/base_langgraph_agent.py` → `src/dawn/core/langgraph_agent.py`
   - Extract interfaces to `src/dawn/core/agent.py`
   - Create protocol adapter interface

3. **Implement dependency injection pattern**
   ```python
   # src/dawn/protocols/base.py
   class ProtocolAdapter(ABC):
       @abstractmethod
       async def start(self, agent: Agent, port: int) -> None:
           """Start protocol server/client"""
       
       @abstractmethod
       async def stop(self) -> None:
           """Stop protocol server/client"""
       
       @abstractmethod
       def get_endpoints(self) -> Dict[str, str]:
           """Get protocol endpoints"""
   ```

4. **Implement structured logging system**
   ```python
   # src/dawn/utils/logging.py
   class StructuredLogger:
       def message_received(self, message: str, source: str, metadata: Dict = None)
       def message_sent(self, message: str, destination: str, metadata: Dict = None)
       def tool_call(self, tool_name: str, arguments: Dict[str, Any])
       def tool_response(self, tool_name: str, response: Any, duration_ms: float)
       def llm_request(self, model: str, messages: list, temperature: float)
       def llm_response(self, model: str, response: str, tokens_used: Dict)
   ```

### Phase 2: Protocol Adapters (Week 2)

1. **Create protocol adapter implementations**
   ```python
   # src/dawn/protocols/acp/adapter.py
   class ACPAdapter(ProtocolAdapter):
       def __init__(self, logger: StructuredLogger = None):
           self.app = None
           self.runner = None
           self.logger = logger or get_logger(__name__)
           
       async def start(self, agent: Agent, port: int) -> None:
           self.logger.protocol_start("acp", port)
           self.app = await self._create_app(agent)
           self.runner = web.AppRunner(self.app)
           await self.runner.setup()
           site = web.TCPSite(self.runner, 'localhost', port)
           await site.start()
   ```

2. **Update pyproject.toml with protocol extras**
   ```toml
   [project]
   dependencies = [
       # Core dependencies (always required)
       "langgraph>=0.0.40",
       "langchain-core>=0.1.45",
       "aiohttp>=3.8.0",
       # ... other core deps
   ]
   
   [project.optional-dependencies]
   a2a = ["a2a-sdk"]
   acp = ["agntcy-acp>=1.5.0"]
   mcp = ["mcp[cli]"]
   all = ["dawn-spec[a2a,acp,mcp]"]
   ```

### Phase 3: Agent Implementations (Week 2-3)

1. **Move agent implementations with logging**
   ```python
   # src/dawn/agents/github.py
   from dawn.utils.logging import get_logger, log_tool_call
   
   class GitHubAgent(LangGraphAgent):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.logger = get_logger(f"agent.{self.agent_id}")
       
       @log_tool_call(self.logger)
       async def github_search(self, query: str) -> Dict[str, Any]:
           # Implementation with automatic logging
           pass
   ```

2. **Create agent registry**
   ```python
   # src/dawn/agents/registry.py
   class AgentRegistry:
       _agents: Dict[str, Type[Agent]] = {}
       
       @classmethod
       def register(cls, name: str):
           def decorator(agent_class: Type[Agent]):
               cls._agents[name] = agent_class
               return agent_class
           return decorator
       
       @classmethod
       def create(cls, name: str, **kwargs) -> Agent:
           if name not in cls._agents:
               raise ValueError(f"Unknown agent: {name}")
           return cls._agents[name](**kwargs)
   ```

### Phase 4: Demo Runners (Week 3)

1. **Create base demo runner with logging**
   ```python
   # src/dawn/runners/base.py
   class BaseDemoRunner:
       def __init__(self, agents: List[str], protocols: List[str], log_level: str = "INFO"):
           self.agents = agents
           self.protocols = protocols
           self.running_agents = {}
           self.logger = get_logger(__name__)
           self.logger.set_level(log_level)
           
       async def start(self):
           """Start all agents with specified protocols"""
           for agent_name in self.agents:
               agent = registry.create(agent_name)
               
               # Set up agent context for logging
               with AgentContext(agent.agent_id):
                   self.logger.agent_start(agent.agent_id, agent_name, self.protocols)
                   
               adapters = []
               
               for protocol in self.protocols:
                   adapter = self._create_adapter(protocol)
                   await adapter.start(agent, self._get_port())
                   adapters.append(adapter)
               
               self.running_agents[agent_name] = (agent, adapters)
       
       async def stop(self):
           """Stop all agents and protocols"""
           for agent, adapters in self.running_agents.values():
               with AgentContext(agent.agent_id):
                   self.logger.agent_stop(agent.agent_id)
                   
               for adapter in adapters:
                   await adapter.stop()
   ```

2. **Create protocol-specific runners**
   ```python
   # src/dawn/runners/acp_runner.py
   class ACPDemoRunner(BaseDemoRunner):
       def __init__(self, agents: List[str], log_level: str = "INFO"):
           super().__init__(agents, ['acp'], log_level)
           
       def _create_adapter(self, protocol: str):
           return ACPAdapter(logger=self.logger)
   ```

### Phase 5: Simplified Examples (Week 4)

1. **Create minimal usage examples**
   ```python
   # examples/basic/simple_agent.py
   import asyncio
   from dawn import registry
   from dawn.runners import ACPDemoRunner
   
   async def main():
       # Create and start demo with debug logging
       runner = ACPDemoRunner(['github', 'arxiv'], log_level="DEBUG")
       await runner.start()
       
       # Run interactive chat
       await runner.chat_loop()
       
       # Cleanup
       await runner.stop()
   
   if __name__ == "__main__":
       # Enable JSON logging for production
       import os
       os.environ['DAWN_LOG_FORMAT'] = 'json'
       
       asyncio.run(main())
   ```

2. **Create multi-protocol example**
   ```python
   # examples/protocols/multi_protocol_demo.py
   from dawn.runners import MultiProtocolRunner
   
   async def main():
       runner = MultiProtocolRunner(
           agents=['github', 'arxiv', 'synthesis'],
           protocols=['a2a', 'acp', 'mcp'],
           log_level="TRACE"  # Maximum verbosity
       )
       await runner.start()
       await runner.demo()
       await runner.stop()
   ```

## Logging System Features

### Log Levels

- **CRITICAL (50)**: System failures
- **ERROR (40)**: Operation failures
- **WARNING (30)**: Warning conditions
- **INFO (20)**: General information (default)
- **DEBUG (10)**: Detailed information, including messages and tool calls
- **TRACE (5)**: Very detailed, including LLM requests/responses

### Environment Variables

```bash
# Set log level
export DAWN_LOG_LEVEL=DEBUG

# Set output format
export DAWN_LOG_FORMAT=json  # or 'human'
```

### Structured Events

All events are logged as structured JSON with:
- Timestamp
- Event type (agent.start, message.received, tool.call, etc.)
- Request ID (for tracing across services)
- Agent ID (for multi-agent scenarios)
- Event-specific data

### Automatic Tool Logging

```python
@log_tool_call(logger)
async def my_tool(arg1, arg2):
    # Automatically logs:
    # - tool.call with arguments
    # - tool.response with result and duration
    # - tool.error if exception occurs
    return result
```

## Migration Steps

### Step 1: Create New Structure
```bash
# Create directory structure
mkdir -p src/dawn/{core,protocols/{a2a,acp,mcp},agents,runners,utils}
touch src/dawn/{__init__.py,core/__init__.py,protocols/__init__.py}
```

### Step 2: Update Imports
Create an import compatibility layer during migration:
```python
# examples/agents/__init__.py (temporary)
import warnings
warnings.warn(
    "Importing from examples.agents is deprecated. "
    "Use dawn.agents instead.",
    DeprecationWarning
)
from dawn.agents import *
```

### Step 3: Update Dependencies
```bash
# Update pyproject.toml and sync
uv sync --all-extras
```

### Step 4: Run Tests
```bash
# Ensure all tests pass after each migration step
pytest tests/
```

### Step 5: Update Documentation
- Update README.md with new structure
- Update CLAUDE.md with architectural changes
- Create migration guide for existing users
- Add logging configuration guide

## Benefits

1. **Clear Separation**: Core functionality in library, demos in examples
2. **Flexible Deployment**: Choose which protocols to include
3. **Dependency Injection**: Easy to add new protocols
4. **Reusable Components**: Demo runners can be imported and customized
5. **Maintainable**: Clear boundaries between components
6. **Testable**: Each component can be tested in isolation
7. **Observable**: Comprehensive logging for debugging and monitoring
8. **Production-Ready**: JSON logging for log aggregation systems

## Timeline

- **Week 1**: Core library structure, base classes, and logging system
- **Week 2**: Protocol adapters and dependency system
- **Week 3**: Agent implementations and registry
- **Week 4**: Demo runners and examples
- **Week 5**: Testing, documentation, and cleanup

## Success Criteria

1. All existing functionality preserved
2. Examples reduced to <100 lines each
3. Protocol support is optional via extras
4. No conditional imports in core code
5. All tests passing
6. Documentation updated
7. Comprehensive logging at all levels
8. Zero performance impact when logging is disabled 