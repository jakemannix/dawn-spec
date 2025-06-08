# Dawn Specification Test Suite

This document describes the test suite for the Dawn specification that can run without requiring LLM API keys.

## Overview

The test suite is designed to validate the core functionality of the Dawn specification using mock implementations and synthetic data instead of actual LLM calls. This allows developers to:

1. Run tests locally without API keys
2. Ensure consistent test results
3. Test edge cases and error conditions
4. Validate the architecture independently of external services

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and mocks
├── unit/                    # Unit tests
│   ├── test_agent.py        # Base Agent class tests
│   ├── test_langgraph_agent.py  # LangGraph implementation tests
│   ├── test_protocol_adapter.py # Protocol adapter tests
│   ├── test_agent_registry.py   # Agent registry tests
│   ├── test_github_agent.py     # GitHub agent tests
│   └── test_demo_runner.py      # Demo runner tests
└── integration/             # Integration tests
    └── test_integration.py  # End-to-end tests
```

## Key Components

### Mock LLM (conftest.py)

The `MockLLM` class simulates LLM responses based on synthetic data:

```python
SYNTHETIC_RESPONSES = {
    "search for popular Python web frameworks": {
        "content": "I'll search for popular Python web frameworks on GitHub.\n\nSearching...",
        "tool_calls": [{
            "name": "github_search",
            "args": {"query": "Python web framework", "max_results": 5},
            "id": "call_123"
        }]
    },
    # ... more responses
}
```

### Mock Tools

Mock implementations of tools return realistic but synthetic data:

```python
class MockTool:
    async def ainvoke(self, input_data: Dict[str, Any]) -> str:
        if self.name == "github_search":
            return "Found 5 repositories: [{'name': 'django/django', 'stars': 50000}, ...]"
        # ... more tool responses
```

### Test Fixtures

Common fixtures provide:
- `mock_llm`: A MockLLM instance
- `mock_github_client`: Mock GitHub API client
- `mock_aiohttp_app`: Mock web application for protocol tests
- `env_without_llm_keys`: Environment without API keys
- `env_with_mock_keys`: Environment with mock API keys

## Running Tests

### Install Dependencies

```bash
# Install all test dependencies
uv sync --extra dev --extra langgraph --extra interop
```

### Run All Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src/dawn --cov-report=html

# Run only unit tests
uv run pytest tests/unit/

# Run specific test file
uv run pytest tests/unit/test_agent.py -v
```

### Test Markers

Tests are marked for selective execution:

```bash
# Run only unit tests
uv run pytest -m unit

# Skip integration tests
uv run pytest -m "not integration"

# Run tests that don't need external services
uv run pytest -m "not llm and not github"
```

## Test Coverage

### Core Components (100% Coverage Goal)

1. **Agent Base Class** (`test_agent.py`)
   - Agent initialization with various parameters
   - Unique agent ID generation
   - Message processing interface
   - Capabilities and tools retrieval
   - Agent card generation
   - Lifecycle methods (initialize/shutdown)

2. **LangGraph Agent** (`test_langgraph_agent.py`)
   - LLM initialization with different providers
   - Tool registration and execution
   - Graph building and execution
   - Error handling in message processing
   - Context preservation

3. **Protocol Adapters** (`test_protocol_adapter.py`)
   - Base adapter lifecycle (start/stop)
   - Protocol endpoint management
   - Request handling
   - Agent card formatting for protocols
   - ACP adapter web server creation

4. **Agent Registry** (`test_agent_registry.py`)
   - Agent registration with metadata
   - Dynamic agent creation
   - Finding agents by tags
   - Registry clearing
   - Metadata retrieval

5. **GitHub Agent** (`test_github_agent.py`)
   - Tool implementations (search, analyze, trending)
   - Issue and PR retrieval
   - Message processing with tool calls
   - Error handling

### Integration Tests

1. **End-to-End Flow** (`test_integration.py`)
   - Agent creation from registry
   - Protocol adapter integration
   - Multiple agents with same protocol
   - Error recovery scenarios

## Synthetic Test Data

### LLM Responses

The test suite includes realistic LLM responses for common queries:
- GitHub repository searches
- Repository analysis requests
- ArXiv paper searches
- Technology comparisons

### Tool Results

Mock tool results include:
- Repository metadata (name, stars, language)
- Issue and PR information
- Search results with realistic data

### Error Scenarios

Tests cover various error conditions:
- Missing API keys
- Network failures
- Invalid tool parameters
- Agent initialization failures

## Best Practices

1. **Use Fixtures**: Always use provided fixtures for consistency
2. **Mock External Services**: Never make real API calls in tests
3. **Test Error Cases**: Include tests for failure scenarios
4. **Use Synthetic Data**: Keep test data realistic but deterministic
5. **Mark Async Tests**: Use `@pytest.mark.asyncio` for async tests
6. **Isolate Tests**: Each test should be independent

## Adding New Tests

When adding new tests:

1. Use appropriate fixtures from `conftest.py`
2. Mock any external dependencies
3. Add synthetic responses for new LLM interactions
4. Mark tests appropriately (unit, integration, etc.)
5. Ensure tests work without API keys

Example:

```python
@pytest.mark.unit
class TestNewAgent:
    def test_initialization(self, mock_llm):
        """Test agent initialization."""
        agent = NewAgent()
        assert agent.name == "Expected Name"
    
    @pytest.mark.asyncio
    async def test_process_message(self, mock_llm):
        """Test message processing."""
        agent = NewAgent()
        result = await agent.process_message("test")
        assert result["success"] is True
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src` is in Python path (handled by conftest.py)
2. **Async Test Failures**: Add `@pytest.mark.asyncio` decorator
3. **Missing Dependencies**: Run `uv sync` with appropriate extras
4. **Mock Validation Errors**: Ensure mocks match expected interfaces

### Debugging

```bash
# Run with verbose output
uv run pytest -vv

# Show print statements
uv run pytest -s

# Run specific test with debugging
uv run pytest tests/unit/test_agent.py::TestBaseAgent::test_agent_initialization -vv
```

## Future Improvements

1. Add performance benchmarks
2. Implement property-based testing
3. Add mutation testing
4. Increase integration test coverage
5. Add visual test reports
6. Implement test data factories 