# Dawn Logging System

## Overview

The Dawn logging system provides comprehensive observability for multi-agent systems with structured JSON logging, multiple verbosity levels, and automatic tracking of agent messages and tool calls.

## Key Features

### 1. Multiple Log Levels

```python
from dawn.utils.logging import get_logger, LogLevel

logger = get_logger(__name__)
logger.set_level(LogLevel.DEBUG)  # or INFO, WARNING, ERROR, TRACE
```

**Available Levels:**
- `TRACE` (5): Very detailed, includes LLM requests/responses
- `DEBUG` (10): Detailed info, includes messages and tool calls
- `INFO` (20): General information (default)
- `WARNING` (30): Warning conditions
- `ERROR` (40): Operation failures
- `CRITICAL` (50): System failures

### 2. Structured JSON Logging

Every log event is structured with consistent fields:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "logger": "agent.github",
  "event": "message.received",
  "request_id": "req-123",
  "agent_id": "github-agent-456",
  "message": "Search for Python repos",
  "source": "user",
  "message_length": 23
}
```

### 3. Automatic Tool Call Logging

Use the decorator to automatically log all tool invocations:

```python
from dawn.utils.logging import log_tool_call

@log_tool_call(logger)
async def github_search(query: str) -> Dict[str, Any]:
    # Your tool implementation
    return results
```

This automatically logs:
- **tool.call**: When the tool is invoked with arguments
- **tool.response**: When it returns successfully with duration
- **tool.error**: If an exception occurs with full traceback

### 4. Agent Message Tracking

Track all agent communications:

```python
# In your agent
logger.message_received(message, source="user")
# ... process message ...
logger.message_sent(response, destination="user")
```

Messages are only logged at DEBUG level or lower to avoid noise in production.

### 5. LLM Interaction Logging

Track LLM API calls at TRACE level:

```python
logger.llm_request(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=150
)

logger.llm_response(
    model="gpt-4",
    response="Hello! How can I help?",
    tokens_used={"prompt": 10, "completion": 5, "total": 15},
    duration_ms=234.5
)
```

### 6. Context Tracking

Use context managers to automatically include agent/request IDs:

```python
from dawn.utils.logging import AgentContext, RequestContext

with RequestContext("req-123"):
    with AgentContext("github-agent"):
        # All logs within these contexts automatically include IDs
        logger.info("Processing request")
```

## Configuration

### Environment Variables

```bash
# Set default log level
export DAWN_LOG_LEVEL=DEBUG

# Use JSON format (for production)
export DAWN_LOG_FORMAT=json

# Use human-readable format (for development)
export DAWN_LOG_FORMAT=human
```

### Programmatic Configuration

```python
# Set level for specific logger
logger = get_logger("dawn.agents.github")
logger.set_level(LogLevel.TRACE)

# Different levels for different components
get_logger("dawn.protocols").set_level(LogLevel.WARNING)
get_logger("dawn.agents").set_level(LogLevel.DEBUG)
```

## Output Formats

### Human-Readable (Development)

```
10:30:45 [DEBUG] [github-agent] Processing search query
[INFO] [message.received] [github-agent]
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "event": "message.received",
  "agent_id": "github-agent",
  "message": "Search for Python repos",
  "source": "user"
}
```

### JSON (Production)

```json
{"timestamp":"2024-01-15T10:30:45.123Z","level":"DEBUG","logger":"agent.github","event":"message.received","request_id":"req-123","agent_id":"github-agent","message":"Search for Python repos","source":"user"}
```

## Usage Examples

### Basic Agent with Logging

```python
from dawn.core import LangGraphAgent
from dawn.utils.logging import get_logger, log_tool_call

class MyAgent(LangGraphAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger(f"agent.{self.agent_id}")
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        # Automatic context tracking
        with AgentContext(self.agent_id):
            self.logger.message_received(message, source="user")
            
            # Process with tools
            result = await self.use_tool(message)
            
            self.logger.message_sent(result, destination="user")
            return {"response": result}
    
    @log_tool_call(logger)
    async def use_tool(self, input: str) -> str:
        # Tool implementation
        return f"Processed: {input}"
```

### Debug Mode for Development

```python
# Enable maximum verbosity for debugging
import os
os.environ['DAWN_LOG_LEVEL'] = 'TRACE'
os.environ['DAWN_LOG_FORMAT'] = 'human'

# Run your agent
runner = DemoRunner(agents=['github'], log_level="TRACE")
await runner.start()
```

### Production Configuration

```python
# JSON logs at INFO level for production
import os
os.environ['DAWN_LOG_LEVEL'] = 'INFO'
os.environ['DAWN_LOG_FORMAT'] = 'json'

# Only log errors from specific components
get_logger("dawn.protocols.mcp").set_level(LogLevel.ERROR)
```

## Performance Considerations

1. **Conditional Logging**: Messages and tool calls are only logged at DEBUG level or below
2. **Lazy Evaluation**: Log messages are only formatted if the level is enabled
3. **Truncation**: Large responses are automatically truncated to prevent log bloat
4. **Async Support**: All logging operations are non-blocking

## Integration with Log Aggregation

The JSON format is designed for easy integration with log aggregation systems:

- **Elasticsearch/Kibana**: Index on `event`, `agent_id`, `request_id`
- **Datadog**: Parse JSON and create metrics from `duration_ms`
- **CloudWatch**: Filter by `level` and `event` type
- **Grafana Loki**: Query by agent and event patterns

## Best Practices

1. **Use Context Managers**: Always wrap agent operations in `AgentContext`
2. **Set Appropriate Levels**: Use TRACE only for debugging, INFO for production
3. **Structure Your Events**: Use consistent event names and data structures
4. **Don't Log Sensitive Data**: Be careful with message content in production
5. **Use Request IDs**: Generate unique request IDs for tracing across services

## Troubleshooting

### Not Seeing Logs?

1. Check log level: `echo $DAWN_LOG_LEVEL`
2. Verify logger name matches your component
3. Ensure handlers are configured (happens automatically)

### Too Many Logs?

1. Increase log level: `export DAWN_LOG_LEVEL=INFO`
2. Disable specific loggers: `get_logger("noisy.component").set_level(LogLevel.ERROR)`

### Need More Detail?

1. Enable TRACE level: `export DAWN_LOG_LEVEL=TRACE`
2. Use human format for readability: `export DAWN_LOG_FORMAT=human`
3. Add custom fields to your logs using `extra` parameter 