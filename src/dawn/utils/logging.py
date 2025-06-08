"""
Dawn Logging System

Provides structured logging with different verbosity levels and JSON formatting
for agent messages and tool calls.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Union
from enum import Enum
from contextvars import ContextVar
from functools import wraps
import traceback


# Context variables for tracking request/agent context
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
agent_id_var: ContextVar[Optional[str]] = ContextVar('agent_id', default=None)


class LogLevel(Enum):
    """Extended log levels for Dawn."""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5  # For very detailed tracing
    

class LogEvent(Enum):
    """Types of events we want to track."""
    # Agent lifecycle
    AGENT_START = "agent.start"
    AGENT_STOP = "agent.stop"
    AGENT_ERROR = "agent.error"
    
    # Message processing
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_SENT = "message.sent"
    MESSAGE_ERROR = "message.error"
    
    # Tool usage
    TOOL_CALL = "tool.call"
    TOOL_RESPONSE = "tool.response"
    TOOL_ERROR = "tool.error"
    
    # Protocol events
    PROTOCOL_START = "protocol.start"
    PROTOCOL_STOP = "protocol.stop"
    PROTOCOL_ERROR = "protocol.error"
    
    # LLM interactions
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"


class StructuredLogger:
    """Structured logger with JSON output support."""
    
    def __init__(self, name: str, level: Union[int, LogLevel] = LogLevel.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.set_level(level)
        
        # Configure handlers if not already configured
        if not self.logger.handlers:
            self._configure_handlers()
    
    def _configure_handlers(self):
        """Configure logging handlers based on environment."""
        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Use JSON formatter for production, human-readable for development
        if self._should_use_json():
            formatter = JsonFormatter()
        else:
            formatter = HumanReadableFormatter()
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _should_use_json(self) -> bool:
        """Determine if JSON logging should be used."""
        import os
        return os.getenv('DAWN_LOG_FORMAT', '').lower() == 'json'
    
    def set_level(self, level: Union[int, LogLevel]):
        """Set the logging level."""
        if isinstance(level, LogLevel):
            level = level.value
        self.logger.setLevel(level)
    
    def _create_event(self, event: LogEvent, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured event."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'logger': self.name,
            'event': event.value,
            'request_id': request_id_var.get(),
            'agent_id': agent_id_var.get(),
            **data
        }
    
    # Agent lifecycle logging
    
    def agent_start(self, agent_id: str, agent_type: str, protocols: list):
        """Log agent startup."""
        event = self._create_event(LogEvent.AGENT_START, {
            'agent_id': agent_id,
            'agent_type': agent_type,
            'protocols': protocols
        })
        self.logger.info(json.dumps(event))
    
    def agent_stop(self, agent_id: str):
        """Log agent shutdown."""
        event = self._create_event(LogEvent.AGENT_STOP, {
            'agent_id': agent_id
        })
        self.logger.info(json.dumps(event))
    
    # Message logging
    
    def message_received(self, message: str, source: str, metadata: Optional[Dict] = None):
        """Log received message (verbose mode)."""
        if self.logger.level <= LogLevel.DEBUG.value:
            event = self._create_event(LogEvent.MESSAGE_RECEIVED, {
                'message': message,
                'source': source,
                'metadata': metadata or {},
                'message_length': len(message)
            })
            self.logger.debug(json.dumps(event))
    
    def message_sent(self, message: str, destination: str, metadata: Optional[Dict] = None):
        """Log sent message (verbose mode)."""
        if self.logger.level <= LogLevel.DEBUG.value:
            event = self._create_event(LogEvent.MESSAGE_SENT, {
                'message': message,
                'destination': destination,
                'metadata': metadata or {},
                'message_length': len(message)
            })
            self.logger.debug(json.dumps(event))
    
    # Tool logging
    
    def tool_call(self, tool_name: str, arguments: Dict[str, Any], agent_id: Optional[str] = None):
        """Log tool invocation (verbose mode)."""
        if self.logger.level <= LogLevel.DEBUG.value:
            event = self._create_event(LogEvent.TOOL_CALL, {
                'tool_name': tool_name,
                'arguments': arguments,
                'agent_id': agent_id or agent_id_var.get()
            })
            self.logger.debug(json.dumps(event))
    
    def tool_response(self, tool_name: str, response: Any, duration_ms: float, agent_id: Optional[str] = None):
        """Log tool response (verbose mode)."""
        if self.logger.level <= LogLevel.DEBUG.value:
            event = self._create_event(LogEvent.TOOL_RESPONSE, {
                'tool_name': tool_name,
                'response': str(response)[:500],  # Truncate large responses
                'duration_ms': duration_ms,
                'agent_id': agent_id or agent_id_var.get()
            })
            self.logger.debug(json.dumps(event))
    
    def tool_error(self, tool_name: str, error: Exception, agent_id: Optional[str] = None):
        """Log tool error."""
        event = self._create_event(LogEvent.TOOL_ERROR, {
            'tool_name': tool_name,
            'error': str(error),
            'error_type': type(error).__name__,
            'traceback': traceback.format_exc(),
            'agent_id': agent_id or agent_id_var.get()
        })
        self.logger.error(json.dumps(event))
    
    # LLM logging
    
    def llm_request(self, model: str, messages: list, temperature: float, max_tokens: int):
        """Log LLM request (trace level)."""
        if self.logger.level <= LogLevel.TRACE.value:
            event = self._create_event(LogEvent.LLM_REQUEST, {
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'message_count': len(messages)
            })
            self.logger.log(LogLevel.TRACE.value, json.dumps(event))
    
    def llm_response(self, model: str, response: str, tokens_used: Dict[str, int], duration_ms: float):
        """Log LLM response (trace level)."""
        if self.logger.level <= LogLevel.TRACE.value:
            event = self._create_event(LogEvent.LLM_RESPONSE, {
                'model': model,
                'response': response[:500],  # Truncate
                'tokens_used': tokens_used,
                'duration_ms': duration_ms
            })
            self.logger.log(LogLevel.TRACE.value, json.dumps(event))
    
    # Standard logging methods
    
    def info(self, message: str, **kwargs):
        """Standard info logging."""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Standard debug logging."""
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Standard warning logging."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Standard error logging."""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        self.logger.error(message, extra=kwargs)


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Check if the message is already JSON
        try:
            message_data = json.loads(record.getMessage())
            return record.getMessage()  # Already formatted
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Format as JSON
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'request_id': request_id_var.get(),
            'agent_id': agent_id_var.get(),
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable log formatter for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading."""
        # Check if the message is JSON and pretty-print it
        try:
            message_data = json.loads(record.getMessage())
            
            # Extract key fields for header
            event = message_data.get('event', 'unknown')
            agent_id = message_data.get('agent_id', 'no-agent')
            
            # Format header
            header = f"[{record.levelname}] [{event}] [{agent_id}]"
            
            # Pretty print the data
            formatted_data = json.dumps(message_data, indent=2)
            
            return f"{header}\n{formatted_data}"
            
        except (json.JSONDecodeError, TypeError):
            # Regular message
            timestamp = datetime.now().strftime('%H:%M:%S')
            agent_id = agent_id_var.get() or 'no-agent'
            
            prefix = f"{timestamp} [{record.levelname}] [{agent_id}]"
            return f"{prefix} {record.getMessage()}"


# Decorators for automatic logging

def log_tool_call(logger: StructuredLogger):
    """Decorator to automatically log tool calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            tool_name = func.__name__
            
            # Log the call
            logger.tool_call(tool_name, {'args': str(args), 'kwargs': str(kwargs)})
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.tool_response(tool_name, result, duration_ms)
                return result
            except Exception as e:
                logger.tool_error(tool_name, e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            tool_name = func.__name__
            
            # Log the call
            logger.tool_call(tool_name, {'args': str(args), 'kwargs': str(kwargs)})
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.tool_response(tool_name, result, duration_ms)
                return result
            except Exception as e:
                logger.tool_error(tool_name, e)
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Factory function for creating loggers

def get_logger(name: str, level: Optional[Union[str, int, LogLevel]] = None) -> StructuredLogger:
    """Get or create a structured logger."""
    if level is None:
        import os
        level_str = os.getenv('DAWN_LOG_LEVEL', 'INFO')
        level = getattr(LogLevel, level_str.upper(), LogLevel.INFO)
    
    return StructuredLogger(name, level)


# Context managers for request/agent tracking

class RequestContext:
    """Context manager for tracking request ID."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.token = None
    
    def __enter__(self):
        self.token = request_id_var.set(self.request_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_var.reset(self.token)


class AgentContext:
    """Context manager for tracking agent ID."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.token = None
    
    def __enter__(self):
        self.token = agent_id_var.set(self.agent_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        agent_id_var.reset(self.token) 