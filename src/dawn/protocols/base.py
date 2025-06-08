"""
Protocol Adapter Interface

This module defines the interface that all protocol adapters must implement
to integrate with the Dawn framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from dawn.core.agent import Agent
from dawn.utils.logging import get_logger


@dataclass
class ProtocolEndpoint:
    """Information about a protocol endpoint."""
    protocol: str
    host: str
    port: int
    base_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """Get the full URL for this endpoint."""
        return self.base_url


class ProtocolAdapter(ABC):
    """
    Base protocol adapter interface.
    
    Protocol adapters handle the translation between Dawn agents
    and specific protocol implementations (A2A, ACP, MCP, etc.).
    """
    
    def __init__(self, logger=None):
        """
        Initialize the protocol adapter.
        
        Args:
            logger: Optional logger instance (creates one if not provided)
        """
        self.logger = logger or get_logger(f"protocol.{self.get_protocol_name()}")
        self.agent: Optional[Agent] = None
        self.endpoint: Optional[ProtocolEndpoint] = None
        self._is_running = False
    
    @abstractmethod
    async def start(self, agent: Agent, host: str = "localhost", port: int = 8080) -> ProtocolEndpoint:
        """
        Start the protocol server/client for an agent.
        
        Args:
            agent: The Dawn agent to expose via this protocol
            host: Host to bind to (default: localhost)
            port: Port to bind to (default: 8080)
            
        Returns:
            ProtocolEndpoint with connection information
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the protocol server/client.
        
        This should clean up all resources and connections.
        """
        pass
    
    @abstractmethod
    def get_protocol_name(self) -> str:
        """
        Get the name of this protocol.
        
        Returns:
            Protocol name (e.g., "a2a", "acp", "mcp")
        """
        pass
    
    @abstractmethod
    def get_protocol_version(self) -> str:
        """
        Get the version of the protocol this adapter supports.
        
        Returns:
            Protocol version string
        """
        pass
    
    def is_running(self) -> bool:
        """
        Check if the protocol adapter is currently running.
        
        Returns:
            True if the adapter is running, False otherwise
        """
        return self._is_running
    
    def get_endpoint(self) -> Optional[ProtocolEndpoint]:
        """
        Get the current endpoint information.
        
        Returns:
            ProtocolEndpoint if running, None otherwise
        """
        return self.endpoint if self.is_running() else None
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming protocol request.
        
        This is a convenience method that adapters can override to provide
        a unified request handling interface.
        
        Args:
            request: The incoming request data
            
        Returns:
            Response data
        """
        if not self.agent:
            return {
                "error": "No agent configured",
                "success": False
            }
        
        # Default implementation delegates to agent
        message = request.get("message", "")
        context = request.get("context", {})
        
        return await self.agent.process_message(message, context)
    
    def get_agent_card_for_protocol(self) -> Dict[str, Any]:
        """
        Get the agent card formatted for this specific protocol.
        
        Subclasses can override this to provide protocol-specific formatting.
        
        Returns:
            Agent card data formatted for the protocol
        """
        if not self.agent:
            return {}
        
        base_card = self.agent.get_agent_card()
        
        # Add protocol-specific metadata
        base_card["protocol"] = {
            "name": self.get_protocol_name(),
            "version": self.get_protocol_version(),
            "endpoint": self.endpoint.url if self.endpoint else None
        }
        
        return base_card 