"""
Base Agent Interface

This module defines the core agent interface that all Dawn agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid

from dawn.utils.logging import get_logger


class Agent(ABC):
    """
    Base agent interface for the Dawn framework.
    
    All agents must implement this interface to ensure compatibility
    with the protocol adapters and runners.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Agent",
        description: str = "A Dawn agent",
        version: str = "1.0.0",
    ):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent (generated if not provided)
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            version: Version of the agent implementation
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version
        self.logger = get_logger(f"agent.{self.name.lower()}")
        
        self.logger.info(
            f"Initialized {self.name} (ID: {self.agent_id}, Version: {self.version})"
        )
    
    @abstractmethod
    async def process_message(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        This is the main entry point for agent interactions.
        
        Args:
            message: The input message to process
            context: Optional context information (e.g., conversation history)
            
        Returns:
            Dict containing at minimum:
                - response: The agent's response
                - success: Boolean indicating if processing was successful
                Additional fields may include:
                - tools_used: List of tools that were invoked
                - reasoning_trace: List of reasoning steps
                - metadata: Any additional metadata
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Return the agent's capabilities in OASF format.
        
        Returns:
            List of capability definitions, each containing:
                - name: Capability name
                - type: Capability type (e.g., "skill", "tool")
                - description: What the capability does
                - parameters: Expected parameters (if applicable)
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[str]:
        """
        Return the names of tools available to this agent.
        
        Returns:
            List of tool names that this agent can use
        """
        pass
    
    def get_agent_card(self) -> Dict[str, Any]:
        """
        Generate an agent card with metadata about this agent.
        
        This provides a standardized way to describe the agent
        for discovery and integration purposes.
        
        Returns:
            Dict containing agent metadata including:
                - id: Agent ID
                - name: Agent name
                - description: Agent description
                - version: Agent version
                - capabilities: List of capabilities
                - tools: List of available tools
        """
        return {
            "id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.get_capabilities(),
            "tools": self.get_tools(),
            "metadata": {
                "framework": "dawn",
                "framework_version": "0.2.0",
            }
        }
    
    async def initialize(self) -> None:
        """
        Perform any async initialization required by the agent.
        
        This is called after construction but before the agent
        starts processing messages. Override if needed.
        """
        pass
    
    async def shutdown(self) -> None:
        """
        Perform cleanup when the agent is shutting down.
        
        Override this method if your agent needs to clean up
        resources, close connections, etc.
        """
        pass 