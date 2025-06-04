"""
Implementation of agents based on AGNTCY's framework specification.

This module implements the core agent components following the AGNTCY framework,
including the Agent Connect Protocol (ACP) and Agent Gateway Protocol (AGP) concepts.
"""
from typing import Dict, List, Optional, Any, Union
import uuid
import datetime


class Capability:
    """
    Represents a capability that an agent can provide based on the OASF.
    
    Following the Open Agent Schema Framework for consistent capability representation.
    """
    
    def __init__(
        self, 
        capability_type: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new capability.
        
        Args:
            capability_type: Type of capability (e.g., "text_generation", "calculation")
            name: Human-readable name for the capability
            description: Description of what the capability does
            version: Semantic version of the capability
            parameters: Optional parameter schema for the capability
            metadata: Additional metadata about the capability
        """
        self.id = str(uuid.uuid4())
        self.type = capability_type
        self.name = name
        self.description = description
        self.version = version
        self.parameters = parameters or {}
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the capability to a dictionary representation.
        
        Returns:
            Dictionary representing the capability in OASF format
        """
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters,
            "metadata": self.metadata
        }


class Agent:
    """
    Implementation of an agent following the AGNTCY framework.
    
    This class provides the core functionality for creating agents that can
    interact according to the AGNTCY specifications.
    """
    
    def __init__(
        self, 
        name: str, 
        description: Optional[str] = None,
        provider: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new Agent instance.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose or capabilities
            provider: The entity that provides this agent
            version: Semantic version of the agent
            metadata: Additional metadata about the agent
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.provider = provider
        self.version = version
        self.metadata = metadata or {}
        self.capabilities: List[Capability] = []
        self.created_at = datetime.datetime.utcnow().isoformat()
        
    def add_capability(self, capability: Capability) -> None:
        """
        Add a capability to this agent.
        
        Args:
            capability: Capability object to add
        """
        self.capabilities.append(capability)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation following OASF.
        
        Returns:
            Dictionary representing the agent
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "version": self.version,
            "created_at": self.created_at,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "metadata": self.metadata
        }


class Message:
    """
    Represents a message in the AGNTCY protocol communication.
    
    Follows the message structure concepts from AGP for agent communication.
    """
    
    def __init__(
        self, 
        sender_id: str,
        recipient_id: str,
        content: Union[str, Dict[str, Any]],
        conversation_id: Optional[str] = None,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new Message.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            content: Message content (string or structured data)
            conversation_id: Optional ID for grouping related messages
            message_type: Type of message (e.g., "text", "request", "response")
            metadata: Optional metadata for the message
        """
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.content = content
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.message_type = message_type
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.utcnow().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary representation.
        
        Returns:
            Dictionary representing the message
        """
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "message_type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }