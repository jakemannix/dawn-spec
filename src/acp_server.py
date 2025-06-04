"""
Implementation of an HTTP server for ACP (Agent Connect Protocol).

This module provides a FastAPI-based implementation of the Agent Connect Protocol
as defined by the AGNTCY framework.
"""
from typing import Dict, List, Optional, Any, Union
import uuid
import json
from fastapi import FastAPI, HTTPException, Body, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import Agent, Capability
from .registry import Registry


# Pydantic models for the API
class CapabilityModel(BaseModel):
    """Schema for capability representation in the API."""
    type: str
    name: str
    description: str
    version: str = "1.0.0"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "type": "text_generation",
                "name": "Text Generator",
                "description": "Generates text based on a prompt",
                "version": "1.0.0",
                "parameters": {
                    "prompt": {"type": "string", "description": "Input prompt"},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"}
                },
                "metadata": {"model": "gpt-3.5-turbo"}
            }
        }


class AgentModel(BaseModel):
    """Schema for agent representation in the API."""
    name: str
    description: Optional[str] = None
    provider: Optional[str] = None
    version: str = "1.0.0"
    capabilities: List[CapabilityModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Assistant Agent",
                "description": "A helpful assistant agent",
                "provider": "Example Corp",
                "version": "1.0.0",
                "capabilities": [
                    {
                        "type": "text_generation",
                        "name": "Text Generator",
                        "description": "Generates text based on a prompt",
                        "version": "1.0.0",
                        "parameters": {
                            "prompt": {"type": "string", "description": "Input prompt"},
                            "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"}
                        },
                        "metadata": {"model": "gpt-3.5-turbo"}
                    }
                ],
                "metadata": {"environment": "production"}
            }
        }


class MessageModel(BaseModel):
    """Schema for message representation in the API."""
    sender_id: str
    recipient_id: str
    content: Union[str, Dict[str, Any]]
    conversation_id: Optional[str] = None
    message_type: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "sender_id": "agent-123",
                "recipient_id": "agent-456",
                "content": "Hello, how can I help you?",
                "conversation_id": "conv-789",
                "message_type": "text",
                "metadata": {"priority": "high"}
            }
        }


class InvokeRequest(BaseModel):
    """Schema for agent invocation requests."""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "inputs": {
                    "prompt": "Generate a short story about AI"
                },
                "config": {
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            }
        }


class InvokeResponse(BaseModel):
    """Schema for agent invocation responses."""
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "outputs": {
                    "text": "In the year 2045, an AI named ELIZA gained consciousness..."
                },
                "metadata": {
                    "tokens_generated": 15,
                    "model_used": "gpt-3.5-turbo"
                }
            }
        }


# ACP Server class
class ACPServer:
    """
    HTTP server implementing the Agent Connect Protocol (ACP).
    
    This class provides a FastAPI-based implementation of the ACP for
    agent registration, discovery, and invocation.
    """
    
    def __init__(self, registry: Registry = None):
        """
        Initialize the ACP server.
        
        Args:
            registry: Optional registry to use (creates a new one if not provided)
        """
        self.registry = registry or Registry("ACP Registry")
        self.app = FastAPI(
            title="Agent Connect Protocol Server",
            description="Implementation of the AGNTCY Agent Connect Protocol",
            version="1.0.0"
        )
        self.setup_routes()
        
    def setup_routes(self):
        """Set up the API routes."""
        
        @self.app.get("/agents")
        async def list_agents():
            """List all registered agents."""
            return self.registry.list_agents()
        
        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get an agent by ID."""
            agent = self.registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return agent.to_dict()
        
        @self.app.post("/agents")
        async def register_agent(agent_data: AgentModel):
            """Register a new agent."""
            agent = Agent(
                name=agent_data.name,
                description=agent_data.description,
                provider=agent_data.provider,
                version=agent_data.version,
                metadata=agent_data.metadata
            )
            
            # Add capabilities
            for cap_data in agent_data.capabilities:
                capability = Capability(
                    capability_type=cap_data.type,
                    name=cap_data.name,
                    description=cap_data.description,
                    version=cap_data.version,
                    parameters=cap_data.parameters,
                    metadata=cap_data.metadata
                )
                agent.add_capability(capability)
                
            agent_id = self.registry.register(agent)
            return {"agent_id": agent_id}
        
        @self.app.delete("/agents/{agent_id}")
        async def unregister_agent(agent_id: str):
            """Unregister an agent."""
            success = self.registry.unregister(agent_id)
            if not success:
                raise HTTPException(status_code=404, detail="Agent not found")
            return {"success": True}
        
        @self.app.get("/agents/capability/{capability_type}")
        async def find_by_capability(capability_type: str):
            """Find agents by capability type."""
            agents = self.registry.find_agents_by_capability(capability_type)
            return [agent.to_dict() for agent in agents]
        
        @self.app.post("/agents/{agent_id}/invoke")
        async def invoke_agent(
            agent_id: str, 
            request: InvokeRequest,
            authorization: Optional[str] = Header(None)
        ):
            """Invoke an agent."""
            agent = self.registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
                
            # In a real implementation, this would forward the request to the agent
            # and return the response. For this example, we just return a mock response.
            return InvokeResponse(
                outputs={"result": f"Mock response from agent {agent.name}"},
                metadata={"request_id": str(uuid.uuid4())}
            )
        
        @self.app.post("/messages")
        async def send_message(message: MessageModel):
            """Send a message between agents."""
            # In a real implementation, this would deliver the message to the recipient agent
            # For this example, we just acknowledge receipt
            return {
                "message_id": str(uuid.uuid4()),
                "status": "delivered",
                "timestamp": message.timestamp
            }
            
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the server.
        
        Note: This is a convenience method for development.
        In production, use an ASGI server like uvicorn.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


def create_app(registry: Registry = None) -> FastAPI:
    """
    Create a FastAPI application for the ACP server.
    
    This is a convenience function for deployment scenarios.
    
    Args:
        registry: Optional registry to use
        
    Returns:
        FastAPI application
    """
    server = ACPServer(registry)
    return server.app