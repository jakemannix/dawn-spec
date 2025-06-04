"""
Implementation of a gateway agent using the Agent Gateway Protocol (AGP).

This module provides a gRPC-based gateway agent that implements the 
IGatewayAgent interface and uses the AGP protocol for agent registration 
and discovery.
"""
import os
import threading
import logging
import time
import uuid
from concurrent import futures
from typing import Dict, List, Optional, Any, Union
import grpc

from .interfaces import IGatewayAgent, IAgent

# Proto imports will be available after generation
# from .proto import agp_pb2, agp_pb2_grpc


class AgpGatewayAgent(IGatewayAgent):
    """
    Gateway agent implementation using the AGP protocol with gRPC.
    
    This class implements the IGatewayAgent interface and provides
    a gRPC server for agent registration and discovery.
    """
    
    def __init__(
        self,
        name: str = "AGP Gateway",
        description: str = "DAWN Gateway Agent using gRPC-based AGP",
        host: str = "localhost",
        port: int = 50051,
        capabilities: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new AGP Gateway Agent.
        
        Args:
            name: Human-readable name for the gateway
            description: Description of the gateway's purpose
            host: Host address to bind the gRPC server to
            port: Port number to bind the gRPC server to
            capabilities: List of gateway capabilities
            metadata: Additional metadata about the gateway
        """
        self._id = f"agp-gateway-{str(uuid.uuid4())[:8]}"
        self._name = name
        self._description = description
        self._host = host
        self._port = port
        self._registry = {}  # Dictionary mapping agent_id to agent_info
        
        # Default capabilities if none provided
        if not capabilities:
            capabilities = [
                {
                    "id": "gateway-registration",
                    "type": "agent_registration",
                    "name": "Agent Registration",
                    "description": "Register and unregister agents with the gateway"
                },
                {
                    "id": "gateway-discovery",
                    "type": "agent_discovery",
                    "name": "Agent Discovery",
                    "description": "Find agents based on capabilities and other criteria"
                }
            ]
        
        self._capabilities = capabilities
        self._metadata = metadata or {}
        
        # gRPC server
        self._server = None
        self._shutdown_event = threading.Event()
        
        # Logging
        self._logger = logging.getLogger("agp_gateway")
    
    def start_server(self):
        """Start the gRPC server for the gateway."""
        self._logger.info(f"Starting AGP Gateway server on {self._host}:{self._port}")
        
        # Create gRPC server
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add service to server - commented until proto generation
        # agp_pb2_grpc.add_AgentGatewayServiceServicer_to_server(
        #     AgentGatewayServicer(self), self._server
        # )
        
        # Bind to address
        address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(address)
        
        # Start server
        self._server.start()
        self._logger.info(f"AGP Gateway server started on {address}")
        
        return address
    
    def stop_server(self):
        """Stop the gRPC server."""
        if self._server:
            self._logger.info("Stopping AGP Gateway server...")
            self._server.stop(grace=5)  # Allow 5 seconds for clean shutdown
            self._logger.info("AGP Gateway server stopped")
    
    def serve_forever(self):
        """Run the server until interrupted."""
        self.start_server()
        
        try:
            # Keep running until shutdown event is set
            while not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received, shutting down")
        finally:
            self.stop_server()
    
    # IGatewayAgent interface implementation
    def get_info(self) -> Dict[str, Any]:
        """Return gateway agent metadata including capabilities."""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities,
            "metadata": {
                **self._metadata,
                "host": self._host,
                "port": self._port
            }
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of gateway capabilities."""
        return self._capabilities
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a gateway capability with given inputs and configuration."""
        # Gateway doesn't directly implement invoke - use specific methods instead
        return {"error": "Gateway does not support direct invocation. Use specific gateway methods instead."}
    
    def health_check(self) -> bool:
        """Check if the gateway is functioning properly."""
        # Simple health check - ensure the server is running
        return self._server is not None and self._server._state.running
    
    def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """Register an agent in the gateway."""
        # Validate agent info
        if "id" not in agent_info:
            agent_id = str(uuid.uuid4())
            agent_info["id"] = agent_id
        else:
            agent_id = agent_info["id"]
        
        # Add to registry
        self._registry[agent_id] = agent_info
        self._logger.info(f"Registered agent: {agent_id} ({agent_info.get('name', 'Unknown')})")
        
        return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._registry:
            agent_info = self._registry.pop(agent_id)
            self._logger.info(f"Unregistered agent: {agent_id} ({agent_info.get('name', 'Unknown')})")
            return True
        
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        return self._registry.get(agent_id)
    
    def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List agents matching optional filters."""
        if not filters:
            return list(self._registry.values())
        
        # Apply filters
        results = []
        for agent in self._registry.values():
            match = True
            for key, value in filters.items():
                if key not in agent or agent[key] != value:
                    match = False
                    break
            
            if match:
                results.append(agent)
        
        return results
    
    def find_agents_by_capability(self, capability_type: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find agents with a specific capability type and parameters."""
        results = []
        
        for agent in self._registry.values():
            for capability in agent.get("capabilities", []):
                if capability.get("type") == capability_type:
                    # If parameters are specified, check if they match
                    if parameters:
                        capability_params = capability.get("parameters", {})
                        if all(capability_params.get(k) == v for k, v in parameters.items()):
                            results.append(agent)
                            break
                    else:
                        results.append(agent)
                        break
        
        return results
    
    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Validate an agent's capabilities and accessibility."""
        if agent_id not in self._registry:
            return {"valid": False, "reason": "Agent not found"}
        
        # TODO: Implement actual validation by attempting to connect and
        # checking agent's capabilities
        
        # Simple validation for now
        return {"valid": True, "agent_id": agent_id}


# This class will be used for the gRPC service implementation
# class AgentGatewayServicer(agp_pb2_grpc.AgentGatewayServiceServicer):
#     """
#     gRPC service implementation for the AgentGatewayService.
#     
#     This class handles the gRPC service implementation of the AGP protocol.
#     """
#     
#     def __init__(self, gateway: AgpGatewayAgent):
#         """
#         Initialize the service with a reference to the gateway agent.
#         
#         Args:
#             gateway: The gateway agent implementation
#         """
#         self._gateway = gateway
#     
#     # Implement service methods...