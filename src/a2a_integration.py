"""
A2A SDK integration for DAWN agents.

This module provides integration between DAWN agents and Google's A2A (Agent-to-Agent) protocol,
allowing DAWN agents to communicate via the standardized A2A protocol.
"""
from typing import Any, Dict, List, Optional
import asyncio
import logging
from a2a_sdk import A2AServer, A2AClient, Tool, ToolParameter
from src.interfaces import IAgent

logger = logging.getLogger(__name__)


class DawnA2AServer:
    """
    Wrapper to expose DAWN agents as A2A servers.
    
    This class takes a DAWN agent and exposes its capabilities as A2A tools,
    allowing other A2A clients to discover and invoke the agent's capabilities.
    """
    
    def __init__(self, dawn_agent: IAgent, port: int = 8080, host: str = "localhost"):
        """
        Initialize the A2A server wrapper.
        
        Args:
            dawn_agent: The DAWN agent to expose via A2A
            port: Port to run the A2A server on
            host: Host to bind the A2A server to
        """
        self.dawn_agent = dawn_agent
        self.port = port
        self.host = host
        self.a2a_server = A2AServer(port=port, host=host)
        self._register_dawn_capabilities()
        
    def _register_dawn_capabilities(self) -> None:
        """
        Register DAWN agent capabilities with the A2A server as tools.
        
        Each DAWN capability becomes an A2A tool that can be discovered and invoked
        by remote A2A clients.
        """
        agent_info = self.dawn_agent.get_info()
        capabilities = self.dawn_agent.get_capabilities()
        
        logger.info(f"Registering {len(capabilities)} capabilities for agent {agent_info.get('name', 'unknown')}")
        
        for capability in capabilities:
            self._create_a2a_tool_from_capability(capability)
            
    def _create_a2a_tool_from_capability(self, capability: Dict[str, Any]) -> None:
        """
        Convert a DAWN capability to an A2A tool and register it.
        
        Args:
            capability: DAWN capability dictionary containing id, name, description, etc.
        """
        capability_id = capability['id']
        capability_name = capability.get('name', capability_id)
        capability_description = capability.get('description', f"Tool for {capability_name}")
        
        # Extract parameters from capability schema
        parameters = []
        if 'parameters' in capability and isinstance(capability['parameters'], dict):
            for param_name, param_info in capability['parameters'].items():
                if isinstance(param_info, dict):
                    param_type = param_info.get('type', 'string')
                    param_description = param_info.get('description', f"Parameter {param_name}")
                    required = param_info.get('required', False)
                    
                    parameters.append(ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=param_description,
                        required=required
                    ))
                else:
                    # Simple parameter definition
                    parameters.append(ToolParameter(
                        name=param_name,
                        type='string',
                        description=f"Parameter {param_name}",
                        required=True
                    ))
        
        # Create A2A tool
        a2a_tool = Tool(
            name=capability_id,
            description=capability_description,
            parameters=parameters
        )
        
        # Define the tool handler function
        async def tool_handler(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Handler function for the A2A tool."""
            try:
                logger.info(f"Invoking DAWN capability {capability_id} with inputs: {inputs}")
                result = self.dawn_agent.invoke(capability_id, inputs)
                logger.info(f"DAWN capability {capability_id} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error invoking DAWN capability {capability_id}: {e}")
                return {"error": str(e), "capability_id": capability_id}
        
        # Register the tool with the A2A server
        self.a2a_server.register_tool(a2a_tool, tool_handler)
        logger.info(f"Registered A2A tool: {capability_id}")
        
    async def start(self) -> None:
        """
        Start the A2A server.
        
        This will make the DAWN agent's capabilities available to A2A clients.
        """
        agent_info = self.dawn_agent.get_info()
        logger.info(f"Starting A2A server for agent {agent_info.get('name', 'unknown')} on {self.host}:{self.port}")
        await self.a2a_server.start()
        
    async def stop(self) -> None:
        """Stop the A2A server."""
        logger.info("Stopping A2A server")
        await self.a2a_server.stop()
        
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the A2A server.
        
        Returns:
            Dictionary containing server information
        """
        agent_info = self.dawn_agent.get_info()
        return {
            "agent_id": agent_info.get('id'),
            "agent_name": agent_info.get('name'),
            "server_host": self.host,
            "server_port": self.port,
            "capabilities_count": len(self.dawn_agent.get_capabilities()),
            "a2a_tools_count": len(self.a2a_server.get_tools()) if hasattr(self.a2a_server, 'get_tools') else 0
        }


class DawnA2AClient:
    """
    Client for DAWN agents to communicate with remote A2A agents.
    
    This class allows DAWN agents to discover and invoke capabilities on remote
    A2A agents using the A2A protocol.
    """
    
    def __init__(self):
        """Initialize the A2A client."""
        self.a2a_client = A2AClient()
        self.connected_agents: Dict[str, Dict[str, Any]] = {}
        
    async def connect_to_agent(self, agent_url: str, agent_id: Optional[str] = None) -> str:
        """
        Connect to a remote A2A agent.
        
        Args:
            agent_url: URL of the remote A2A agent
            agent_id: Optional ID to use for the agent (will generate if not provided)
            
        Returns:
            Agent ID for the connected agent
        """
        if agent_id is None:
            agent_id = f"remote_agent_{len(self.connected_agents)}"
            
        try:
            # Connect to the remote agent
            await self.a2a_client.connect(agent_url)
            
            # Store connection info
            self.connected_agents[agent_id] = {
                "url": agent_url,
                "connected": True,
                "tools": None  # Will be populated when tools are discovered
            }
            
            logger.info(f"Connected to A2A agent at {agent_url} with ID {agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to connect to A2A agent at {agent_url}: {e}")
            raise
            
    async def discover_tools(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Discover available tools on a remote A2A agent.
        
        Args:
            agent_id: ID of the connected agent
            
        Returns:
            List of tool definitions
        """
        if agent_id not in self.connected_agents:
            raise ValueError(f"Agent {agent_id} is not connected")
            
        try:
            tools = await self.a2a_client.discover_tools()
            
            # Cache the tools
            self.connected_agents[agent_id]["tools"] = tools
            
            logger.info(f"Discovered {len(tools)} tools on agent {agent_id}")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to discover tools on agent {agent_id}: {e}")
            raise
            
    async def invoke_remote_capability(
        self, 
        agent_id: str, 
        tool_name: str, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a capability on a remote A2A agent.
        
        Args:
            agent_id: ID of the connected agent
            tool_name: Name of the tool/capability to invoke
            inputs: Input parameters for the tool
            
        Returns:
            Result from the remote agent
        """
        if agent_id not in self.connected_agents:
            raise ValueError(f"Agent {agent_id} is not connected")
            
        try:
            logger.info(f"Invoking tool {tool_name} on agent {agent_id} with inputs: {inputs}")
            result = await self.a2a_client.invoke_tool(tool_name, inputs)
            logger.info(f"Tool {tool_name} on agent {agent_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to invoke tool {tool_name} on agent {agent_id}: {e}")
            return {"error": str(e), "tool_name": tool_name, "agent_id": agent_id}
            
    def get_connected_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about connected agents.
        
        Returns:
            Dictionary mapping agent IDs to their connection info
        """
        return self.connected_agents.copy()
        
    async def disconnect_agent(self, agent_id: str) -> bool:
        """
        Disconnect from a remote A2A agent.
        
        Args:
            agent_id: ID of the agent to disconnect from
            
        Returns:
            True if successfully disconnected
        """
        if agent_id not in self.connected_agents:
            return False
            
        try:
            await self.a2a_client.disconnect()
            del self.connected_agents[agent_id]
            logger.info(f"Disconnected from agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from agent {agent_id}: {e}")
            return False