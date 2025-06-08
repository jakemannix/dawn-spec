"""
MCP SDK integration for DAWN agents.

This module provides integration between DAWN agents and the Model Context Protocol (MCP),
allowing DAWN agents to expose their capabilities as MCP tools and to consume
MCP tools from other agents.
"""
from typing import Any, Dict, List, Optional, Callable
import asyncio
import logging
import inspect
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool, Resource
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.interfaces import IAgent
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MCPTool(ABC):
    """Abstract base class for MCP tools"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"MCPTool.{name}")
    
    @abstractmethod
    async def call(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments"""
        pass
    
    def __str__(self) -> str:
        return f"MCPTool({self.name})"
    
    def __repr__(self) -> str:
        return f"MCPTool(name='{self.name}', description='{self.description}')"


class MCPToolRegistry:
    """Registry for managing MCP tools"""
    
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self.logger = logging.getLogger("MCPToolRegistry")
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register an MCP tool"""
        self._tools[tool.name] = tool
        self.logger.info(f"Registered MCP tool: {tool.name}")
    
    def unregister_tool(self, name: str) -> Optional[MCPTool]:
        """Unregister an MCP tool"""
        tool = self._tools.pop(name, None)
        if tool:
            self.logger.info(f"Unregistered MCP tool: {name}")
        return tool
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get an MCP tool by name"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get list of all registered tools"""
        return list(self._tools.values())
    
    def list_tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return list(self._tools.keys())
    
    def clear(self) -> None:
        """Clear all registered tools"""
        self._tools.clear()
        self.logger.info("Cleared all MCP tools")


# Example/Mock MCP tools for testing
class MockMCPTool(MCPTool):
    """Mock MCP tool for testing purposes"""
    
    def __init__(self, name: str, description: str = "", mock_response: Any = "Mock response"):
        super().__init__(name, description)
        self.mock_response = mock_response
    
    async def call(self, arguments: Dict[str, Any]) -> Any:
        """Return mock response"""
        self.logger.info(f"Mock call to {self.name} with args: {arguments}")
        return self.mock_response


class EchoMCPTool(MCPTool):
    """Echo MCP tool that returns the input arguments"""
    
    def __init__(self):
        super().__init__("echo", "Echo tool that returns input arguments")
    
    async def call(self, arguments: Dict[str, Any]) -> Any:
        """Echo back the input arguments"""
        return f"Echo: {arguments}"


# Default registry instance
default_registry = MCPToolRegistry()


def get_default_registry() -> MCPToolRegistry:
    """Get the default MCP tool registry"""
    return default_registry


class DawnMCPServer:
    """
    MCP server that exposes DAWN agent capabilities as tools.
    
    This class takes a DAWN agent and exposes its capabilities as MCP tools,
    allowing MCP clients to discover and invoke the agent's capabilities.
    """
    
    def __init__(self, dawn_agent: IAgent, server_name: str = "dawn-agent"):
        """
        Initialize the MCP server wrapper.
        
        Args:
            dawn_agent: The DAWN agent to expose via MCP
            server_name: Name for the MCP server
        """
        self.dawn_agent = dawn_agent
        self.server_name = server_name
        self.mcp = FastMCP(server_name)
        self.registered_tools: Dict[str, Dict[str, Any]] = {}
        
        # Register DAWN capabilities as MCP tools
        self._register_dawn_capabilities()
        
        # Register agent info as MCP resources
        self._register_agent_resources()
        
    def _register_dawn_capabilities(self) -> None:
        """
        Register DAWN agent capabilities as MCP tools.
        
        Each DAWN capability becomes an MCP tool that can be discovered and invoked
        by MCP clients.
        """
        agent_info = self.dawn_agent.get_info()
        capabilities = self.dawn_agent.get_capabilities()
        
        logger.info(f"Registering {len(capabilities)} capabilities as MCP tools for agent {agent_info.get('name', 'unknown')}")
        
        for capability in capabilities:
            self._create_mcp_tool_from_capability(capability)
            
    def _create_mcp_tool_from_capability(self, capability: Dict[str, Any]) -> None:
        """
        Convert a DAWN capability to an MCP tool and register it.
        
        Args:
            capability: DAWN capability dictionary containing id, name, description, etc.
        """
        capability_id = capability['id']
        capability_name = capability.get('name', capability_id)
        capability_description = capability.get('description', f"Tool for {capability_name}")
        
        # Create a sanitized function name for MCP
        function_name = self._sanitize_function_name(capability_id)
        
        # Extract input schema from capability parameters
        input_schema = self._extract_input_schema(capability)
        
        # Create the tool function dynamically
        async def tool_function(**kwargs) -> Any:
            """Dynamically created MCP tool function."""
            try:
                logger.info(f"MCP tool {function_name} invoked with args: {kwargs}")
                
                # Invoke the DAWN capability
                result = self.dawn_agent.invoke(capability_id, kwargs)
                
                logger.info(f"MCP tool {function_name} completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"Error in MCP tool {function_name}: {e}")
                return {"error": str(e), "capability_id": capability_id}
        
        # Set function metadata
        tool_function.__name__ = function_name
        tool_function.__doc__ = capability_description
        
        # Add type annotations based on input schema
        if input_schema and 'properties' in input_schema:
            annotations = {}
            for param_name, param_info in input_schema['properties'].items():
                param_type = param_info.get('type', 'str')
                if param_type == 'string':
                    annotations[param_name] = str
                elif param_type == 'integer':
                    annotations[param_name] = int
                elif param_type == 'number':
                    annotations[param_name] = float
                elif param_type == 'boolean':
                    annotations[param_name] = bool
                else:
                    annotations[param_name] = Any
            annotations['return'] = Any
            tool_function.__annotations__ = annotations
        
        # Register the tool with FastMCP
        self.mcp.tool()(tool_function)
        
        # Store tool information
        self.registered_tools[function_name] = {
            "capability_id": capability_id,
            "capability_name": capability_name,
            "function_name": function_name,
            "description": capability_description,
            "input_schema": input_schema
        }
        
        logger.info(f"Registered MCP tool: {function_name} (capability: {capability_id})")
        
    def _sanitize_function_name(self, capability_id: str) -> str:
        """
        Sanitize capability ID to create a valid Python function name.
        
        Args:
            capability_id: The capability ID to sanitize
            
        Returns:
            A valid Python function name
        """
        # Replace hyphens and other invalid characters with underscores
        sanitized = capability_id.replace('-', '_').replace(' ', '_').replace('.', '_')
        
        # Ensure it starts with a letter or underscore
        if not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"tool_{sanitized}"
            
        return sanitized.lower()
        
    def _extract_input_schema(self, capability: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract input schema from DAWN capability parameters.
        
        Args:
            capability: DAWN capability dictionary
            
        Returns:
            JSON Schema for the input parameters
        """
        parameters = capability.get('parameters', {})
        
        if not parameters:
            return None
            
        # Convert DAWN parameters to JSON Schema
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param_info in parameters.items():
            if isinstance(param_info, dict):
                prop = {
                    "type": param_info.get('type', 'string'),
                    "description": param_info.get('description', f"Parameter {param_name}")
                }
                
                # Add enum values if available
                if 'enum' in param_info:
                    prop['enum'] = param_info['enum']
                    
                # Add default value if available
                if 'default' in param_info:
                    prop['default'] = param_info['default']
                    
                schema['properties'][param_name] = prop
                
                # Add to required if specified
                if param_info.get('required', False):
                    schema['required'].append(param_name)
            else:
                # Simple parameter definition
                schema['properties'][param_name] = {
                    "type": "string",
                    "description": f"Parameter {param_name}"
                }
                schema['required'].append(param_name)
                
        return schema
        
    def _register_agent_resources(self) -> None:
        """Register agent information as MCP resources."""
        
        @self.mcp.resource("agent://info")
        def get_agent_info() -> str:
            """Get agent information as a resource."""
            agent_info = self.dawn_agent.get_info()
            return f"Agent Information:\n{agent_info}"
            
        @self.mcp.resource("agent://capabilities")
        def get_agent_capabilities() -> str:
            """Get agent capabilities as a resource."""
            capabilities = self.dawn_agent.get_capabilities()
            cap_summary = []
            for cap in capabilities:
                cap_summary.append(f"- {cap.get('name', cap['id'])}: {cap.get('description', 'No description')}")
            return f"Agent Capabilities:\n" + "\n".join(cap_summary)
            
        @self.mcp.resource("agent://health")
        def get_agent_health() -> str:
            """Get agent health status as a resource."""
            health = self.dawn_agent.health_check()
            return f"Agent Health: {'Healthy' if health else 'Unhealthy'}"
            
    def get_server(self) -> FastMCP:
        """
        Get the underlying FastMCP server instance.
        
        Returns:
            The FastMCP server instance
        """
        return self.mcp
        
    def get_registered_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about registered tools.
        
        Returns:
            Dictionary mapping tool names to their information
        """
        return self.registered_tools.copy()
        
    async def run(self) -> None:
        """Run the MCP server."""
        logger.info(f"Starting MCP server: {self.server_name}")
        await self.mcp.run()


class DawnMCPClient:
    """
    MCP client for DAWN agents to use remote agent tools.
    
    This class allows DAWN agents to discover and invoke MCP tools from remote
    MCP servers, enabling the centralized intelligence pattern where one agent
    uses another's capabilities as tools.
    """
    
    def __init__(self):
        """Initialize the MCP client."""
        self.sessions: Dict[str, ClientSession] = {}
        self.server_info: Dict[str, Dict[str, Any]] = {}
        
    async def connect_to_server(self, server_id: str, server_params: StdioServerParameters) -> bool:
        """
        Connect to an MCP server.
        
        Args:
            server_id: Unique identifier for the server
            server_params: Parameters for connecting to the server
            
        Returns:
            True if connection was successful
        """
        try:
            logger.info(f"Connecting to MCP server: {server_id}")
            
            # Connect using stdio client
            async with stdio_client(server_params) as (read, write):
                session = ClientSession(read, write)
                await session.initialize()
                
                self.sessions[server_id] = session
                self.server_info[server_id] = {
                    "server_params": server_params,
                    "connected": True,
                    "tools": None,
                    "resources": None
                }
                
                logger.info(f"Successfully connected to MCP server: {server_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_id}: {e}")
            return False
            
    async def list_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """
        List available tools on an MCP server.
        
        Args:
            server_id: ID of the connected server
            
        Returns:
            List of tool definitions
        """
        if server_id not in self.sessions:
            raise ValueError(f"Server {server_id} is not connected")
            
        try:
            session = self.sessions[server_id]
            response = await session.list_tools()
            
            tools = []
            for tool in response.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                })
                
            # Cache the tools
            self.server_info[server_id]["tools"] = tools
            
            logger.info(f"Listed {len(tools)} tools from server {server_id}")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools from server {server_id}: {e}")
            raise
            
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            server_id: ID of the connected server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Result from the tool
        """
        if server_id not in self.sessions:
            raise ValueError(f"Server {server_id} is not connected")
            
        try:
            logger.info(f"Calling tool {tool_name} on server {server_id} with args: {arguments}")
            
            session = self.sessions[server_id]
            response = await session.call_tool(tool_name, arguments)
            
            logger.info(f"Tool {tool_name} completed successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on server {server_id}: {e}")
            return {"error": str(e), "tool_name": tool_name, "server_id": server_id}
            
    async def list_resources(self, server_id: str) -> List[Dict[str, Any]]:
        """
        List available resources on an MCP server.
        
        Args:
            server_id: ID of the connected server
            
        Returns:
            List of resource definitions
        """
        if server_id not in self.sessions:
            raise ValueError(f"Server {server_id} is not connected")
            
        try:
            session = self.sessions[server_id]
            response = await session.list_resources()
            
            resources = []
            for resource in response.resources:
                resources.append({
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                })
                
            # Cache the resources
            self.server_info[server_id]["resources"] = resources
            
            logger.info(f"Listed {len(resources)} resources from server {server_id}")
            return resources
            
        except Exception as e:
            logger.error(f"Failed to list resources from server {server_id}: {e}")
            raise
            
    async def read_resource(self, server_id: str, resource_uri: str) -> Any:
        """
        Read a resource from an MCP server.
        
        Args:
            server_id: ID of the connected server
            resource_uri: URI of the resource to read
            
        Returns:
            Resource content
        """
        if server_id not in self.sessions:
            raise ValueError(f"Server {server_id} is not connected")
            
        try:
            logger.info(f"Reading resource {resource_uri} from server {server_id}")
            
            session = self.sessions[server_id]
            response = await session.read_resource(resource_uri)
            
            logger.info(f"Resource {resource_uri} read successfully")
            return response.contents
            
        except Exception as e:
            logger.error(f"Failed to read resource {resource_uri} from server {server_id}: {e}")
            return {"error": str(e), "resource_uri": resource_uri, "server_id": server_id}
            
    def get_connected_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about connected servers.
        
        Returns:
            Dictionary mapping server IDs to their information
        """
        return {
            server_id: {
                "connected": info["connected"],
                "tools_count": len(info["tools"]) if info["tools"] else 0,
                "resources_count": len(info["resources"]) if info["resources"] else 0
            }
            for server_id, info in self.server_info.items()
        }
        
    async def disconnect_server(self, server_id: str) -> bool:
        """
        Disconnect from an MCP server.
        
        Args:
            server_id: ID of the server to disconnect from
            
        Returns:
            True if successfully disconnected
        """
        if server_id not in self.sessions:
            return False
            
        try:
            session = self.sessions[server_id]
            await session.close()
            
            del self.sessions[server_id]
            if server_id in self.server_info:
                self.server_info[server_id]["connected"] = False
                
            logger.info(f"Disconnected from MCP server: {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {server_id}: {e}")
            return False