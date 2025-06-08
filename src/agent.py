"""
Implementation of agents based on AGNTCY's framework specification.

This module implements the core agent components following the AGNTCY framework,
including the Agent Connect Protocol (ACP) and Agent Gateway Protocol (AGP) concepts.
"""
from typing import Dict, List, Optional, Any, Union
import uuid
import datetime
import logging

logger = logging.getLogger(__name__)


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
        metadata: Optional[Dict[str, Any]] = None,
        business_logic_schema: Optional[str] = None
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
            business_logic_schema: Optional business logic schema type for payload validation
        """
        self.id = str(uuid.uuid4())
        self.type = capability_type
        self.name = name
        self.description = description
        self.version = version
        self.parameters = parameters or {}
        self.metadata = metadata or {}
        self.business_logic_schema = business_logic_schema
        
        # Initialize schema validator
        self._schema_validator = None
        
    def set_business_logic_schema(self, schema_type: str) -> bool:
        """
        Set business logic schema for payload validation.
        
        Args:
            schema_type: The schema type to use for validation
            
        Returns:
            True if schema was set successfully
        """
        try:
            from .schemas import schema_validator
            
            if schema_validator.get_schema(schema_type):
                self.business_logic_schema = schema_type
                self._schema_validator = schema_validator
                logger.info(f"Set business logic schema '{schema_type}' for capability {self.id}")
                return True
            else:
                logger.error(f"Schema type '{schema_type}' not found")
                return False
        except ImportError:
            logger.error("Schema validation not available")
            return False
            
    def validate_input_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input payload against business logic schema.
        
        Args:
            payload: The payload to validate
            
        Returns:
            Validation result dictionary
        """
        if not self.business_logic_schema:
            return {
                "valid": True,
                "message": "No schema validation configured"
            }
            
        if not self._schema_validator:
            try:
                from .schemas import schema_validator
                self._schema_validator = schema_validator
            except ImportError:
                return {
                    "valid": False,
                    "error": "Schema validation not available"
                }
                
        return self._schema_validator.validate_payload(self.business_logic_schema, payload)
        
    def validate_output_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output payload against business logic schema.
        
        Note: This uses the same schema as input validation. In practice,
        you might want separate input/output schemas.
        
        Args:
            payload: The payload to validate
            
        Returns:
            Validation result dictionary
        """
        return self.validate_input_payload(payload)
        
    def get_schema_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the business logic schema.
        
        Returns:
            Schema information or None if no schema is set
        """
        if not self.business_logic_schema or not self._schema_validator:
            return None
            
        try:
            from .schemas import schema_validator
            schema = schema_validator.get_schema(self.business_logic_schema)
            if schema:
                return {
                    "schema_type": self.business_logic_schema,
                    "schema_id": schema.schema_id,
                    "title": schema.title,
                    "description": schema.description,
                    "version": schema.schema_version
                }
        except ImportError:
            pass
            
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the capability to a dictionary representation.
        
        Returns:
            Dictionary representing the capability in OASF format
        """
        result = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters,
            "metadata": self.metadata
        }
        
        # Add schema information if available
        if self.business_logic_schema:
            result["business_logic_schema"] = self.business_logic_schema
            schema_info = self.get_schema_info()
            if schema_info:
                result["schema_info"] = schema_info
                
        return result


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


class A2ACapableAgent(Agent):
    """
    Enhanced Agent class with A2A protocol capabilities.
    
    This class extends the base Agent with A2A server and client functionality,
    allowing the agent to both expose its capabilities via A2A and communicate
    with other A2A agents.
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
        Initialize an A2A-capable agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose or capabilities
            provider: The entity that provides this agent
            version: Semantic version of the agent
            metadata: Additional metadata about the agent
        """
        super().__init__(name, description, provider, version, metadata)
        
        # A2A integration components (initialized on demand)
        self.a2a_server: Optional['DawnA2AServer'] = None
        self.a2a_client: Optional['DawnA2AClient'] = None
        
    async def start_a2a_server(self, port: int = 8080, host: str = "localhost") -> 'DawnA2AServer':
        """
        Start this agent as an A2A server.
        
        Args:
            port: Port to run the A2A server on
            host: Host to bind the A2A server to
            
        Returns:
            The A2A server instance
        """
        # Import here to avoid circular dependencies
        from .a2a_integration import DawnA2AServer
        
        if self.a2a_server is not None:
            raise RuntimeError("A2A server is already running")
            
        self.a2a_server = DawnA2AServer(self, port, host)
        await self.a2a_server.start()
        return self.a2a_server
        
    async def stop_a2a_server(self) -> None:
        """Stop the A2A server if it's running."""
        if self.a2a_server is not None:
            await self.a2a_server.stop()
            self.a2a_server = None
            
    def get_a2a_client(self) -> 'DawnA2AClient':
        """
        Get the A2A client for communicating with remote agents.
        
        Returns:
            The A2A client instance (creates one if it doesn't exist)
        """
        if self.a2a_client is None:
            # Import here to avoid circular dependencies
            from .a2a_integration import DawnA2AClient
            self.a2a_client = DawnA2AClient()
        return self.a2a_client
        
    async def connect_to_a2a_agent(self, agent_url: str, agent_id: Optional[str] = None) -> str:
        """
        Connect to a remote A2A agent.
        
        Args:
            agent_url: URL of the remote A2A agent
            agent_id: Optional ID for the agent
            
        Returns:
            Agent ID for the connected agent
        """
        client = self.get_a2a_client()
        return await client.connect_to_agent(agent_url, agent_id)
        
    async def invoke_remote_a2a_capability(
        self, 
        agent_id: str, 
        capability_id: str, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a capability on a remote A2A agent.
        
        Args:
            agent_id: ID of the connected remote agent
            capability_id: ID of the capability to invoke
            inputs: Input parameters for the capability
            
        Returns:
            Result from the remote agent
        """
        client = self.get_a2a_client()
        return await client.invoke_remote_capability(agent_id, capability_id, inputs)
        
    async def discover_remote_a2a_capabilities(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Discover capabilities available on a remote A2A agent.
        
        Args:
            agent_id: ID of the connected remote agent
            
        Returns:
            List of capability definitions
        """
        client = self.get_a2a_client()
        return await client.discover_tools(agent_id)
        
    def get_a2a_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the A2A server if it's running.
        
        Returns:
            Server information dictionary or None if server is not running
        """
        if self.a2a_server is not None:
            return self.a2a_server.get_server_info()
        return None
        
    def get_connected_a2a_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about connected A2A agents.
        
        Returns:
            Dictionary mapping agent IDs to their connection info
        """
        if self.a2a_client is not None:
            return self.a2a_client.get_connected_agents()
        return {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation, including A2A info.
        
        Returns:
            Dictionary representing the agent
        """
        result = super().to_dict()
        
        # Add A2A-specific information
        result["a2a_capabilities"] = {
            "server_running": self.a2a_server is not None,
            "server_info": self.get_a2a_server_info(),
            "connected_agents": self.get_connected_a2a_agents()
        }
        
        return result


class MCPCapableAgent(A2ACapableAgent):
    """
    Enhanced Agent class with both A2A and MCP protocol capabilities.
    
    This class extends the A2ACapableAgent with MCP server and client functionality,
    allowing the agent to expose its capabilities as MCP tools and to consume
    MCP tools from other agents.
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
        Initialize an MCP and A2A capable agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose or capabilities
            provider: The entity that provides this agent
            version: Semantic version of the agent
            metadata: Additional metadata about the agent
        """
        super().__init__(name, description, provider, version, metadata)
        
        # MCP integration components (initialized on demand)
        self.mcp_server: Optional['DawnMCPServer'] = None
        self.mcp_client: Optional['DawnMCPClient'] = None
        
    def start_mcp_server(self, server_name: Optional[str] = None) -> 'DawnMCPServer':
        """
        Start this agent as an MCP server.
        
        Args:
            server_name: Optional name for the MCP server
            
        Returns:
            The MCP server instance
        """
        # Import here to avoid circular dependencies
        from .mcp_integration import DawnMCPServer
        
        if self.mcp_server is not None:
            raise RuntimeError("MCP server is already running")
            
        server_name = server_name or f"dawn-{self.name.lower().replace(' ', '-')}"
        self.mcp_server = DawnMCPServer(self, server_name)
        return self.mcp_server
        
    def stop_mcp_server(self) -> None:
        """Stop the MCP server if it's running."""
        if self.mcp_server is not None:
            self.mcp_server = None
            
    def get_mcp_client(self) -> 'DawnMCPClient':
        """
        Get the MCP client for communicating with remote MCP servers.
        
        Returns:
            The MCP client instance (creates one if it doesn't exist)
        """
        if self.mcp_client is None:
            # Import here to avoid circular dependencies
            from .mcp_integration import DawnMCPClient
            self.mcp_client = DawnMCPClient()
        return self.mcp_client
        
    async def connect_to_mcp_server(self, server_id: str, server_params) -> bool:
        """
        Connect to a remote MCP server.
        
        Args:
            server_id: ID for the MCP server
            server_params: MCP server connection parameters
            
        Returns:
            True if connection was successful
        """
        client = self.get_mcp_client()
        return await client.connect_to_server(server_id, server_params)
        
    async def call_mcp_tool(
        self, 
        server_id: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on a remote MCP server.
        
        Args:
            server_id: ID of the connected MCP server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Result from the remote tool
        """
        client = self.get_mcp_client()
        return await client.call_tool(server_id, tool_name, arguments)
        
    async def discover_mcp_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """
        Discover tools available on a remote MCP server.
        
        Args:
            server_id: ID of the connected MCP server
            
        Returns:
            List of tool definitions
        """
        client = self.get_mcp_client()
        return await client.list_tools(server_id)
        
    async def discover_mcp_resources(self, server_id: str) -> List[Dict[str, Any]]:
        """
        Discover resources available on a remote MCP server.
        
        Args:
            server_id: ID of the connected MCP server
            
        Returns:
            List of resource definitions
        """
        client = self.get_mcp_client()
        return await client.list_resources(server_id)
        
    async def read_mcp_resource(self, server_id: str, resource_uri: str) -> Any:
        """
        Read a resource from a remote MCP server.
        
        Args:
            server_id: ID of the connected MCP server
            resource_uri: URI of the resource to read
            
        Returns:
            Resource content
        """
        client = self.get_mcp_client()
        return await client.read_resource(server_id, resource_uri)
        
    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get agent capabilities formatted as MCP tools.
        
        Returns:
            List of MCP tool definitions
        """
        tools = []
        for capability in self.capabilities:
            # Convert capability to MCP tool format
            capability_dict = capability.to_dict()
            tool = {
                "name": capability_dict['id'].replace('-', '_').lower(),
                "description": capability_dict.get('description', f"Tool for {capability_dict.get('name', capability_dict['id'])}"),
                "inputSchema": self._capability_to_mcp_schema(capability_dict)
            }
            tools.append(tool)
        return tools
        
    def _capability_to_mcp_schema(self, capability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a DAWN capability to MCP input schema.
        
        Args:
            capability: DAWN capability dictionary
            
        Returns:
            MCP-compatible input schema
        """
        parameters = capability.get('parameters', {})
        
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
                schema['properties'][param_name] = prop
                
                if param_info.get('required', False):
                    schema['required'].append(param_name)
            else:
                schema['properties'][param_name] = {
                    "type": "string",
                    "description": f"Parameter {param_name}"
                }
                schema['required'].append(param_name)
                
        return schema
        
    def get_mcp_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the MCP server if it's running.
        
        Returns:
            Server information dictionary or None if server is not running
        """
        if self.mcp_server is not None:
            return {
                "server_name": self.mcp_server.server_name,
                "registered_tools": self.mcp_server.get_registered_tools(),
                "tools_count": len(self.mcp_server.get_registered_tools())
            }
        return None
        
    def get_connected_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about connected MCP servers.
        
        Returns:
            Dictionary mapping server IDs to their connection info
        """
        if self.mcp_client is not None:
            return self.mcp_client.get_connected_servers()
        return {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation, including MCP info.
        
        Returns:
            Dictionary representing the agent
        """
        result = super().to_dict()
        
        # Add MCP-specific information
        result["mcp_capabilities"] = {
            "server_running": self.mcp_server is not None,
            "server_info": self.get_mcp_server_info(),
            "connected_servers": self.get_connected_mcp_servers(),
            "available_as_tools": self.get_mcp_tools()
        }
        
        return result