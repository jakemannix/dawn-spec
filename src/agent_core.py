# src/agent_core.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

# Preserve existing MCP and A2A imports
from .mcp_integration import MCPToolRegistry, MCPTool
from .a2a_integration_v2 import A2ASkill, A2AMessage


class AgentImplementationType(Enum):
    """Available agent implementation types"""
    TEXT_MATCHING = "text_matching"
    LANGGRAPH = "langgraph" 
    OPENAI_SDK = "openai_sdk"
    AUTOGEN = "autogen"
    CUSTOM = "custom"


@dataclass
class AgentContext:
    """Shared context across all agent implementations"""
    user_message: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    available_tools: List[MCPTool]
    available_skills: List[A2ASkill]
    agent_capabilities: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class AgentResponse:
    """Standardized response from any agent implementation"""
    response_text: str
    tools_used: List[str] = None
    skills_invoked: List[str] = None
    reasoning_trace: List[str] = None
    confidence_score: float = None
    next_actions: List[str] = None
    metadata: Dict[str, Any] = None


class AgentImplementation(ABC):
    """Abstract base class for all agent implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    async def process_request(self, context: AgentContext) -> AgentResponse:
        """Process a user request and return a response"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent implementation"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of the agent implementation"""
        pass
    
    @property
    @abstractmethod
    def implementation_type(self) -> AgentImplementationType:
        """Return the implementation type"""
        pass


class MCPToolProtocol(Protocol):
    """Protocol ensuring MCP tools work across all implementations"""
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool with given arguments"""
        ...
    
    def get_available_tools(self) -> List[MCPTool]:
        """Get list of available MCP tools"""
        ...


class AgentOrchestrator:
    """Main orchestrator that manages different agent implementations"""
    
    def __init__(self):
        self._implementations: Dict[AgentImplementationType, AgentImplementation] = {}
        self._active_implementation: AgentImplementationType = AgentImplementationType.TEXT_MATCHING
        self._mcp_registry = MCPToolRegistry()
        self._a2a_skills: List[A2ASkill] = []
        self.logger = logging.getLogger(__name__)
        
    async def register_implementation(
        self, 
        impl_type: AgentImplementationType, 
        implementation: AgentImplementation
    ):
        """Register a new agent implementation"""
        await implementation.initialize()
        self._implementations[impl_type] = implementation
        self.logger.info(f"Registered agent implementation: {impl_type.value}")
    
    async def set_active_implementation(self, impl_type: AgentImplementationType):
        """Switch to a different agent implementation"""
        if impl_type not in self._implementations:
            raise ValueError(f"Implementation {impl_type.value} not registered")
        
        old_impl = self._active_implementation
        self._active_implementation = impl_type
        self.logger.info(f"Switched agent implementation: {old_impl.value} -> {impl_type.value}")
    
    async def process_request(
        self, 
        user_message: str, 
        session_id: str = "default",
        conversation_history: List[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process a request using the active implementation"""
        
        # Build unified context
        context = AgentContext(
            user_message=user_message,
            session_id=session_id,
            conversation_history=conversation_history or [],
            available_tools=self._mcp_registry.get_all_tools(),
            available_skills=self._a2a_skills,
            agent_capabilities=self._get_agent_capabilities()
        )
        
        # Delegate to active implementation
        implementation = self._implementations[self._active_implementation]
        response = await implementation.process_request(context)
        
        # Add orchestrator metadata
        response.metadata = response.metadata or {}
        response.metadata["implementation_used"] = self._active_implementation.value
        response.metadata["session_id"] = session_id
        
        return response
    
    def register_mcp_tool(self, tool: MCPTool):
        """Register an MCP tool (available to all implementations)"""
        self._mcp_registry.register_tool(tool)
    
    def register_a2a_skill(self, skill: A2ASkill):
        """Register an A2A skill (available to all implementations)"""
        self._a2a_skills.append(skill)
    
    def _get_agent_capabilities(self) -> Dict[str, Any]:
        """Get current agent capabilities"""
        return {
            "tools": [tool.name for tool in self._mcp_registry.get_all_tools()],
            "skills": [skill.id for skill in self._a2a_skills],
            "active_implementation": self._active_implementation.value
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all implementations"""
        status = {
            "active_implementation": self._active_implementation.value,
            "implementations": {}
        }
        
        for impl_type, implementation in self._implementations.items():
            try:
                # Basic health check - could be expanded
                status["implementations"][impl_type.value] = {
                    "status": "healthy",
                    "type": implementation.implementation_type.value
                }
            except Exception as e:
                status["implementations"][impl_type.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return status


# Utility base class for MCP integration
class MCPIntegratedAgent(AgentImplementation):
    """Base class that provides MCP integration helpers"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._mcp_cache = {}
    
    async def call_mcp_tool(self, context: AgentContext, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Helper to call MCP tools from any implementation"""
        for tool in context.available_tools:
            if tool.name == tool_name:
                try:
                    result = await tool.call(arguments)
                    self.logger.info(f"MCP tool {tool_name} called successfully")
                    return result
                except Exception as e:
                    self.logger.error(f"MCP tool {tool_name} failed: {e}")
                    raise
        
        raise ValueError(f"MCP tool {tool_name} not found")
    
    async def invoke_a2a_skill(self, context: AgentContext, skill_id: str, inputs: Dict[str, Any]) -> Any:
        """Helper to invoke A2A skills from any implementation"""
        for skill in context.available_skills:
            if skill.id == skill_id:
                try:
                    if skill.handler:
                        result = await skill.handler(inputs)
                    else:
                        result = f"Mock result for skill {skill_id} with inputs: {inputs}"
                    self.logger.info(f"A2A skill {skill_id} invoked successfully")
                    return result
                except Exception as e:
                    self.logger.error(f"A2A skill {skill_id} failed: {e}")
                    raise
        
        raise ValueError(f"A2A skill {skill_id} not found")


# Export the main components
__all__ = [
    'AgentOrchestrator',
    'AgentImplementation', 
    'MCPIntegratedAgent',
    'AgentContext',
    'AgentResponse',
    'AgentImplementationType'
] 