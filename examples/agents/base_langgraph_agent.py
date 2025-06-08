#!/usr/bin/env python3
"""
Base Multi-Protocol LangGraph Agent

This base class provides a foundation for agents that support:
- Google A2A (Agent-to-Agent) protocol
- AGNTCY ACP (Agent Connect Protocol)
- MCP (Model Context Protocol)
- LangGraph for ReACT reasoning
"""

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type, Union
from typing_extensions import Annotated, TypedDict

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import BaseTool, tool
    from langchain_core.runnables import RunnableConfig
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Install with: uv pip install -e '.[langgraph]'")

# Protocol imports
try:
    from a2a import A2AServer, A2AClient, A2ATask
    from a2a.types import A2AMessage
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    print("Warning: A2A SDK not available. Install with: uv pip install -e '.[interop]'")

try:
    from agntcy_acp import AsyncACPClient, ApiClientConfiguration
    from agntcy_acp.models import RunCreate, AgentInfo
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False
    print("Warning: AGNTCY ACP not available. Install with: uv pip install -e '.[interop]'")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP not available. Install with: uv pip install -e '.[interop]'")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools_used: List[str]
    skills_invoked: List[str]
    reasoning_trace: List[str]
    context: Optional[Dict[str, Any]]


class MultiProtocolLangGraphAgent(ABC):
    """
    Base class for agents that support multiple protocols and use LangGraph for reasoning.
    
    This class provides:
    - LangGraph ReACT implementation
    - Multi-LLM support (OpenAI, Anthropic, Google)
    - Protocol adapters for A2A, ACP, and MCP
    - Unified agent card generation
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "MultiProtocolAgent",
        description: str = "A multi-protocol agent",
        version: str = "1.0.0",
        port: int = 8080,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version
        self.port = port
        
        # LangGraph components
        self.llm = None
        self.graph = None
        self.tools = []
        
        # Protocol servers
        self.a2a_server = None
        self.acp_client = None
        self.mcp_session = None
        
        # Initialize LLM
        self._initialize_llm()
        
    def _initialize_llm(self) -> None:
        """Initialize LLM with fallback support for multiple providers."""
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model=os.getenv("DEFAULT_MODEL", "gpt-4"),
                    temperature=float(os.getenv("TEMPERATURE", "0.7")),
                    max_tokens=int(os.getenv("MAX_TOKENS", "1000"))
                )
                logger.info("Initialized OpenAI LLM")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Try Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.llm = ChatAnthropic(
                    model=os.getenv("DEFAULT_MODEL", "claude-3-opus-20240229"),
                    temperature=float(os.getenv("TEMPERATURE", "0.7")),
                    max_tokens=int(os.getenv("MAX_TOKENS", "1000"))
                )
                logger.info("Initialized Anthropic LLM")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {e}")
        
        # Try Google
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=os.getenv("DEFAULT_MODEL", "gemini-pro"),
                    temperature=float(os.getenv("TEMPERATURE", "0.7")),
                    max_tokens=int(os.getenv("MAX_TOKENS", "1000"))
                )
                logger.info("Initialized Google Gemini LLM")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Google: {e}")
        
        # No LLM available
        raise RuntimeError(
            "No LLM API keys found! Please set one of:\n"
            "- OPENAI_API_KEY for OpenAI\n"
            "- ANTHROPIC_API_KEY for Anthropic\n"
            "- GOOGLE_API_KEY for Google Gemini\n"
            "in your .env file or environment variables."
        )
    
    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return the list of tools available to this agent."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return OASF capability definitions for this agent."""
        pass
    
    def build_graph(self) -> StateGraph:
        """Build the LangGraph ReACT graph."""
        # Get tools from subclass
        self.tools = self.get_tools()
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Define the agent node
        async def agent_node(state: AgentState) -> AgentState:
            messages = state["messages"]
            reasoning_trace = state.get("reasoning_trace", [])
            
            # Add reasoning step
            reasoning_trace.append(f"Processing message: {messages[-1].content[:100]}...")
            
            # Get LLM response
            response = await llm_with_tools.ainvoke(messages)
            
            # Track tool usage
            tools_used = state.get("tools_used", [])
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call["name"])
                    reasoning_trace.append(f"Calling tool: {tool_call['name']}")
            
            return {
                "messages": [response],
                "reasoning_trace": reasoning_trace,
                "tools_used": tools_used,
                "skills_invoked": state.get("skills_invoked", [])
            }
        
        # Build the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message using the LangGraph ReACT pattern."""
        if not self.graph:
            self.graph = self.build_graph()
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "tools_used": [],
            "skills_invoked": [],
            "reasoning_trace": [],
            "context": context or {}
        }
        
        # Run the graph
        config = RunnableConfig(
            metadata={
                "agent_id": self.agent_id,
                "agent_name": self.name
            }
        )
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Extract response
            messages = final_state["messages"]
            last_message = messages[-1] if messages else None
            response_text = last_message.content if last_message else "No response generated"
            
            return {
                "response": response_text,
                "tools_used": final_state.get("tools_used", []),
                "skills_invoked": final_state.get("skills_invoked", []),
                "reasoning_trace": final_state.get("reasoning_trace", []),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Generate a unified agent card compatible with all protocols."""
        # Get tools directly (don't rely on self.tools which may not be initialized)
        tools = self.get_tools()
        
        agent_card = {
            # Basic metadata
            "id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            
            # AGNTCY ACP fields
            "acp": {
                "endpoint": f"http://localhost:{self.port}/acp",
                "protocol_version": "1.0"
            },
            
            # Google A2A fields
            "a2a": {
                "server": {
                    "url": f"http://localhost:{self.port}",
                    "protocol": "a2a"
                }
            },
            
            # MCP compatibility
            "mcp": {
                "tools": [tool.name for tool in tools],
                "resources": []
            },
            
            # OASF capabilities
            "capabilities": self.get_capabilities(),
            
            # Discovery metadata
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "tags": ["research", "langgraph", "multi-protocol"]
            }
        }
        
        return agent_card
    
    async def start_a2a_server(self) -> None:
        """Start the A2A server for this agent."""
        if not A2A_AVAILABLE:
            logger.warning("A2A SDK not available, skipping A2A server")
            return
        
        # TODO: Implement A2A server setup
        logger.info(f"Starting A2A server on port {self.port}")
    
    async def start_acp_endpoints(self) -> None:
        """Start ACP endpoints for this agent."""
        if not ACP_AVAILABLE:
            logger.warning("ACP SDK not available, skipping ACP endpoints")
            return
        
        # TODO: Implement ACP endpoint setup
        logger.info(f"Starting ACP endpoints on port {self.port}")
    
    async def _start_agent_card_server(self) -> None:
        """Start a simple HTTP server to serve the agent card."""
        try:
            from aiohttp import web
            
            async def serve_agent_card(request):
                """Serve the agent card as JSON."""
                card = self.get_agent_card()
                return web.json_response(card)
            
            async def serve_well_known(request):
                """Serve agent card at /.well-known/agent.json."""
                card = self.get_agent_card()
                return web.json_response(card)
            
            app = web.Application()
            app.router.add_get('/agent.json', serve_agent_card)
            app.router.add_get('/.well-known/agent.json', serve_well_known)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            logger.info(f"Agent card server started at http://localhost:{self.port}/agent.json")
            
        except ImportError:
            logger.warning("aiohttp not available, agent card server not started")
        except Exception as e:
            logger.warning(f"Failed to start agent card server: {e}")
    
    async def register_mcp_tools(self) -> None:
        """Register tools with MCP."""
        if not MCP_AVAILABLE:
            logger.warning("MCP SDK not available, skipping MCP registration")
            return
        
        # TODO: Implement MCP tool registration
        logger.info("Registering MCP tools")
    
    async def start_all_protocols(self) -> None:
        """Start all protocol servers and endpoints."""
        # Always start agent card server for discovery
        await self._start_agent_card_server()
        
        await asyncio.gather(
            self.start_a2a_server(),
            self.start_acp_endpoints(),
            self.register_mcp_tools()
        )
    
    async def stop_all_protocols(self) -> None:
        """Stop all protocol servers and endpoints."""
        # TODO: Implement cleanup
        logger.info("Stopping all protocol servers") 