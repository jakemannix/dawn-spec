"""
LangGraph Agent Implementation

This module provides a base agent implementation using LangGraph for
ReACT-style reasoning with tool usage.
"""

import os
from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dawn.core.agent import Agent
from dawn.utils.logging import get_logger, AgentContext


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools_used: List[str]
    skills_invoked: List[str]
    reasoning_trace: List[str]
    context: Optional[Dict[str, Any]]


class LangGraphAgent(Agent):
    """
    Base agent implementation using LangGraph for reasoning.
    
    This provides ReACT-style reasoning with tool usage, multi-LLM support,
    and integration with the Dawn framework.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "LangGraphAgent",
        description: str = "A LangGraph-powered agent",
        version: str = "1.0.0",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the LangGraph agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            version: Version of the agent implementation
            model: LLM model to use (defaults to env var or provider default)
            temperature: LLM temperature setting
            max_tokens: Maximum tokens for LLM responses
        """
        super().__init__(agent_id, name, description, version)
        
        # LLM configuration
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # LangGraph components
        self.llm = None
        self.graph = None
        self.tools = []
        
        # Initialize LLM
        self._initialize_llm()
        
    def _initialize_llm(self) -> None:
        """Initialize LLM with fallback support for multiple providers."""
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model=self.model or os.getenv("OPENAI_MODEL", "gpt-4"),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self.logger.info(f"Initialized OpenAI LLM (model: {self.llm.model_name})")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Try Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.llm = ChatAnthropic(
                    model=self.model or os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self.logger.info(f"Initialized Anthropic LLM (model: {self.llm.model})")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic: {e}")
        
        # Try Google
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model or os.getenv("GOOGLE_MODEL", "gemini-pro"),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self.logger.info(f"Initialized Google Gemini LLM (model: {self.llm.model})")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google: {e}")
        
        # No LLM available
        raise RuntimeError(
            "No LLM API keys found! Please set one of:\n"
            "- OPENAI_API_KEY for OpenAI\n"
            "- ANTHROPIC_API_KEY for Anthropic\n"
            "- GOOGLE_API_KEY for Google Gemini\n"
            "in your .env file or environment variables."
        )
    
    def get_langgraph_tools(self) -> List[BaseTool]:
        """
        Return the list of LangChain tools available to this agent.
        
        Subclasses should override this method to provide their tools.
        
        Returns:
            List of LangChain BaseTool instances
        """
        return []
    
    def build_graph(self) -> StateGraph:
        """Build the LangGraph ReACT graph."""
        # Get tools from subclass
        self.tools = self.get_langgraph_tools()
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
        
        # Define the agent node
        async def agent_node(state: AgentState) -> AgentState:
            messages = state["messages"]
            reasoning_trace = state.get("reasoning_trace", [])
            
            # Add reasoning step
            reasoning_trace.append(f"Processing message: {messages[-1].content[:100]}...")
            
            # Log LLM request (trace level)
            with AgentContext(self.agent_id):
                self.logger.llm_request(
                    model=str(self.llm.model_name if hasattr(self.llm, 'model_name') else self.llm.model),
                    messages=[{"role": m.type, "content": m.content} for m in messages],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            # Get LLM response
            response = await llm_with_tools.ainvoke(messages)
            
            # Track tool usage
            tools_used = state.get("tools_used", [])
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call["name"])
                    reasoning_trace.append(f"Calling tool: {tool_call['name']}")
            
            # Log LLM response (trace level)
            with AgentContext(self.agent_id):
                self.logger.llm_response(
                    model=str(self.llm.model_name if hasattr(self.llm, 'model_name') else self.llm.model),
                    response=response.content,
                    tokens_used={"prompt": 0, "completion": 0, "total": 0},  # Would need token counting
                    duration_ms=0  # Would need timing
                )
            
            return {
                "messages": [response],
                "reasoning_trace": reasoning_trace,
                "tools_used": tools_used,
                "skills_invoked": state.get("skills_invoked", [])
            }
        
        # Build the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        
        if self.tools:
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
        else:
            workflow.add_edge(START, "agent")
            workflow.add_edge("agent", END)
        
        return workflow.compile()
    
    async def process_message(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a message using the LangGraph ReACT pattern.
        
        Args:
            message: The input message to process
            context: Optional context information
            
        Returns:
            Dict containing the response and metadata
        """
        # Build graph if not already built
        if not self.graph:
            self.graph = self.build_graph()
        
        # Use agent context for all logging
        with AgentContext(self.agent_id):
            # Log received message
            self.logger.message_received(message, source="user")
            
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
                
                # Log sent message
                self.logger.message_sent(response_text, destination="user")
                
                return {
                    "response": response_text,
                    "tools_used": final_state.get("tools_used", []),
                    "skills_invoked": final_state.get("skills_invoked", []),
                    "reasoning_trace": final_state.get("reasoning_trace", []),
                    "success": True
                }
                
            except Exception as e:
                self.logger.error(f"Error processing message: {e}", exception=e)
                return {
                    "response": f"Error: {str(e)}",
                    "success": False,
                    "error": str(e)
                }
    
    def get_tools(self) -> List[str]:
        """Return the names of tools available to this agent."""
        return [tool.name for tool in self.get_langgraph_tools()]
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """
        Convenience method for chat-style interaction.
        
        This is a wrapper around process_message for backward compatibility.
        
        Args:
            message: The input message
            
        Returns:
            Dict containing the response
        """
        return await self.process_message(message) 