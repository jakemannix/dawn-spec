# src/implementations/langgraph_agent.py

from typing import Dict, Any, List, Optional, Sequence
from typing_extensions import Annotated, TypedDict
import asyncio
from datetime import datetime

from ..agent_core import MCPIntegratedAgent, AgentContext, AgentResponse, AgentImplementationType

# Try to import LangGraph dependencies
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import BaseTool, tool
    from langchain_core.runnables import RunnableConfig
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        """State for the LangGraph agent"""
        messages: Annotated[Sequence[BaseMessage], add_messages]
        context: Optional[AgentContext]
        tools_used: List[str]
        skills_invoked: List[str]
        reasoning_trace: List[str]

    class LangGraphAgent(MCPIntegratedAgent):
        """
        LangGraph-based agent implementation with MCP tool integration.
        Supports both simple ReACT patterns and complex multi-agent workflows.
        """
        
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.llm_config = config.get('llm', {})
            self.graph_config = config.get('graph', {})
            self.llm = None
            self.graph = None
            self.tools = []
            
        @property
        def implementation_type(self) -> AgentImplementationType:
            return AgentImplementationType.LANGGRAPH
        
        async def initialize(self) -> None:
            """Initialize the LangGraph agent"""
            self.logger.info("Initializing LangGraphAgent")
            await self._initialize_llm()
            self.logger.info("LangGraphAgent initialized successfully")
        
        async def shutdown(self) -> None:
            """Shutdown the LangGraph agent"""
            self.logger.info("Shutting down LangGraphAgent")
        
        async def process_request(self, context: AgentContext) -> AgentResponse:
            """Process request using LangGraph with MCP integration"""
            
            # Initialize graph with current context (tools/skills)
            await self._initialize_graph(context)
            
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=context.user_message)],
                "context": context,
                "tools_used": [],
                "skills_invoked": [],
                "reasoning_trace": []
            }
            
            try:
                # Run the graph
                config = RunnableConfig(
                    metadata={
                        "session_id": context.session_id,
                        "agent_type": "langgraph"
                    }
                )
                
                self.logger.info(f"Starting LangGraph execution for query: {context.user_message[:100]}...")
                final_state = await self.graph.ainvoke(initial_state, config)
                self.logger.info("LangGraph execution completed successfully")
                
                # Extract response from final state
                messages = final_state["messages"]
                last_message = messages[-1] if messages else None
                
                response_text = last_message.content if last_message else "No response generated"
                
                # Validate we got a proper response
                if not response_text or response_text.strip() == "":
                    self.logger.warning("LangGraph produced empty response")
                    response_text = "I processed your request but didn't generate a response. Please try rephrasing your question."
                
                return AgentResponse(
                    response_text=response_text,
                    tools_used=final_state.get("tools_used", []),
                    skills_invoked=final_state.get("skills_invoked", []),
                    reasoning_trace=final_state.get("reasoning_trace", []),
                    confidence_score=self._calculate_confidence(final_state),
                    metadata={
                        "llm_model": self.llm_config.get("model", "unknown"),
                        "messages_count": len(messages),
                        "graph_type": self.graph_config.get("type", "react"),
                        "execution_status": "success"
                    }
                )
                
            except Exception as e:
                self.logger.error(f"LangGraph processing failed: {e}", exc_info=True)
                return AgentResponse(
                    response_text=f"I encountered an error processing your request: {str(e)}",
                    reasoning_trace=[f"LangGraph execution failed: {e}"],
                    confidence_score=0.0,
                    metadata={
                        "execution_status": "error",
                        "error_type": type(e).__name__
                    }
                )
        
        async def _initialize_llm(self) -> None:
            """Initialize the LLM based on configuration"""
            import os
            
            provider = self.llm_config.get("provider", "openai")
            model = self.llm_config.get("model", os.getenv("DEFAULT_MODEL", "gpt-4"))
            temperature = self.llm_config.get("temperature", float(os.getenv("TEMPERATURE", "0.7")))
            max_tokens = self.llm_config.get("max_tokens", int(os.getenv("MAX_TOKENS", "1000")))
            
            if provider == "openai":
                api_key = self.llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.logger.warning("OpenAI API key not found. LangGraph agent will be limited to text-only mode.")
                    # Create a dummy LLM for now - it will fail when actually used
                    api_key = "dummy-key"
                
                self.llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    max_tokens=max_tokens
                )
            elif provider == "anthropic":
                api_key = self.llm_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    self.logger.warning("Anthropic API key not found. LangGraph agent will be limited to text-only mode.")
                    # Create a dummy LLM for now - it will fail when actually used
                    api_key = "dummy-key"
                
                self.llm = ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    max_tokens=max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        
        async def _initialize_graph(self, context: AgentContext) -> None:
            """Initialize LangGraph with tools and skills from context"""
            
            # Convert MCP tools to LangChain tools
            self.tools = []
            for mcp_tool in context.available_tools:
                langchain_tool = self._convert_mcp_to_langchain_tool(mcp_tool, context)
                self.tools.append(langchain_tool)
            
            # Convert A2A skills to LangChain tools
            for a2a_skill in context.available_skills:
                langchain_tool = self._convert_a2a_to_langchain_tool(a2a_skill, context)
                self.tools.append(langchain_tool)
            
            # Create the graph based on configuration
            graph_type = self.graph_config.get("type", "react")
            
            if graph_type == "react":
                self.graph = self._create_react_graph()
            else:
                # Simplified version - just do ReACT for now
                self.graph = self._create_react_graph()
        
        def _convert_mcp_to_langchain_tool(self, mcp_tool, context: AgentContext) -> BaseTool:
            """Convert MCP tool to LangChain tool"""
            
            @tool
            async def mcp_tool_wrapper(**kwargs) -> str:
                """Wrapper for MCP tool"""
                try:
                    result = await self.call_mcp_tool(context, mcp_tool.name, kwargs)
                    return str(result)
                except Exception as e:
                    return f"Error calling {mcp_tool.name}: {e}"
            
            # Set the tool name and description manually
            mcp_tool_wrapper.name = mcp_tool.name
            mcp_tool_wrapper.description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
            
            return mcp_tool_wrapper
        
        def _convert_a2a_to_langchain_tool(self, a2a_skill, context: AgentContext) -> BaseTool:
            """Convert A2A skill to LangChain tool"""
            
            @tool
            async def a2a_skill_wrapper(**kwargs) -> str:
                """Wrapper for A2A skill"""
                try:
                    result = await self.invoke_a2a_skill(context, a2a_skill.id, kwargs)
                    return str(result)
                except Exception as e:
                    return f"Error invoking {a2a_skill.id}: {e}"
            
            # Set the tool name and description manually
            a2a_skill_wrapper.name = f"skill_{a2a_skill.id.replace('.', '_')}"
            a2a_skill_wrapper.description = f"A2A skill: {a2a_skill.description}"
            
            return a2a_skill_wrapper
        
        def _create_react_graph(self) -> StateGraph:
            """Create a simplified ReACT-style graph"""
            
            # Bind tools to the LLM
            llm_with_tools = self.llm.bind_tools(self.tools)
            
            # Define the agent node
            async def agent_node(state: AgentState) -> AgentState:
                messages = state["messages"]
                reasoning_trace = state.get("reasoning_trace", [])
                
                reasoning_trace.append(f"Agent thinking at {datetime.now().isoformat()}")
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                return {
                    "messages": messages,
                    "reasoning_trace": reasoning_trace,
                    "tools_used": state.get("tools_used", []),
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
                path_map=["tools", "__end__"]
            )
            workflow.add_edge("tools", "agent")
            
            return workflow.compile()
        
        def _calculate_confidence(self, final_state: Dict[str, Any]) -> float:
            """Calculate confidence score based on execution results"""
            tools_used = final_state.get("tools_used", [])
            reasoning_trace = final_state.get("reasoning_trace", [])
            
            # Simple heuristic-based confidence calculation
            confidence = 0.5  # Base confidence
            
            if tools_used:
                confidence += 0.3
            
            if len(reasoning_trace) > 3:
                confidence += 0.2
            
            return min(1.0, max(0.0, confidence))

else:
    # Fallback implementation when LangGraph is not available
    class LangGraphAgent(MCPIntegratedAgent):
        """Stub implementation when LangGraph is not available"""
        
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
        
        @property
        def implementation_type(self) -> AgentImplementationType:
            return AgentImplementationType.LANGGRAPH
        
        async def initialize(self) -> None:
            raise ImportError("LangGraph is not installed. Please install it with: uv pip install -e .[langgraph]")
        
        async def shutdown(self) -> None:
            pass
        
        async def process_request(self, context: AgentContext) -> AgentResponse:
            return AgentResponse(
                response_text="LangGraph is not available. Please install it with: uv pip install -e .[langgraph]",
                confidence_score=0.0
            )


__all__ = ["LangGraphAgent"] 