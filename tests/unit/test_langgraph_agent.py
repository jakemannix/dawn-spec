"""
Unit tests for the LangGraph agent implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, ANY
from typing import List

from langchain_core.tools import BaseTool
from dawn.core.langgraph_agent import LangGraphAgent, AgentState


class MockLangGraphAgentImpl(LangGraphAgent):
    """Test implementation of LangGraphAgent."""
    
    def __init__(self, **kwargs):
        # Mock the LLM initialization
        with patch.object(self, '_initialize_llm'):
            super().__init__(**kwargs)
            # Set a mock LLM after init
            self.llm = Mock()
    
    def get_langgraph_tools(self) -> List[BaseTool]:
        """Return mock tools."""
        tool1 = Mock(spec=BaseTool)
        tool1.name = "test_tool"
        tool1.description = "A test tool"
        
        tool2 = Mock(spec=BaseTool)
        tool2.name = "another_tool"
        tool2.description = "Another test tool"
        
        return [tool1, tool2]
    
    def get_capabilities(self) -> List[dict]:
        """Return test capabilities."""
        return [
            {"id": "cap1", "name": "Capability 1"},
            {"id": "cap2", "name": "Capability 2"}
        ]


class TestLangGraphAgent:
    """Test the LangGraph agent implementation."""
    
    def test_initialization_without_llm_keys(self, env_without_llm_keys):
        """Test that initialization fails without LLM keys."""
        with pytest.raises(RuntimeError, match="No LLM API keys found"):
            # Need to provide a concrete implementation since LangGraphAgent is abstract
            class TestAgent(LangGraphAgent):
                def get_capabilities(self):
                    return []
            TestAgent()
    
    def test_initialization_with_openai(self, env_with_mock_keys, monkeypatch):
        """Test initialization with OpenAI."""
        mock_openai = Mock()
        monkeypatch.setattr("dawn.core.langgraph_agent.ChatOpenAI", mock_openai)
        
        class TestAgent(LangGraphAgent):
            def get_capabilities(self):
                return []
        
        agent = TestAgent()
        mock_openai.assert_called_once()
        assert agent.llm is not None
    
    def test_initialization_with_anthropic(self, monkeypatch):
        """Test initialization with Anthropic."""
        # Remove OpenAI key, keep Anthropic
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key")
        
        mock_anthropic = Mock()
        monkeypatch.setattr("dawn.core.langgraph_agent.ChatAnthropic", mock_anthropic)
        
        class TestAgent(LangGraphAgent):
            def get_capabilities(self):
                return []
        
        agent = TestAgent()
        mock_anthropic.assert_called_once()
    
    def test_initialization_with_google(self, monkeypatch):
        """Test initialization with Google."""
        # Remove other keys, keep Google
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")
        
        mock_google = Mock()
        monkeypatch.setattr("dawn.core.langgraph_agent.ChatGoogleGenerativeAI", mock_google)
        
        class TestAgent(LangGraphAgent):
            def get_capabilities(self):
                return []
        
        agent = TestAgent()
        mock_google.assert_called_once()
    
    def test_get_tools_default(self, env_with_mock_keys):
        """Test that default implementation returns empty list."""
        with patch.object(LangGraphAgent, '_initialize_llm'):
            class TestAgent(LangGraphAgent):
                def get_capabilities(self):
                    return []
            agent = TestAgent()
            assert agent.get_langgraph_tools() == []
    
    def test_get_tools_names(self):
        """Test get_tools returns tool names."""
        agent = MockLangGraphAgentImpl()
        tools = agent.get_tools()
        
        assert len(tools) == 2
        assert "test_tool" in tools
        assert "another_tool" in tools
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, mock_llm):
        """Test successful message processing."""
        agent = MockLangGraphAgentImpl()
        agent.llm = mock_llm
        
        # Mock the graph execution
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "messages": [Mock(content="This is the response")],
            "tools_used": ["test_tool"],
            "reasoning_trace": ["Step 1", "Step 2"]
        })
        
        with patch.object(agent, 'build_graph', return_value=mock_graph):
            result = await agent.process_message("Test message")
        
        assert result["success"] is True
        assert result["response"] == "This is the response"
        assert "test_tool" in result["tools_used"]
        assert len(result["reasoning_trace"]) == 2
    
    @pytest.mark.asyncio
    async def test_process_message_with_context(self, mock_llm):
        """Test message processing with context."""
        agent = MockLangGraphAgentImpl()
        agent.llm = mock_llm
        
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "messages": [Mock(content="Response with context")],
            "tools_used": [],
            "reasoning_trace": []
        })
        
        with patch.object(agent, 'build_graph', return_value=mock_graph):
            context = {"previous": "conversation"}
            result = await agent.process_message("Test", context)
            
            # Verify context was passed to graph
            call_args = mock_graph.ainvoke.call_args[0][0]
            assert call_args["context"] == context
    
    @pytest.mark.asyncio
    async def test_process_message_error(self, mock_llm):
        """Test error handling in message processing."""
        agent = MockLangGraphAgentImpl()
        agent.llm = mock_llm
        
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        with patch.object(agent, 'build_graph', return_value=mock_graph):
            result = await agent.process_message("Test message")
        
        assert result["success"] is False
        assert "Test error" in result["response"]
        assert "Test error" in result["error"]
    
    def test_build_graph(self, mock_llm):
        """Test graph building."""
        agent = MockLangGraphAgentImpl()
        agent.llm = mock_llm
        
        # Mock the StateGraph and its methods
        with patch("dawn.core.langgraph_agent.StateGraph") as mock_state_graph:
            mock_workflow = Mock()
            mock_state_graph.return_value = mock_workflow
            mock_workflow.compile.return_value = Mock()
            
            # Mock ToolNode to avoid recursion issues with mock tools
            with patch("dawn.core.langgraph_agent.ToolNode") as mock_tool_node:
                graph = agent.build_graph()
            
            # Verify graph construction
            mock_state_graph.assert_called_once_with(AgentState)
            mock_workflow.add_node.assert_any_call("agent", ANY)
            mock_workflow.add_node.assert_any_call("tools", mock_tool_node.return_value)
            mock_workflow.compile.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_method(self, mock_llm):
        """Test the chat convenience method."""
        agent = MockLangGraphAgentImpl()
        agent.llm = mock_llm
        
        # Mock process_message
        with patch.object(agent, 'process_message', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"response": "Chat response", "success": True}
            
            result = await agent.chat("Hello")
            
            mock_process.assert_called_once_with("Hello")
            assert result["response"] == "Chat response" 