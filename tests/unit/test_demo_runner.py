"""
Unit tests for the demo runner.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from dawn.runners.base import DemoRunner
from dawn.core.agent import Agent
from dawn.protocols.base import ProtocolAdapter, ProtocolEndpoint


class TestDemoRunner:
    """Test the demo runner base class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock(spec=Agent)
        agent.agent_id = "test-agent-123"
        agent.name = "Test Agent"
        agent.process_message = AsyncMock(return_value={
            "response": "Test response",
            "success": True
        })
        agent.initialize = AsyncMock()
        agent.shutdown = AsyncMock()
        return agent
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock protocol adapter."""
        adapter = Mock(spec=ProtocolAdapter)
        adapter.start = AsyncMock(return_value=ProtocolEndpoint(
            protocol="test",
            host="localhost",
            port=8080,
            base_url="http://localhost:8080"
        ))
        adapter.stop = AsyncMock()
        adapter.get_protocol_name = Mock(return_value="test")
        adapter.get_agent_card_for_protocol = Mock(return_value={
            "id": "test-agent",
            "name": "Test Agent",
            "protocol": {"name": "test"}
        })
        return adapter
    
    def test_initialization(self, mock_agent, mock_adapter):
        """Test runner initialization."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        assert runner.agent == mock_agent
        assert runner.adapter == mock_adapter
        assert runner._running is False
    
    def test_initialization_with_configs(self, mock_agent, mock_adapter):
        """Test runner initialization with configurations."""
        runner = DemoRunner(
            mock_agent,
            mock_adapter,
            host="0.0.0.0",
            port=9000,
            show_ui=False
        )
        
        assert runner.host == "0.0.0.0"
        assert runner.port == 9000
        assert runner.show_ui is False
    
    @pytest.mark.asyncio
    async def test_start(self, mock_agent, mock_adapter):
        """Test starting the runner."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        # Mock the run method to avoid infinite loop
        runner._run = AsyncMock()
        
        await runner.start()
        
        # Verify initialization sequence
        mock_agent.initialize.assert_called_once()
        mock_adapter.start.assert_called_once_with(mock_agent, "localhost", 8080)
        runner._run.assert_called_once()
        assert runner._running is True
    
    @pytest.mark.asyncio
    async def test_stop(self, mock_agent, mock_adapter):
        """Test stopping the runner."""
        runner = DemoRunner(mock_agent, mock_adapter)
        runner._running = True
        
        await runner.stop()
        
        assert runner._running is False
        mock_adapter.stop.assert_called_once()
        mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message(self, mock_agent, mock_adapter):
        """Test sending a message to the agent."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        result = await runner.send_message("Test message")
        
        mock_agent.process_message.assert_called_once_with("Test message", None)
        assert result["response"] == "Test response"
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_send_message_with_context(self, mock_agent, mock_adapter):
        """Test sending a message with context."""
        runner = DemoRunner(mock_agent, mock_adapter)
        context = {"key": "value"}
        
        await runner.send_message("Test message", context)
        
        mock_agent.process_message.assert_called_once_with("Test message", context)
    
    def test_display_welcome(self, mock_agent, mock_adapter, capsys):
        """Test welcome message display."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        runner._display_welcome()
        
        captured = capsys.readouterr()
        assert "Welcome to Dawn Agent Demo" in captured.out
        assert "Test Agent" in captured.out
        assert "Protocol: test" in captured.out
    
    def test_display_endpoint_info(self, mock_agent, mock_adapter, capsys):
        """Test endpoint info display."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        endpoint = ProtocolEndpoint(
            protocol="test",
            host="localhost",
            port=8080,
            base_url="http://localhost:8080"
        )
        
        runner._display_endpoint_info(endpoint)
        
        captured = capsys.readouterr()
        assert "Agent running" in captured.out
        assert "http://localhost:8080" in captured.out
    
    def test_display_instructions(self, mock_agent, mock_adapter, capsys):
        """Test instructions display."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        runner._display_instructions()
        
        captured = capsys.readouterr()
        assert "Instructions:" in captured.out
        assert "Type messages" in captured.out
        assert "'quit' to exit" in captured.out
    
    def test_display_agent_card(self, mock_agent, mock_adapter, capsys):
        """Test agent card display."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        runner._display_agent_card()
        
        captured = capsys.readouterr()
        assert "Agent Card:" in captured.out
        mock_adapter.get_agent_card_for_protocol.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_interactive_input(self, mock_agent, mock_adapter):
        """Test handling interactive input."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        # Test normal message
        await runner._handle_interactive_input("Hello agent")
        mock_agent.process_message.assert_called_with("Hello agent", None)
        
        # Test quit
        await runner._handle_interactive_input("quit")
        assert runner._running is False
        
        # Reset
        runner._running = True
        
        # Test exit
        await runner._handle_interactive_input("exit")
        assert runner._running is False
    
    @pytest.mark.asyncio
    async def test_run_with_message(self, mock_agent, mock_adapter, capsys):
        """Test running with a pre-specified message."""
        runner = DemoRunner(mock_agent, mock_adapter, message="Test message")
        
        # Mock adapter start
        runner.adapter.start = AsyncMock(return_value=ProtocolEndpoint(
            protocol="test",
            host="localhost",
            port=8080,
            base_url="http://localhost:8080"
        ))
        
        # Run with message (should process and exit)
        await runner.run()
        
        # Verify message was processed
        mock_agent.initialize.assert_called_once()
        mock_agent.process_message.assert_called_once_with("Test message", None)
        mock_agent.shutdown.assert_called_once()
        
        # Check output
        captured = capsys.readouterr()
        assert "Test response" in captured.out
    
    @pytest.mark.asyncio
    async def test_run_loop_error_handling(self, mock_agent, mock_adapter):
        """Test error handling in the run loop."""
        runner = DemoRunner(mock_agent, mock_adapter)
        runner._running = True
        
        # Mock process_message to raise an error
        mock_agent.process_message = AsyncMock(
            side_effect=Exception("Processing error")
        )
        
        # Mock input to return a message then quit
        with patch('builtins.input', side_effect=["test", "quit"]):
            await runner._run_interactive_loop()
        
        # Should handle error gracefully and continue
        assert not runner._running  # Should exit after "quit"
    
    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_agent, mock_adapter):
        """Test cleanup when exception occurs during start."""
        runner = DemoRunner(mock_agent, mock_adapter)
        
        # Make adapter.start raise an exception
        mock_adapter.start = AsyncMock(side_effect=Exception("Startup error"))
        
        with pytest.raises(Exception, match="Startup error"):
            await runner.start()
        
        # Verify cleanup was attempted
        mock_agent.initialize.assert_called_once()
        # Note: adapter.stop won't be called because start failed 