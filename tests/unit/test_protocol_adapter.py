"""
Unit tests for protocol adapters.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from dawn.protocols.base import ProtocolAdapter, ProtocolEndpoint
from dawn.protocols.acp.adapter import ACPAdapter
from dawn.core.agent import Agent


class MockProtocolAdapter(ProtocolAdapter):
    """Test implementation of ProtocolAdapter."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_called = False
        self.stop_called = False
    
    async def start(self, agent, host="localhost", port=8080):
        """Mock start implementation."""
        self.start_called = True
        self.agent = agent
        self._is_running = True
        self.endpoint = ProtocolEndpoint(
            protocol="test",
            host=host,
            port=port,
            base_url=f"http://{host}:{port}"
        )
        return self.endpoint
    
    async def stop(self):
        """Mock stop implementation."""
        self.stop_called = True
        self._is_running = False
        self.endpoint = None
    
    def get_protocol_name(self):
        return "test"
    
    def get_protocol_version(self):
        return "1.0"


class MockProtocolAdapterBase:
    """Test the base ProtocolAdapter class."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = MockProtocolAdapter()
        assert adapter.agent is None
        assert adapter.endpoint is None
        assert not adapter.is_running()
    
    def test_initialization_with_logger(self):
        """Test adapter initialization with custom logger."""
        logger = Mock()
        adapter = MockProtocolAdapter(logger=logger)
        assert adapter.logger == logger
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test start/stop lifecycle."""
        adapter = MockProtocolAdapter()
        agent = Mock(spec=Agent)
        
        # Start adapter
        endpoint = await adapter.start(agent, "localhost", 9000)
        assert adapter.start_called
        assert adapter.is_running()
        assert adapter.agent == agent
        assert endpoint.port == 9000
        assert endpoint.protocol == "test"
        
        # Stop adapter
        await adapter.stop()
        assert adapter.stop_called
        assert not adapter.is_running()
        assert adapter.endpoint is None
    
    def test_get_endpoint(self):
        """Test get_endpoint method."""
        adapter = MockProtocolAdapter()
        
        # Should return None when not running
        assert adapter.get_endpoint() is None
        
        # Should return endpoint when running
        adapter._is_running = True
        adapter.endpoint = ProtocolEndpoint(
            protocol="test",
            host="localhost",
            port=8080,
            base_url="http://localhost:8080"
        )
        endpoint = adapter.get_endpoint()
        assert endpoint is not None
        assert endpoint.port == 8080
    
    @pytest.mark.asyncio
    async def test_handle_request(self):
        """Test default request handling."""
        adapter = MockProtocolAdapter()
        
        # Without agent
        result = await adapter.handle_request({"message": "test"})
        assert result["error"] == "No agent configured"
        assert result["success"] is False
        
        # With agent
        agent = AsyncMock(spec=Agent)
        agent.process_message.return_value = {
            "response": "Test response",
            "success": True
        }
        adapter.agent = agent
        
        result = await adapter.handle_request({
            "message": "test message",
            "context": {"key": "value"}
        })
        
        agent.process_message.assert_called_once_with("test message", {"key": "value"})
        assert result["response"] == "Test response"
        assert result["success"] is True
    
    def test_get_agent_card_for_protocol(self):
        """Test agent card formatting for protocol."""
        adapter = MockProtocolAdapter()
        
        # Without agent
        card = adapter.get_agent_card_for_protocol()
        assert card == {}
        
        # With agent
        agent = Mock(spec=Agent)
        agent.get_agent_card.return_value = {
            "id": "test-agent",
            "name": "Test Agent"
        }
        adapter.agent = agent
        adapter.endpoint = ProtocolEndpoint(
            protocol="test",
            host="localhost",
            port=8080,
            base_url="http://localhost:8080"
        )
        
        card = adapter.get_agent_card_for_protocol()
        assert card["id"] == "test-agent"
        assert card["name"] == "Test Agent"
        assert card["protocol"]["name"] == "test"
        assert card["protocol"]["version"] == "1.0"
        assert card["protocol"]["endpoint"] == "http://localhost:8080"


class TestACPAdapter:
    """Test the ACP protocol adapter."""
    
    def test_initialization(self):
        """Test ACP adapter initialization."""
        adapter = ACPAdapter()
        assert adapter.get_protocol_name() == "acp"
        assert adapter.get_protocol_version() == "1.0"
        assert adapter.app is None
        assert adapter.runner is None
        assert adapter.site is None
    
    @pytest.mark.asyncio
    async def test_start_creates_web_app(self, mock_aiohttp_app):
        """Test that start creates a web application."""
        adapter = ACPAdapter()
        agent = Mock(spec=Agent)
        agent.agent_id = "test-123"
        agent.name = "Test Agent"
        
        # Mock the aiohttp components
        mock_runner = AsyncMock()
        mock_site = AsyncMock()
        
        # Patch where the objects are used in the adapter module
        with patch("dawn.protocols.acp.adapter.web.AppRunner", return_value=mock_runner):
            with patch("dawn.protocols.acp.adapter.web.TCPSite", return_value=mock_site):
                endpoint = await adapter.start(agent, "localhost", 8080)
        
        # Verify endpoint
        assert endpoint.protocol == "acp"
        assert endpoint.port == 8080
        assert endpoint.base_url == "http://localhost:8080"
        
        # Verify web app setup
        mock_runner.setup.assert_called_once()
        mock_site.start.assert_called_once()
        
        # Verify adapter state
        assert adapter.is_running()
        assert adapter.agent == agent
    
    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Test that start fails when already running."""
        adapter = ACPAdapter()
        adapter._is_running = True
        
        agent = Mock(spec=Agent)
        
        with pytest.raises(RuntimeError, match="already running"):
            await adapter.start(agent)
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping the adapter."""
        adapter = ACPAdapter()
        
        # Mock the components
        adapter._is_running = True
        mock_site = AsyncMock()
        mock_runner = AsyncMock()
        adapter.site = mock_site
        adapter.runner = mock_runner
        
        await adapter.stop()
        
        # Check the mocks were called before they were set to None
        mock_site.stop.assert_called_once()
        mock_runner.cleanup.assert_called_once()
        assert not adapter._is_running
        assert adapter.endpoint is None
        assert adapter.site is None
        assert adapter.runner is None
    
    @pytest.mark.asyncio
    async def test_handle_chat_request(self, mock_aiohttp_app):
        """Test handling chat requests."""
        adapter = ACPAdapter()
        agent = AsyncMock(spec=Agent)
        agent.agent_id = "test-agent-123"  # Add missing attribute
        agent.process_message.return_value = {
            "response": "Test response",
            "success": True,
            "tools_used": ["tool1"],
            "reasoning_trace": ["step1"]
        }
        
        # Create mock request
        request = AsyncMock()
        request.json.return_value = {
            "message": "Test message",
            "context": {"key": "value"}
        }
        request.app = {"agent": agent}
        
        # Test successful chat
        with patch("dawn.protocols.acp.adapter.web.json_response") as mock_response:
            await adapter._handle_chat(request)
            
            agent.process_message.assert_called_once_with("Test message", {"key": "value"})
            
            # Check the response
            mock_response.assert_called_once()
            response_data = mock_response.call_args[0][0]
            assert response_data["response"] == "Test response"
            assert response_data["success"] is True
            assert response_data["metadata"]["tools_used"] == ["tool1"]
            assert response_data["metadata"]["agent_id"] == "test-agent-123"
    
    @pytest.mark.asyncio
    async def test_handle_chat_error(self, mock_aiohttp_app):
        """Test error handling in chat requests."""
        adapter = ACPAdapter()
        agent = AsyncMock(spec=Agent)
        agent.agent_id = "test-agent-123"  # Add missing attribute
        agent.process_message.side_effect = Exception("Test error")
        
        request = AsyncMock()
        request.json.return_value = {"message": "Test"}
        request.app = {"agent": agent}
        
        with patch("dawn.protocols.acp.adapter.web.json_response") as mock_response:
            await adapter._handle_chat(request)
            
            # Verify error response was called
            mock_response.assert_called_once()
            response_data = mock_response.call_args[0][0]
            assert "Test error" in response_data["error"]
            assert response_data["success"] is False
            
            # Check status code
            assert mock_response.call_args[1]["status"] == 500 