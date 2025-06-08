"""
Unit tests for the base Agent class.
"""

import pytest
from typing import Dict, Any, List

from dawn.core.agent import Agent


class MockTestAgent(Agent):
    """Test implementation of Agent for unit testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_message_called = False
        self.last_message = None
        self.last_context = None
    
    async def process_message(
        self, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Track calls for testing."""
        self.process_message_called = True
        self.last_message = message
        self.last_context = context
        
        return {
            "response": f"Processed: {message}",
            "success": True,
            "tools_used": ["test_tool"],
            "reasoning_trace": ["Step 1", "Step 2"]
        }
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return test capabilities."""
        return [
            {
                "id": "test_capability",
                "name": "Test Capability",
                "description": "A test capability",
                "type": "test"
            }
        ]
    
    def get_tools(self) -> List[str]:
        """Return test tools."""
        return ["test_tool", "another_tool"]


class TestBaseAgent:
    """Test the base Agent interface."""
    
    def test_agent_initialization(self):
        """Test agent initialization with various parameters."""
        # Test with defaults
        agent = MockTestAgent()
        assert agent.name == "Agent"
        assert agent.description == "A Dawn agent"
        assert agent.version == "1.0.0"
        assert agent.agent_id is not None
        
        # Test with custom values
        agent = MockTestAgent(
            agent_id="custom-id",
            name="Custom Agent",
            description="A custom test agent",
            version="2.0.0"
        )
        assert agent.agent_id == "custom-id"
        assert agent.name == "Custom Agent"
        assert agent.description == "A custom test agent"
        assert agent.version == "2.0.0"
    
    def test_agent_id_generation(self):
        """Test that agent IDs are unique when not provided."""
        agent1 = MockTestAgent()
        agent2 = MockTestAgent()
        assert agent1.agent_id != agent2.agent_id
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test the process_message method."""
        agent = MockTestAgent()
        
        # Test without context
        result = await agent.process_message("Test message")
        assert agent.process_message_called
        assert agent.last_message == "Test message"
        assert agent.last_context is None
        assert result["success"] is True
        assert result["response"] == "Processed: Test message"
        assert "test_tool" in result["tools_used"]
        
        # Test with context
        context = {"key": "value"}
        result = await agent.process_message("Another message", context)
        assert agent.last_message == "Another message"
        assert agent.last_context == context
    
    def test_get_capabilities(self):
        """Test the get_capabilities method."""
        agent = MockTestAgent()
        capabilities = agent.get_capabilities()
        
        assert len(capabilities) == 1
        assert capabilities[0]["id"] == "test_capability"
        assert capabilities[0]["name"] == "Test Capability"
        assert capabilities[0]["type"] == "test"
    
    def test_get_tools(self):
        """Test the get_tools method."""
        agent = MockTestAgent()
        tools = agent.get_tools()
        
        assert len(tools) == 2
        assert "test_tool" in tools
        assert "another_tool" in tools
    
    def test_get_agent_card(self):
        """Test agent card generation."""
        agent = MockTestAgent(
            agent_id="test-123",
            name="Test Agent",
            description="A test agent",
            version="1.2.3"
        )
        
        card = agent.get_agent_card()
        
        assert card["id"] == "test-123"
        assert card["name"] == "Test Agent"
        assert card["description"] == "A test agent"
        assert card["version"] == "1.2.3"
        assert card["capabilities"] == agent.get_capabilities()
        assert card["tools"] == agent.get_tools()
        assert card["metadata"]["framework"] == "dawn"
    
    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        """Test the initialize and shutdown lifecycle methods."""
        agent = MockTestAgent()
        
        # These should not raise exceptions
        await agent.initialize()
        await agent.shutdown()
    
    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Should fail because abstract methods aren't implemented
            class BadAgent(Agent):
                pass
            
            BadAgent() 