"""
Unit tests for the agent registry.
"""

import pytest
from typing import List, Dict, Any

from dawn.agents.registry import AgentRegistry, registry
from dawn.core.agent import Agent


class MockAgentImpl(Agent):
    """Test agent implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def process_message(self, message: str, context=None):
        return {"response": f"Test: {message}", "success": True}
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [{"id": "test", "name": "Test"}]
    
    def get_tools(self) -> List[str]:
        return ["test_tool"]


class AnotherTestAgent(Agent):
    """Another test agent implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def process_message(self, message: str, context=None):
        return {"response": f"Another: {message}", "success": True}
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [{"id": "another", "name": "Another"}]
    
    def get_tools(self) -> List[str]:
        return ["another_tool"]


class TestAgentRegistry:
    """Test the agent registry functionality."""
    
    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear the registry before and after each test."""
        AgentRegistry.clear()
        yield
        AgentRegistry.clear()
    
    def test_register_agent(self):
        """Test registering an agent."""
        @AgentRegistry.register("test_agent", description="A test agent", tags=["test"])
        class RegisteredAgent(MockAgentImpl):
            pass
        
        assert "test_agent" in AgentRegistry._agents
        assert AgentRegistry._agents["test_agent"] == RegisteredAgent
        
        metadata = AgentRegistry._metadata["test_agent"]
        assert metadata["description"] == "A test agent"
        assert metadata["tags"] == ["test"]
        assert metadata["class"] == "RegisteredAgent"
    
    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        AgentRegistry.register("agent1")(MockAgentImpl)
        AgentRegistry.register("agent2")(AnotherTestAgent)
        
        assert len(AgentRegistry._agents) == 2
        assert "agent1" in AgentRegistry._agents
        assert "agent2" in AgentRegistry._agents
    
    def test_register_overwrites_existing(self):
        """Test that re-registering overwrites existing agent."""
        AgentRegistry.register("test")(MockAgentImpl)
        AgentRegistry.register("test")(AnotherTestAgent)
        
        assert AgentRegistry._agents["test"] == AnotherTestAgent
    
    def test_create_agent(self):
        """Test creating an agent from registry."""
        AgentRegistry.register("test_agent")(MockAgentImpl)
        
        # Create agent with custom parameters
        agent = AgentRegistry.create("test_agent", description="Custom Description")
        assert isinstance(agent, MockAgentImpl)
        
        # The name parameter would conflict with the registry's name param,
        # so we test with a different parameter
        assert agent.description == "Custom Description"
    
    def test_create_unknown_agent(self):
        """Test creating an unknown agent raises error."""
        with pytest.raises(ValueError, match="Unknown agent: unknown"):
            AgentRegistry.create("unknown")
    
    def test_list_agents(self):
        """Test listing registered agents."""
        AgentRegistry.register("agent1")(MockAgentImpl)
        AgentRegistry.register("agent2")(AnotherTestAgent)
        
        agents = AgentRegistry.list_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents
    
    def test_get_agent_info(self):
        """Test getting agent metadata."""
        AgentRegistry.register(
            "test_agent",
            description="Test description",
            tags=["test", "example"],
            custom_field="custom_value"
        )(MockAgentImpl)
        
        info = AgentRegistry.get_agent_info("test_agent")
        
        assert info["description"] == "Test description"
        assert info["tags"] == ["test", "example"]
        assert info["custom_field"] == "custom_value"
        assert info["class"] == "MockAgentImpl"
    
    def test_get_unknown_agent_info(self):
        """Test getting info for unknown agent raises error."""
        with pytest.raises(ValueError, match="Unknown agent: unknown"):
            AgentRegistry.get_agent_info("unknown")
    
    def test_find_by_tag(self):
        """Test finding agents by tag."""
        AgentRegistry.register("agent1", tags=["research", "github"])(MockAgentImpl)
        AgentRegistry.register("agent2", tags=["research", "arxiv"])(AnotherTestAgent)
        AgentRegistry.register("agent3", tags=["synthesis"])(MockAgentImpl)
        
        # Find by single tag
        research_agents = AgentRegistry.find_by_tag("research")
        assert len(research_agents) == 2
        assert "agent1" in research_agents
        assert "agent2" in research_agents
        
        # Find by specific tag
        github_agents = AgentRegistry.find_by_tag("github")
        assert len(github_agents) == 1
        assert "agent1" in github_agents
        
        # Find by non-existent tag
        unknown_agents = AgentRegistry.find_by_tag("unknown")
        assert len(unknown_agents) == 0
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        AgentRegistry.register("agent1")(MockAgentImpl)
        AgentRegistry.register("agent2")(AnotherTestAgent)
        
        assert len(AgentRegistry._agents) == 2
        
        AgentRegistry.clear()
        
        assert len(AgentRegistry._agents) == 0
        assert len(AgentRegistry._metadata) == 0
    
    def test_registry_instance(self):
        """Test the convenience registry instance."""
        # The module-level registry should work the same way
        registry.register("test")(MockAgentImpl)
        
        agent = registry.create("test")
        assert isinstance(agent, MockAgentImpl)
        
        registry.clear()
    
    def test_register_without_optional_params(self):
        """Test registering with minimal parameters."""
        @AgentRegistry.register("minimal")
        class MinimalAgent(MockAgentImpl):
            """Minimal agent implementation."""
            pass
        
        info = AgentRegistry.get_agent_info("minimal")
        assert info["description"] == "Minimal agent implementation."  # From docstring
        assert info["tags"] == []
        assert info["class"] == "MinimalAgent" 