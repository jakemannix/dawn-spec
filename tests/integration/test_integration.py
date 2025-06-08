"""
Integration tests for Dawn agents.
"""

import pytest
import asyncio
from unittest.mock import patch

from dawn.agents.github import GitHubAgent
from dawn.protocols.acp.adapter import ACPAdapter
from dawn.runners.base import DemoRunner
from dawn.agents.registry import registry
from tests.conftest import get_available_port


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for agents with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_github_agent_with_acp(self, mock_github_client, mock_llm, available_port):
        """Test GitHub agent running with ACP protocol."""
        print(f"🔧 Starting ACP integration test on port {available_port}")
        
        # Create agent with mocked dependencies
        with patch("dawn.agents.github.Github", return_value=mock_github_client):
            with patch.object(GitHubAgent, '_initialize_llm'):
                agent = GitHubAgent()
                agent.llm = mock_llm
                agent.github_client = mock_github_client
        
        # Create ACP adapter
        adapter = ACPAdapter()
        
        # Create runner with available port
        runner = DemoRunner(
            agent,
            adapter,
            port=available_port,
            show_ui=False
        )
        
        print(f"📡 ACP demo server will be available at http://localhost:{available_port}")
        
        # Initialize agent and start adapter directly (avoid interactive loop)
        await runner.agent.initialize()
        endpoint = await runner.adapter.start(runner.agent, runner.host, runner.port)
        
        # Verify the endpoint
        assert endpoint.protocol == "acp"
        assert endpoint.port == available_port
        assert endpoint.base_url == f"http://localhost:{available_port}"
        
        # Send a test message directly to the agent
        result = await runner.send_message(
            "search for popular Python web frameworks"
        )
        
        assert result["success"] is True
        assert "response" in result
        
        print(f"✅ Successfully tested ACP integration on port {available_port}")
        
        # Stop the runner cleanly
        await runner.stop()
    
    @pytest.mark.asyncio
    async def test_agent_registry_integration(self, mock_llm):
        """Test agent registry with actual agent creation."""
        # Clear registry first
        registry.clear()
        
        # Register test agent
        @registry.register("test_integration", description="Integration test agent", tags=["test"])
        class IntegrationTestAgent(GitHubAgent):
            """Test agent for integration."""
            def __init__(self, **kwargs):
                with patch.object(self, '_initialize_llm'):
                    super().__init__(**kwargs)
                    self.llm = mock_llm
        
        # Create agent from registry
        agent = registry.create("test_integration")
        
        # Now the agent name should match the registry key
        assert agent.name == "test_integration"
        assert isinstance(agent, GitHubAgent)
        
        # Test agent functionality
        result = await agent.process_message("test message")
        assert result["success"] is True
        
        # Find by tag - this should work because we registered with the tag
        test_agents = registry.find_by_tag("test")
        assert "test_integration" in test_agents
        
        # Clean up
        registry.clear()
    
    @pytest.mark.asyncio
    async def test_multiple_agents_same_protocol(self, mock_llm, mock_github_client):
        """Test running multiple agents with the same protocol adapter."""
        agents = []
        
        # Create multiple agents
        for i in range(3):
            with patch("dawn.agents.github.Github", return_value=mock_github_client):
                with patch.object(GitHubAgent, '_initialize_llm'):
                    agent = GitHubAgent(
                        agent_id=f"agent-{i}",
                        name=f"Agent {i}"
                    )
                    agent.llm = mock_llm
                    agents.append(agent)
        
        # Test each agent independently
        for i, agent in enumerate(agents):
            adapter = ACPAdapter()
            
            # Get an available port for each agent
            port = get_available_port()
            print(f"🔧 Starting Agent {i} on port {port}")
            
            # Start on available port
            endpoint = await adapter.start(agent, port=port)
            assert endpoint.port == port
            
            print(f"📡 Agent {i} demo server available at http://localhost:{port}")
            
            # Send message
            result = await agent.process_message(f"Message for agent {i}")
            assert result["success"] is True
            
            print(f"✅ Agent {i} test completed on port {port}")
            
            # Stop adapter
            await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, mock_llm):
        """Test agent error recovery."""
        with patch.object(GitHubAgent, '_initialize_llm'):
            agent = GitHubAgent()
            agent.llm = mock_llm
        
        # Make the agent fail on first message
        call_count = 0
        original_process = agent.process_message
        
        async def failing_process(message, context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated failure")
            return await original_process(message, context)
        
        agent.process_message = failing_process
        
        # First call should fail - but we need to catch the exception
        try:
            result1 = await agent.process_message("First message")
            # If we get here, the exception wasn't raised
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert str(e) == "Simulated failure"
        
        # Second call should succeed
        result2 = await agent.process_message("Second message")
        assert result2["success"] is True 