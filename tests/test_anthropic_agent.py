"""
Unit tests for the Anthropic agent implementation.

These tests verify that the AnthropicAgent class correctly implements
the IAgent interface and properly handles API calls to Anthropic's Claude models.
"""
import pytest
import uuid
from unittest.mock import patch, MagicMock
from typing import Dict, List, Optional, Any

from src.anthropic_agent import AnthropicAgent, create_anthropic_agent
from src.interfaces import IAgent


# Test that AnthropicAgent correctly implements IAgent interface
def test_anthropic_agent_implements_iagent():
    """Test that AnthropicAgent implements the IAgent interface."""
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[
            {
                "id": "test-capability",
                "type": "test",
                "name": "Test Capability",
                "description": "Test description"
            }
        ]
    )
    
    assert isinstance(agent, IAgent)


# Test the factory function creates a properly configured agent
def test_create_anthropic_agent():
    """Test that the factory function creates a properly configured agent."""
    agent = create_anthropic_agent(
        name="Test Agent",
        description="Test description",
        model="claude-3-opus-20240229",
        temperature=0.5,
        max_tokens=2000
    )
    
    info = agent.get_info()
    assert info["name"] == "Test Agent"
    assert info["description"] == "Test description"
    assert info["model"] == "claude-3-opus-20240229"
    assert info["provider"] == "anthropic"
    
    # Verify standard capabilities are present
    capabilities = agent.get_capabilities()
    capability_ids = [cap["id"] for cap in capabilities]
    assert "text-generation" in capability_ids
    assert "chat" in capability_ids
    assert "summarization" in capability_ids
    assert "classification" in capability_ids
    assert "extraction" in capability_ids


# Test the get_info method returns correct information
def test_get_info():
    """Test that get_info returns the correct agent information."""
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[
            {
                "id": "test-capability",
                "type": "test",
                "name": "Test Capability",
                "description": "Test description"
            }
        ],
        model="claude-3-sonnet-20240229"
    )
    
    info = agent.get_info()
    assert info["name"] == "Test Agent"
    assert info["description"] == "Test description"
    assert isinstance(info["id"], str)
    assert info["model"] == "claude-3-sonnet-20240229"
    assert info["provider"] == "anthropic"
    assert len(info["capabilities"]) == 1
    assert info["capabilities"][0]["id"] == "test-capability"


# Test get_capabilities method
def test_get_capabilities():
    """Test that get_capabilities returns the correct capabilities."""
    test_capabilities = [
        {
            "id": "capability-1",
            "type": "type1",
            "name": "Name 1",
            "description": "Description 1"
        },
        {
            "id": "capability-2",
            "type": "type2",
            "name": "Name 2",
            "description": "Description 2"
        }
    ]
    
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=test_capabilities
    )
    
    capabilities = agent.get_capabilities()
    assert capabilities == test_capabilities
    assert len(capabilities) == 2
    assert capabilities[0]["id"] == "capability-1"
    assert capabilities[1]["id"] == "capability-2"


# Test invoke method with mocked Anthropic API
@patch('anthropic.Anthropic')
@patch('src.anthropic_agent.APIConfig')
def test_invoke_text_generation(mock_api_config, mock_anthropic):
    """Test that invoke properly calls the Anthropic API for text generation."""
    # Configure mocks
    mock_api_config.is_anthropic_configured.return_value = True
    mock_api_config.ANTHROPIC_API_KEY = "mock-api-key"
    
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Generated text")]
    mock_response.model = "claude-3-sonnet-20240229"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_response.stop_reason = "stop_sequence"
    
    mock_client.messages.create.return_value = mock_response
    
    # Create agent
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[
            {
                "id": "text-generation",
                "type": "text_generation",
                "name": "Text Generation",
                "description": "Generate text based on a prompt"
            }
        ],
        model="claude-3-sonnet-20240229"
    )
    
    # Invoke the agent
    result = agent.invoke("text-generation", {"prompt": "Test prompt"})
    
    # Verify API was called correctly
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    assert call_args["model"] == "claude-3-sonnet-20240229"
    assert isinstance(call_args["messages"], list)
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["messages"][0]["content"] == "Test prompt"
    
    # Verify result
    assert result["content"] == "Generated text"
    assert result["model"] == "claude-3-sonnet-20240229"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 20
    assert result["stop_reason"] == "stop_sequence"


# Test invoke method with chat capability
@patch('anthropic.Anthropic')
@patch('src.anthropic_agent.APIConfig')
def test_invoke_chat(mock_api_config, mock_anthropic):
    """Test that invoke properly handles chat interactions."""
    # Configure mocks
    mock_api_config.is_anthropic_configured.return_value = True
    mock_api_config.ANTHROPIC_API_KEY = "mock-api-key"
    
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Chat response")]
    mock_response.model = "claude-3-sonnet-20240229"
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 25
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    # Create agent
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[
            {
                "id": "chat",
                "type": "chat",
                "name": "Chat",
                "description": "Chat conversation"
            }
        ],
        model="claude-3-sonnet-20240229"
    )
    
    # Invoke the agent with chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    result = agent.invoke("chat", {"messages": messages})
    
    # Verify API was called correctly
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    assert call_args["model"] == "claude-3-sonnet-20240229"
    
    # Verify messages were properly transformed for Anthropic
    api_messages = call_args["messages"]
    assert len(api_messages) == 3  # System message handled separately
    
    # Verify system message was extracted and set correctly
    assert call_args["system"] == "You are a helpful assistant"
    
    # Verify result
    assert result["content"] == "Chat response"
    assert result["model"] == "claude-3-sonnet-20240229"
    assert result["usage"]["input_tokens"] == 15
    assert result["usage"]["output_tokens"] == 25


# Test error handling
@patch('anthropic.Anthropic')
@patch('src.anthropic_agent.APIConfig')
def test_invoke_api_error(mock_api_config, mock_anthropic):
    """Test that invoke properly handles API errors."""
    # Configure mocks
    mock_api_config.is_anthropic_configured.return_value = True
    mock_api_config.ANTHROPIC_API_KEY = "mock-api-key"
    
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # Make the API call raise a generic exception (simulating API error)
    mock_client.messages.create.side_effect = Exception("API Error")
    
    # Create agent
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[
            {
                "id": "text-generation",
                "type": "text_generation",
                "name": "Text Generation",
                "description": "Generate text based on a prompt"
            }
        ]
    )
    
    # Invoke the agent
    result = agent.invoke("text-generation", {"prompt": "Test prompt"})
    
    # Verify error handling
    assert "error" in result
    assert "API Error" in result["error"]


# Test health check
@patch('anthropic.Anthropic')
@patch('src.anthropic_agent.APIConfig')
def test_health_check(mock_api_config, mock_anthropic):
    """Test that health_check properly verifies API connectivity."""
    # Configure mocks
    mock_api_config.is_anthropic_configured.return_value = True
    mock_api_config.ANTHROPIC_API_KEY = "mock-api-key"
    
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="OK")]
    mock_client.messages.create.return_value = mock_response
    
    # Create agent
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[]
    )
    
    # Check health
    health = agent.health_check()
    
    # Verify API was called correctly
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    assert call_args["max_tokens"] == 10
    assert "working" in call_args["messages"][0]["content"].lower()
    
    # Verify result
    assert health is True


# Test health check with API error
@patch('anthropic.Anthropic')
@patch('src.anthropic_agent.APIConfig')
def test_health_check_error(mock_api_config, mock_anthropic):
    """Test that health_check properly handles API errors."""
    # Configure mocks
    mock_api_config.is_anthropic_configured.return_value = True
    mock_api_config.ANTHROPIC_API_KEY = "mock-api-key"
    
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # Make the API call raise a generic exception (simulating API error)
    mock_client.messages.create.side_effect = Exception("API Error")
    
    # Create agent
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[]
    )
    
    # Check health
    health = agent.health_check()
    
    # Verify result
    assert health is False


# Test unconfigured API
@patch('src.anthropic_agent.APIConfig')
def test_unconfigured_api(mock_api_config):
    """Test agent behavior when API is not configured."""
    # Configure mock
    mock_api_config.is_anthropic_configured.return_value = False
    
    # Create agent
    agent = AnthropicAgent(
        name="Test Agent",
        description="Test description",
        capabilities=[
            {
                "id": "text-generation",
                "type": "text_generation",
                "name": "Text Generation",
                "description": "Generate text based on a prompt"
            }
        ]
    )
    
    # Invoke the agent
    result = agent.invoke("text-generation", {"prompt": "Test prompt"})
    
    # Verify error handling
    assert "error" in result
    assert "not configured" in result["error"]
    
    # Check health
    health = agent.health_check()
    assert health is False