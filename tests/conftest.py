"""
Pytest configuration and shared fixtures for Dawn tests.
"""

import sys
import os
import pytest
import asyncio
import socket
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import uuid

# Add src to path so we can import dawn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration


def get_available_port():
    """Find an available port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def available_port():
    """Provide an available port for testing."""
    port = get_available_port()
    print(f"Using available port: {port}")
    return port


# Synthetic test data for mocking LLM responses
SYNTHETIC_RESPONSES = {
    "search for popular Python web frameworks": {
        "content": "I'll search for popular Python web frameworks on GitHub.\n\nSearching...",
        "tool_calls": [{
            "name": "github_search",
            "args": {"query": "Python web framework", "max_results": 5},
            "id": "call_123"
        }]
    },
    "analyze the django/django repository": {
        "content": "I'll analyze the Django repository for you.\n\nAnalyzing...",
        "tool_calls": [{
            "name": "github_analyze_repo",
            "args": {"repo_path": "django/django"},
            "id": "call_456"
        }]
    },
    "find recent papers on transformers": {
        "content": "I'll search for recent papers on transformers.\n\nSearching arXiv...",
        "tool_calls": [{
            "name": "arxiv_search",
            "args": {"query": "transformers", "max_results": 5},
            "id": "call_789"
        }]
    },
    "compare Python and JavaScript": {
        "content": "I'll compare Python and JavaScript for you.\n\nAnalyzing...",
        "tool_calls": [{
            "name": "compare_technologies",
            "args": {"tech1": "Python", "tech2": "JavaScript"},
            "id": "call_abc"
        }]
    },
    "default": {
        "content": "I understand your request. Let me help you with that.",
        "tool_calls": []
    }
}


class MockLLM(BaseChatModel):
    """Mock LLM for testing without API keys."""
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Synchronous generation (not used in our async code)."""
        raise NotImplementedError("Use async methods")
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async generation of mock responses."""
        # Get the last human message
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break
        
        # Find matching synthetic response
        response_data = SYNTHETIC_RESPONSES.get("default")
        if last_human_msg:
            for key, value in SYNTHETIC_RESPONSES.items():
                if key in last_human_msg.lower():
                    response_data = value
                    break
        
        # Create mock AIMessage
        ai_message = AIMessage(
            content=response_data["content"],
            additional_kwargs={
                "tool_calls": response_data.get("tool_calls", [])
            }
        )
        
        # Create ChatResult
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
    
    def bind_tools(self, tools):
        """Return self for tool binding (no-op in mock)."""
        return self
    
    @property
    def model_name(self) -> str:
        return "mock-llm"


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name: str, description: str = "Mock tool"):
        self.name = name
        self.description = description
    
    async def ainvoke(self, input_data: Dict[str, Any]) -> str:
        """Mock tool invocation."""
        if self.name == "github_search":
            return (
                "Found 5 repositories: "
                "[{'name': 'django/django', 'stars': 50000}, "
                "{'name': 'pallets/flask', 'stars': 40000}]"
            )
        elif self.name == "github_analyze_repo":
            return (
                "Repository analysis for django/django: "
                "{'name': 'django/django', 'stars': 50000, "
                "'languages': ['Python'], 'contributors': ['django']}"
            )
        elif self.name == "arxiv_search":
            return (
                "Found 3 papers: "
                "[{'title': 'Attention Is All You Need', 'id': '1706.03762'}]"
            )
        else:
            return f"Mock result from {self.name}"


@pytest.fixture
def mock_llm():
    """Provide a mock LLM instance."""
    return MockLLM()


@pytest.fixture
def mock_github_client():
    """Mock GitHub client."""
    # Import Github to use as spec
    from github import Github
    
    # Create mock with Github spec to pass isinstance checks
    mock = Mock(spec=Github)
    
    # Mock search_repositories
    repo1 = Mock()
    repo1.full_name = "django/django"
    repo1.description = "The Web framework for perfectionists with deadlines."
    repo1.html_url = "https://github.com/django/django"
    repo1.stargazers_count = 50000
    repo1.language = "Python"
    
    # Create a mock iterator for search results
    class MockSearchResults:
        def __init__(self, results):
            self.results = results
            self.index = 0
        
        def __iter__(self):
            return iter(self.results)
        
        def __getitem__(self, key):
            return self.results[key]
    
    mock.search_repositories.return_value = MockSearchResults([repo1])
    
    # Mock get_repo
    repo1.get_languages.return_value = {"Python": 10000, "JavaScript": 5000}
    repo1.forks_count = 15000
    repo1.open_issues_count = 500
    repo1.topics = ["web", "framework", "python"]
    repo1.created_at = Mock(strftime=Mock(return_value="2012-01-01"))
    repo1.updated_at = Mock(strftime=Mock(return_value="2024-01-01"))
    
    # Mock commits
    commit1 = Mock()
    commit1.sha = "1234567890abcdef"
    commit1.commit.message = "Fix bug in feature"
    commit1.commit.author.date = Mock(strftime=Mock(return_value="2024-01-01"))
    repo1.get_commits.return_value = [commit1]
    
    # Mock contributors
    contributor1 = Mock()
    contributor1.login = "contributor1"
    repo1.get_contributors.return_value = [contributor1]
    
    # Mock contents
    content1 = Mock()
    content1.name = "README.md"
    content1.type = "file"
    content1.path = "README.md"
    repo1.get_contents.return_value = [content1]
    
    mock.get_repo.return_value = repo1
    
    return mock


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_agent_id():
    """Generate a test agent ID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_aiohttp_app(monkeypatch):
    """Mock aiohttp web application."""
    app = Mock()
    app.router = Mock()
    app.router.add_get = Mock()
    app.router.add_post = Mock()
    app.__setitem__ = Mock()
    app.__getitem__ = Mock(return_value=Mock())
    
    # Mock the web module
    mock_web = Mock()
    mock_web.Application.return_value = app
    mock_web.AppRunner = AsyncMock
    mock_web.TCPSite = AsyncMock
    mock_web.json_response = Mock(return_value=Mock())
    
    monkeypatch.setattr("aiohttp.web", mock_web)
    
    return app


@pytest.fixture
def env_without_llm_keys(monkeypatch):
    """Remove all LLM API keys from environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def env_with_mock_keys(monkeypatch):
    """Set mock API keys for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-google-key") 