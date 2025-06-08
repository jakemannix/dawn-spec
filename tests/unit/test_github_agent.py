"""
Unit tests for the GitHub agent.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from dawn.agents.github import GitHubAgent


class TestGitHubAgent:
    """Test the GitHub agent implementation."""
    
    @pytest.fixture
    def github_agent(self, mock_github_client, mock_llm):
        """Create a GitHub agent with mocked dependencies."""
        with patch("dawn.agents.github.Github", return_value=mock_github_client):
            with patch.object(GitHubAgent, '_initialize_llm'):
                agent = GitHubAgent()
                agent.llm = mock_llm
                agent.github = mock_github_client
                return agent
    
    def test_initialization(self, github_agent):
        """Test GitHub agent initialization."""
        assert github_agent.name == "github"
        assert github_agent.description == "GitHub research agent with real API integration"
        assert github_agent.version == "1.0.0"
        assert github_agent.github is not None
    
    def test_get_tools(self, github_agent):
        """Test getting tool names."""
        tools = github_agent.get_tools()
        
        expected_tools = [
            "github_search",
            "github_fetch_file",
            "github_analyze_repo"
        ]
        
        assert len(tools) == len(expected_tools)
        for tool in expected_tools:
            assert tool in tools
    
    def test_get_capabilities(self, github_agent):
        """Test getting capabilities."""
        capabilities = github_agent.get_capabilities()
        
        assert len(capabilities) == 3
        
        # Check some capabilities
        cap_names = [cap["name"] for cap in capabilities]
        assert "GitHub Repository Search" in cap_names
        assert "GitHub Repository Analyzer" in cap_names
        assert "GitHub File Fetcher" in cap_names
    
    @pytest.mark.asyncio
    async def test_github_search_tool(self, github_agent):
        """Test the GitHub search tool."""
        from langchain_core.messages import HumanMessage
        
        # Get the search tool
        tools = github_agent.get_langgraph_tools()
        search_tool = next(t for t in tools if t.name == "github_search")
        
        # Test search
        result = await search_tool.ainvoke({
            "query": "Python web framework",
            "max_results": 5
        })
        
        assert "django/django" in result
        assert "50000" in result  # Stars count
        
        # Verify GitHub API was called
        github_agent.github.search_repositories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_github_analyze_repo_tool(self, github_agent):
        """Test the GitHub analyze repo tool."""
        tools = github_agent.get_langgraph_tools()
        analyze_tool = next(t for t in tools if t.name == "github_analyze_repo")
        
        # Test analysis
        result = await analyze_tool.ainvoke({
            "repo_path": "django/django"
        })
        
        assert "django/django" in result
        assert "50000" in result  # Stars
        assert "Python" in result  # Language
        
        # Verify GitHub API was called
        github_agent.github.get_repo.assert_called_once_with("django/django")
    
    @pytest.mark.skip(reason="github_trending tool not implemented")
    @pytest.mark.asyncio
    async def test_github_trending_tool(self, github_agent):
        """Test the GitHub trending tool."""
        # Mock trending repos (search with specific query)
        trending_repos = []
        for i in range(3):
            repo = Mock()
            repo.full_name = f"trending/repo{i}"
            repo.description = f"Trending repo {i}"
            repo.html_url = f"https://github.com/trending/repo{i}"
            repo.stargazers_count = 10000 * (i + 1)
            repo.language = "Python"
            trending_repos.append(repo)
        
        github_agent.github.search_repositories.return_value = trending_repos
        
        tools = github_agent.get_langgraph_tools()
        trending_tool = next(t for t in tools if t.name == "github_trending")
        
        result = await trending_tool.ainvoke({
            "language": "python",
            "since": "weekly"
        })
        
        assert "trending/repo0" in result
        assert "trending/repo1" in result
        assert "10000" in result
    
    @pytest.mark.skip(reason="github_repo_issues tool not implemented")
    @pytest.mark.asyncio
    async def test_github_repo_issues_tool(self, github_agent):
        """Test the GitHub repo issues tool."""
        # Mock repo and issues
        repo = Mock()
        
        issue1 = Mock()
        issue1.number = 123
        issue1.title = "Bug in feature X"
        issue1.state = "open"
        issue1.user.login = "user1"
        issue1.created_at.strftime.return_value = "2024-01-01"
        issue1.html_url = "https://github.com/test/repo/issues/123"
        
        repo.get_issues.return_value = [issue1]
        github_agent.github.get_repo.return_value = repo
        
        tools = github_agent.get_langgraph_tools()
        issues_tool = next(t for t in tools if t.name == "github_repo_issues")
        
        result = await issues_tool.ainvoke({
            "repo_path": "test/repo",
            "state": "open",
            "max_results": 10
        })
        
        assert "#123" in result
        assert "Bug in feature X" in result
        assert "open" in result
        assert "user1" in result
    
    @pytest.mark.skip(reason="github_repo_prs tool not implemented")
    @pytest.mark.asyncio
    async def test_github_repo_prs_tool(self, github_agent):
        """Test the GitHub repo PRs tool."""
        # Mock repo and PRs
        repo = Mock()
        
        pr1 = Mock()
        pr1.number = 456
        pr1.title = "Add new feature"
        pr1.state = "open"
        pr1.user.login = "contributor1"
        pr1.created_at.strftime.return_value = "2024-01-02"
        pr1.html_url = "https://github.com/test/repo/pull/456"
        
        repo.get_pulls.return_value = [pr1]
        github_agent.github_client.get_repo.return_value = repo
        
        tools = github_agent.get_langgraph_tools()
        prs_tool = next(t for t in tools if t.name == "github_repo_prs")
        
        result = await prs_tool.ainvoke({
            "repo_path": "test/repo",
            "state": "open",
            "max_results": 10
        })
        
        assert "#456" in result
        assert "Add new feature" in result
        assert "open" in result
        assert "contributor1" in result
    
    @pytest.mark.asyncio
    async def test_process_message_search(self, github_agent, mock_llm):
        """Test processing a search message."""
        # Mock the LLM to return a tool call
        from langchain_core.messages import AIMessage
        mock_response = AIMessage(
            content="I'll search for Python web frameworks on GitHub.",
            additional_kwargs={
                "tool_calls": [{
                    "name": "github_search",
                    "args": {"query": "Python web framework", "max_results": 5},
                    "id": "call_123"
                }]
            }
        )
        
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "messages": [mock_response],
            "tools_used": ["github_search"],
            "reasoning_trace": ["Searching GitHub"]
        })
        
        with patch.object(github_agent, 'build_graph', return_value=mock_graph):
            result = await github_agent.process_message("search for popular Python web frameworks")
        
        assert result["success"] is True
        assert "github_search" in result["tools_used"]
    
    @pytest.mark.asyncio
    async def test_process_message_error_handling(self, github_agent, mock_llm):
        """Test error handling in process_message."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("GitHub API error"))
        
        with patch.object(github_agent, 'build_graph', return_value=mock_graph):
            result = await github_agent.process_message("analyze django/django")
        
        assert result["success"] is False
        assert "GitHub API error" in result["error"] 