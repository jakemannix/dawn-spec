#!/usr/bin/env python3
"""
LangGraph GitHub Research Agent

A GitHub research agent powered by LangGraph that provides:
- Repository search with semantic understanding
- File content fetching
- Deep repository analysis
- Multi-protocol support (A2A, ACP, MCP)
"""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.tools import BaseTool, tool
from github import Github, GithubException
from pydantic import Field

from .base_langgraph_agent import MultiProtocolLangGraphAgent

logger = logging.getLogger(__name__)


class GitHubSearchTool(BaseTool):
    """Tool for searching GitHub repositories."""
    
    name: str = "github_search"
    description: str = "Search GitHub repositories by query. Returns top repositories matching the search criteria."
    github: Github = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, github_client: Github) -> None:
        super().__init__(github=github_client)
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute GitHub search."""
        try:
            repositories = self.github.search_repositories(query, sort="stars")
            results = []
            
            count = 0
            for repo in repositories:
                if count >= max_results:
                    break
                    
                results.append({
                    "name": repo.full_name,
                    "description": repo.description or "No description",
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "language": repo.language or "Unknown",
                    "updated_at": repo.updated_at.strftime("%Y-%m-%d") if repo.updated_at else "Unknown"
                })
                count += 1
            
            return f"Found {len(results)} repositories: {results}"
            
        except GithubException as e:
            return f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Async version of search."""
        return self._run(query, max_results)


class GitHubFetchFileTool(BaseTool):
    """Tool for fetching file content from GitHub."""
    
    name: str = "github_fetch_file"
    description: str = "Fetch raw file content from a GitHub repository. Requires repo path (owner/repo) and file path."
    github: Github = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, github_client: Github) -> None:
        super().__init__(github=github_client)
    
    def _run(self, repo_path: str, file_path: str) -> str:
        """Fetch file content from GitHub."""
        try:
            repo = self.github.get_repo(repo_path)
            file_content = repo.get_contents(file_path)
            
            if file_content.encoding == "base64":
                content = file_content.decoded_content.decode('utf-8')
                return f"File content from {repo_path}/{file_path}:\n\n{content[:2000]}..."
            else:
                return f"Cannot decode file with encoding: {file_content.encoding}"
                
        except GithubException as e:
            return f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
        except Exception as e:
            return f"Failed to fetch file: {str(e)}"
    
    async def _arun(self, repo_path: str, file_path: str) -> str:
        """Async version of fetch."""
        return self._run(repo_path, file_path)


class GitHubAnalyzeRepoTool(BaseTool):
    """Tool for deep analysis of a GitHub repository."""
    
    name: str = "github_analyze_repo"
    description: str = "Analyze a GitHub repository in detail. Provides information about structure, contributors, activity, etc."
    github: Github = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, github_client: Github) -> None:
        super().__init__(github=github_client)
    
    def _run(self, repo_path: str) -> str:
        """Analyze a GitHub repository."""
        try:
            repo = self.github.get_repo(repo_path)
            
            # Get languages
            languages = list(repo.get_languages().keys())
            
            # Get recent commits
            recent_commits = []
            commits = repo.get_commits()[:5]
            for commit in commits:
                recent_commits.append({
                    "sha": commit.sha[:7],
                    "message": commit.commit.message.split('\n')[0],
                    "date": commit.commit.author.date.strftime("%Y-%m-%d")
                })
            
            # Get top contributors
            contributors = []
            for contributor in repo.get_contributors()[:5]:
                contributors.append(contributor.login)
            
            # Get repository structure
            contents = repo.get_contents("")
            structure = []
            for content in contents[:10]:
                structure.append({
                    "name": content.name,
                    "type": content.type,
                    "path": content.path
                })
            
            analysis = {
                "name": repo.full_name,
                "description": repo.description,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "languages": languages,
                "recent_commits": recent_commits,
                "top_contributors": contributors,
                "structure": structure,
                "topics": repo.topics,
                "created_at": repo.created_at.strftime("%Y-%m-%d"),
                "updated_at": repo.updated_at.strftime("%Y-%m-%d")
            }
            
            return f"Repository analysis for {repo_path}: {analysis}"
            
        except GithubException as e:
            return f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    async def _arun(self, repo_path: str) -> str:
        """Async version of analyze."""
        return self._run(repo_path)


class LangGraphGitHubAgent(MultiProtocolLangGraphAgent):
    """
    GitHub research agent powered by LangGraph with real API integration.
    
    This agent provides:
    - Semantic repository search
    - File content retrieval
    - Deep repository analysis
    - Multi-protocol support
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "GitHub Research Agent",
        port: int = 8081
    ):
        # Initialize GitHub client
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github = Github(self.github_token) if self.github_token else Github()
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="LangGraph-powered GitHub research agent with real API integration",
            version="1.0.0",
            port=port
        )
        
        logger.info(f"Initialized {name} with GitHub API access")
    
    def get_tools(self) -> List[BaseTool]:
        """Return GitHub-specific tools."""
        return [
            GitHubSearchTool(self.github),
            GitHubFetchFileTool(self.github),
            GitHubAnalyzeRepoTool(self.github)
        ]
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return OASF capability definitions."""
        return [
            {
                "id": "github_search",
                "type": "research.github.search",
                "name": "GitHub Repository Search",
                "description": "Search GitHub repositories with semantic understanding",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query for repositories",
                        "required": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "metadata": {
                    "examples": ["search for machine learning frameworks", "find Python web frameworks"],
                    "tags": ["github", "search", "research"]
                }
            },
            {
                "id": "github_fetch_file",
                "type": "research.github.file",
                "name": "GitHub File Fetcher",
                "description": "Fetch raw file content from GitHub repositories",
                "parameters": {
                    "repo_path": {
                        "type": "string",
                        "description": "Repository path (owner/repo)",
                        "required": True
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to file in repository",
                        "required": True
                    }
                },
                "metadata": {
                    "examples": ["fetch pytorch/pytorch README.md", "get langchain/langchain setup.py"],
                    "tags": ["github", "file", "content"]
                }
            },
            {
                "id": "github_analyze_repo",
                "type": "research.github.analysis",
                "name": "GitHub Repository Analyzer",
                "description": "Deep analysis of GitHub repository structure and metadata",
                "parameters": {
                    "repo_path": {
                        "type": "string",
                        "description": "Repository path (owner/repo)",
                        "required": True
                    }
                },
                "metadata": {
                    "examples": ["analyze pytorch/pytorch", "examine tensorflow/tensorflow"],
                    "tags": ["github", "analysis", "research"]
                }
            }
        ]
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """
        Natural language chat interface for the GitHub agent.
        
        This method provides a user-friendly interface that:
        - Understands natural language queries
        - Uses LangGraph ReACT for reasoning
        - Returns formatted responses
        """
        logger.info(f"GitHub agent received message: {message}")
        
        # Process with LangGraph
        result = await self.process_message(message)
        
        if result.get("success"):
            # Format the response nicely
            response = f"🔍 **GitHub Research Results**\n\n{result['response']}"
            
            # Add tool usage information
            if result.get("tools_used"):
                response += f"\n\n📊 **Tools used:** {', '.join(result['tools_used'])}"
            
            # Add reasoning trace if available
            if result.get("reasoning_trace") and len(result["reasoning_trace"]) > 1:
                response += "\n\n🧠 **Reasoning steps:**"
                for step in result["reasoning_trace"]:
                    response += f"\n• {step}"
            
            return {
                "response": response,
                "type": "github_research",
                "data": result,
                "success": True
            }
        else:
            return {
                "response": f"❌ GitHub research failed: {result.get('error', 'Unknown error')}",
                "type": "error",
                "success": False
            }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = LangGraphGitHubAgent()
        
        # Test queries
        queries = [
            "Search for popular Python machine learning frameworks",
            "Analyze the pytorch/pytorch repository",
            "Fetch the README.md from langchain-ai/langchain"
        ]
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            result = await agent.chat(query)
            print(result["response"])
    
    asyncio.run(test_agent()) 