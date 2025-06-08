"""
A2A Interoperability Demo

This demo shows how DAWN agents can communicate using Google's A2A protocol,
demonstrating the peer-to-peer communication paradigm where both agents
maintain their own planning and reasoning capabilities.

Enhanced version with real research capabilities from research_agent_demo.py.
"""
import asyncio
import logging
import sys
import os
import uuid
import time
from typing import Dict, Any, List, Optional

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import DAWN components
from src.agent import A2ACapableAgent, Capability
from src.config import APIConfig

# Import research libraries
import openai
from github import Github, GithubException
import arxiv
from duckduckgo_search import DDGS

# Configure OpenAI client if available
client = None
if APIConfig.is_openai_configured():
    client = openai.OpenAI(
        api_key=APIConfig.OPENAI_API_KEY,
        organization=APIConfig.OPENAI_ORG_ID
    )


class A2AGitHubResearchAgent(A2ACapableAgent):
    """
    A2A-capable GitHub research agent that can search repositories and analyze them.
    Can also delegate tasks to other agents via A2A protocol.
    """
    
    def __init__(self, agent_name: str = "GitHubResearcher"):
        super().__init__(
            name=agent_name,
            description="GitHub research agent capable of repository search and analysis",
            provider="DAWN Research Suite",
            version="1.0.0"
        )
        
        # Initialize GitHub client
        self.github_token = APIConfig.GITHUB_TOKEN
        self.github = Github(self.github_token) if self.github_token else Github()
        
        # Add research capabilities
        self._add_github_capabilities()
        
        logger.info(f"Initialized {agent_name} with GitHub API access")
        
    def _add_github_capabilities(self):
        """Add GitHub research capabilities to the agent."""
        
        # GitHub search capability
        search_cap = Capability(
            capability_type="github_search",
            name="GitHub Repository Search",
            description="Search GitHub repositories for relevant information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query for GitHub repositories",
                    "required": True
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            }
        )
        self.add_capability(search_cap)
        
        # Repository analysis capability
        analysis_cap = Capability(
            capability_type="github_repo_analysis",
            name="GitHub Repository Analysis",
            description="Analyze a specific GitHub repository for detailed information",
            parameters={
                "repo_url": {
                    "type": "string",
                    "description": "URL of the GitHub repository to analyze",
                    "required": True
                }
            }
        )
        self.add_capability(analysis_cap)
        
        # Research workflow capability (can delegate to other agents)
        workflow_cap = Capability(
            capability_type="github_research_workflow",
            name="GitHub Research Workflow",
            description="Execute a complete GitHub research workflow, potentially using other agents",
            parameters={
                "research_question": {
                    "type": "string",
                    "description": "The research question to investigate",
                    "required": True
                },
                "remote_agents": {
                    "type": "array",
                    "description": "URLs of remote agents to collaborate with",
                    "items": {"type": "string"},
                    "default": []
                }
            }
        )
        self.add_capability(workflow_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"A2AGitHubResearchAgent invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id or cap.name == capability_id or cap.type == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "github_search":
                return self._handle_github_search(inputs)
            elif capability.type == "github_repo_analysis":
                return self._handle_repo_analysis(inputs)
            elif capability.type == "github_research_workflow":
                # This requires async execution
                return {"error": "This capability requires async execution. Use invoke_async instead."}
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in capability {capability_id}: {e}")
            return {"error": str(e)}
            
    async def invoke_async(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of invoke for capabilities that require remote calls."""
        logger.info(f"A2AGitHubResearchAgent async invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id or cap.name == capability_id or cap.type == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "github_research_workflow":
                return await self._handle_research_workflow(inputs)
            else:
                # For non-async capabilities, call the regular invoke
                return self.invoke(capability_id, inputs, config)
                
        except Exception as e:
            logger.error(f"Error in async capability {capability_id}: {e}")
            return {"error": str(e)}
    
    def _handle_github_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Search GitHub for repositories matching a query."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        if not query:
            return {"error": "No query provided for GitHub search"}
        
        logger.info(f"Searching GitHub for: {query}")
        
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
                
            logger.info(f"Found {len(results)} repositories matching '{query}'")
            return {
                "results": results,
                "count": len(results),
                "query": query,
                "success": True
            }
        except GithubException as e:
            error_message = f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
            logger.error(error_message)
            return {"error": error_message}
        except Exception as e:
            error_message = f"GitHub search failed: {str(e)}"
            logger.error(error_message)
            return {"error": error_message}
    
    def _handle_repo_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific GitHub repository."""
        repo_url = inputs.get("repository_url", inputs.get("repo_url", ""))
        if not repo_url:
            return {"error": "No repository URL provided for analysis"}
        
        logger.info(f"Analyzing GitHub repository: {repo_url}")
        
        try:
            # Extract owner and repo name from URL
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 5 or parts[2] != 'github.com':
                return {"error": f"Invalid GitHub URL: {repo_url}"}
                
            owner = parts[3]
            repo_name = parts[4]
            
            # Fetch repository details
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Get languages and topics
            languages_data = repo.get_languages()
            languages = list(languages_data.keys())
            topics = repo.topics
            
            # Get recent commits
            recent_commits = []
            try:
                commits = repo.get_commits()[:5]
                for commit in commits:
                    recent_commits.append({
                        "sha": commit.sha[:7],
                        "message": commit.commit.message.split('\n')[0],
                        "date": commit.commit.author.date.strftime("%Y-%m-%d") if commit.commit.author.date else "Unknown"
                    })
            except:
                recent_commits = []
            
            analysis = {
                "repo_name": f"{owner}/{repo_name}",
                "languages": languages,
                "topics": topics,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "recent_commits": recent_commits,
                "description": repo.description
            }
            
            logger.info(f"Successfully analyzed repository {owner}/{repo_name}")
            return {
                "analysis": analysis,
                "success": True
            }
            
        except GithubException as e:
            error_message = f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
            logger.error(error_message)
            return {"error": error_message}
        except Exception as e:
            error_message = f"Repository analysis failed: {str(e)}"
            logger.error(error_message)
            return {"error": error_message}
    
    async def _handle_research_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle research workflow that can involve multiple agents."""
        research_question = inputs.get("research_question", "")
        remote_agents = inputs.get("remote_agents", [])
        
        if not research_question:
            return {"error": "No research question provided"}
        
        logger.info(f"Starting GitHub research workflow for: {research_question}")
        
        # Step 1: Search GitHub based on the research question
        search_result = self._handle_github_search({
            "query": research_question,
            "max_results": 3
        })
        
        if "error" in search_result:
            return search_result
        
        # Step 2: Analyze top repositories
        analyses = []
        for repo in search_result.get("results", []):
            analysis_result = self._handle_repo_analysis({
                "repo_url": repo["url"]
            })
            if "error" not in analysis_result:
                analyses.append(analysis_result["analysis"])
        
        # Step 3: If remote agents are provided, delegate synthesis to them
        synthesis_result = None
        if remote_agents and client:  # Only if we have OpenAI configured for synthesis
            try:
                # Connect to first available remote agent for synthesis
                for agent_url in remote_agents:
                    try:
                        logger.info(f"Attempting to connect to remote agent: {agent_url}")
                        remote_agent_id = await self.connect_to_a2a_agent(agent_url, f"synthesis_{uuid.uuid4().hex[:8]}")
                        
                        # Discover synthesis capabilities
                        remote_capabilities = await self.discover_remote_a2a_capabilities(remote_agent_id)
                        
                        # Look for synthesis capability
                        synthesis_cap = None
                        for cap in remote_capabilities:
                            if "synthesis" in cap.get("name", "").lower() or "analysis" in cap.get("name", "").lower():
                                synthesis_cap = cap
                                break
                        
                        if synthesis_cap:
                            # Delegate synthesis to remote agent
                            synthesis_result = await self.invoke_remote_a2a_capability(
                                remote_agent_id,
                                synthesis_cap["name"],
                                {
                                    "sources": analyses,
                                    "question": research_question,
                                    "source_type": "github"
                                }
                            )
                            logger.info("Successfully delegated synthesis to remote agent")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to connect to remote agent {agent_url}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error connecting to remote agents: {e}")
        
        return {
            "research_question": research_question,
            "github_search": search_result,
            "repository_analyses": analyses,
            "synthesis": synthesis_result,
            "remote_agents_used": len([url for url in remote_agents if synthesis_result]),
            "success": True
        }
    
    async def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Chat interface for natural language interaction with GitHub research agent.
        
        Args:
            user_message: Natural language query from user
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"GitHub agent received chat message: {user_message}")
        
        # Simple intent detection for GitHub-related queries
        message_lower = user_message.lower()
        
        # Detect repository analysis intent
        if 'github.com' in user_message or '//' in user_message:
            # Extract GitHub URL
            import re
            url_match = re.search(r'github\.com/([^/\s]+/[^/\s]+)', user_message)
            if url_match:
                repo_name = url_match.group(1)
                result = self._handle_repo_analysis({
                    "repository_url": f"https://github.com/{repo_name}",
                    "analysis_depth": "basic"
                })
                
                if result.get("success"):
                    repo = result['analysis']
                    response = f"🔍 **Analysis of {repo['repo_name']}:**\n\n"
                    response += f"📝 **Description:** {repo['description']}\n"
                    response += f"⭐ **Stars:** {repo['stars']} | 🍴 **Forks:** {repo['forks']}\n"
                    response += f"🏷️ **Languages:** {', '.join(repo['languages'][:3]) if repo['languages'] else 'Unknown'}\n"
                    response += f"🏷️ **Topics:** {', '.join(repo['topics'][:5]) if repo['topics'] else 'None'}\n\n"
                    
                    if 'topics' in repo and repo['topics']:
                        response += f"🏷️ **Topics:** {', '.join(repo['topics'])}\n\n"
                        
                    if 'recent_commits' in result:
                        response += f"**Recent Activity:**\n"
                        for commit in result['recent_commits'][:3]:
                            response += f"• {commit['message'][:60]}{'...' if len(commit['message']) > 60 else ''}\n"
                    
                    return {
                        "response": response,
                        "type": "github_analysis",
                        "data": result,
                        "success": True
                    }
                else:
                    return {
                        "response": f"❌ Analysis failed: {result.get('error', 'Repository not found')}",
                        "type": "error",
                        "success": False
                    }
        
        # Detect search intent
        elif any(word in message_lower for word in ['find', 'search', 'look for', 'show me', 'repositories', 'repos', 'projects']):
            # Extract query from natural language
            query = self._extract_search_query(user_message)
            
            result = self._handle_github_search({
                "query": query,
                "language": None,
                "max_results": 5
            })
            
            if result.get("success"):
                response = f"🔍 **Found {len(result['results'])} GitHub repositories for '{query}':**\n\n"
                for i, repo in enumerate(result['results'], 1):
                    response += f"**{i}. {repo['name']}**\n"
                    response += f"   📝 {repo['description'][:100]}{'...' if len(repo['description']) > 100 else ''}\n"
                    response += f"   ⭐ {repo['stars']} stars | 🏷️ {repo['language']}\n"
                    response += f"   🔗 {repo['url']}\n\n"
                    
                return {
                    "response": response,
                    "type": "github_search_results",
                    "data": result,
                    "success": True
                }
            else:
                return {
                    "response": f"❌ Search failed: {result.get('error', 'Unknown error')}",
                    "type": "error",
                    "success": False
                }
        else:
            # Default: treat as search query
            query = user_message
            result = self._handle_github_search({
                "query": query,
                "language": None,
                "max_results": 3
            })
            
            if result.get("success"):
                response = f"🔍 **GitHub search results for '{query}':**\n\n"
                for i, repo in enumerate(result['results'], 1):
                    response += f"**{i}. {repo['name']}**\n"
                    response += f"   📝 {repo['description'][:80]}{'...' if len(repo['description']) > 80 else ''}\n"
                    response += f"   ⭐ {repo['stars']} | 🏷️ {repo['language']}\n\n"
                    
                return {
                    "response": response,
                    "type": "github_search_results",
                    "data": result,
                    "success": True
                }
            else:
                return {
                    "response": f"❌ Search failed: {result.get('error', 'Unknown error')}",
                    "type": "error",
                    "success": False
                }
    
    def _extract_search_query(self, message: str) -> str:
        """Extract search query from natural language message."""
        # Simple extraction - remove common words and keep the core query
        message_lower = message.lower()
        
        # Remove common command words
        remove_words = ['find', 'search', 'look for', 'show me', 'tell me about', 'repositories', 'repos', 'projects']
        
        query = message
        for word in remove_words:
            query = query.replace(word, '').strip()
            
        # Clean up extra spaces
        query = ' '.join(query.split())
        
        return query if query else message

    def health_check(self) -> bool:
        """Return the health status of the agent."""
        try:
            # Test GitHub API access
            rate_limit = self.github.get_rate_limit()
            return rate_limit.core.remaining > 0
        except:
            return False


class A2AArXivResearchAgent(A2ACapableAgent):
    """
    A2A-capable arXiv research agent that can search and analyze academic papers.
    """
    
    def __init__(self, agent_name: str = "ArXivResearcher"):
        super().__init__(
            name=agent_name,
            description="arXiv research agent capable of paper search and analysis",
            provider="DAWN Research Suite",
            version="1.0.0"
        )
        
        # Add research capabilities
        self._add_arxiv_capabilities()
        
        logger.info(f"Initialized {agent_name} with arXiv API access")
        
    def _add_arxiv_capabilities(self):
        """Add arXiv research capabilities to the agent."""
        
        # arXiv search capability
        search_cap = Capability(
            capability_type="arxiv_search",
            name="arXiv Paper Search",
            description="Search arXiv for relevant academic papers",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query for arXiv papers",
                    "required": True
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            }
        )
        self.add_capability(search_cap)
        
        # Paper summary capability
        summary_cap = Capability(
            capability_type="arxiv_paper_summary",
            name="arXiv Paper Summary",
            description="Retrieve and summarize a specific arXiv paper",
            parameters={
                "paper_id": {
                    "type": "string",
                    "description": "arXiv paper ID to summarize",
                    "required": True
                }
            }
        )
        self.add_capability(summary_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"A2AArXivResearchAgent invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id or cap.name == capability_id or cap.type == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "arxiv_search":
                return self._handle_arxiv_search(inputs)
            elif capability.type == "arxiv_paper_summary":
                return self._handle_paper_summary(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in capability {capability_id}: {e}")
            return {"error": str(e)}
    
    def _handle_arxiv_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Search arXiv for papers matching a query."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        if not query:
            return {"error": "No query provided for arXiv search"}
        
        logger.info(f"Searching arXiv for: {query}")
        
        try:
            # Respect arXiv's courtesy rate limit
            time.sleep(1)
            
            # Create the search query
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                authors = [author.name for author in paper.authors]
                paper_id = paper.get_short_id()
                results.append({
                    "id": paper_id,
                    "title": paper.title,
                    "authors": authors,
                    "summary": paper.summary.replace("\n", " ")[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                    "categories": paper.categories
                })
            
            logger.info(f"Found {len(results)} papers matching '{query}'")
            return {
                "results": results,
                "count": len(results),
                "query": query,
                "success": True
            }
        except Exception as e:
            error_message = f"arXiv search failed: {str(e)}"
            logger.error(error_message)
            return {"error": error_message}
    
    def _handle_paper_summary(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and analyze a paper from arXiv."""
        paper_id = inputs.get("paper_id", "")
        
        if not paper_id:
            return {"error": "No paper ID provided for summarization"}
        
        logger.info(f"Retrieving arXiv paper: {paper_id}")
        
        try:
            # Respect arXiv's courtesy rate limit
            time.sleep(1)
            
            # Fetch the specific paper by ID
            search = arxiv.Search(id_list=[paper_id])
            papers = list(search.results())
            
            if not papers:
                return {"error": f"Paper with ID {paper_id} not found"}
                
            paper = papers[0]
            
            # Extract metadata
            authors = [author.name for author in paper.authors]
            categories = paper.categories
            summary = paper.summary.replace("\n", " ")
            
            result = {
                "paper_id": paper_id,
                "title": paper.title,
                "authors": authors,
                "summary": summary,
                "categories": categories,
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                "url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": paper.pdf_url,
                "success": True
            }
            
            logger.info(f"Successfully retrieved paper: {paper.title}")
            return result
            
        except Exception as e:
            error_message = f"Paper retrieval failed: {str(e)}"
            logger.error(error_message)
            return {"error": error_message}
    
    async def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Chat interface for natural language interaction with arXiv research agent.
        
        Args:
            user_message: Natural language query from user
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"arXiv agent received chat message: {user_message}")
        
        # Simple intent detection for arXiv-related queries
        message_lower = user_message.lower()
        
        # Detect search intent
        if any(word in message_lower for word in ['find', 'search', 'look for', 'show me', 'papers', 'research', 'articles']):
            # Extract query from natural language
            query = self._extract_search_query(user_message)
            
            result = self._handle_arxiv_search({
                "query": query,
                "max_results": 5
            })
            
            if result.get("success"):
                response = f"📚 **Found {len(result['results'])} arXiv papers for '{query}':**\n\n"
                for i, paper in enumerate(result['results'], 1):
                    response += f"**{i}. {paper['title']}**\n"
                    response += f"   👥 Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
                    response += f"   📅 Published: {paper['published']}\n"
                    response += f"   🏷️ Categories: {', '.join(paper['categories'])}\n"
                    response += f"   📝 {paper['summary'][:150]}...\n"
                    response += f"   🔗 {paper['pdf_url']}\n\n"
                    
                return {
                    "response": response,
                    "type": "arxiv_search_results",
                    "data": result,
                    "success": True
                }
            else:
                return {
                    "response": f"❌ Search failed: {result.get('error', 'Unknown error')}",
                    "type": "error",
                    "success": False
                }
                
        # Detect summary intent  
        elif any(word in message_lower for word in ['summarize', 'summary', 'analyze', 'explain']):
            if 'arxiv.org' in user_message:
                # Extract arXiv URL or ID
                import re
                url_match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', user_message)
                if url_match:
                    arxiv_id = url_match.group(1)
                    result = self._handle_paper_summary({"arxiv_id": arxiv_id})
                    
                    if result.get("success"):
                        paper = result['paper_info']
                        response = f"📄 **Summary of {paper['title']}:**\n\n"
                        response += f"👥 **Authors:** {', '.join(paper['authors'])}\n"
                        response += f"📅 **Published:** {paper['published']}\n"
                        response += f"🏷️ **Categories:** {', '.join(paper['categories'])}\n\n"
                        response += f"**Abstract Summary:**\n{paper['summary']}\n\n"
                        response += f"**Key Insights:**\n{paper['key_insights']}\n"
                        
                        return {
                            "response": response,
                            "type": "arxiv_analysis",
                            "data": result,
                            "success": True
                        }
                    else:
                        return {
                            "response": f"❌ Summary failed: {result.get('error', 'Unknown error')}",
                            "type": "error",
                            "success": False
                        }
            else:
                return {
                    "response": "🤔 Please provide an arXiv URL for summary (e.g., https://arxiv.org/abs/2301.00001)",
                    "type": "instruction",
                    "success": False
                }
        else:
            # Default: treat as search query
            query = user_message
            result = self._handle_arxiv_search({
                "query": query,
                "max_results": 3
            })
            
            if result.get("success"):
                response = f"📚 **arXiv search results for '{query}':**\n\n"
                for i, paper in enumerate(result['results'], 1):
                    response += f"**{i}. {paper['title']}**\n"
                    response += f"   👥 {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}\n"
                    response += f"   📅 {paper['published']}\n\n"
                    
                return {
                    "response": response,
                    "type": "arxiv_search_results", 
                    "data": result,
                    "success": True
                }
            else:
                return {
                    "response": f"❌ Search failed: {result.get('error', 'Unknown error')}",
                    "type": "error",
                    "success": False
                }
    
    def _extract_search_query(self, message: str) -> str:
        """Extract search query from natural language message."""
        # Simple extraction - remove common words and keep the core query
        message_lower = message.lower()
        
        # Remove common command words
        remove_words = ['find', 'search', 'look for', 'show me', 'tell me about', 'papers', 'research', 'articles']
        
        query = message
        for word in remove_words:
            query = query.replace(word, '').strip()
            
        # Clean up extra spaces
        query = ' '.join(query.split())
        
        return query if query else message

    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


class A2ASynthesisAgent(A2ACapableAgent):
    """
    A2A-capable synthesis agent that can combine information from multiple sources.
    """
    
    def __init__(self, agent_name: str = "SynthesisAgent"):
        super().__init__(
            name=agent_name,
            description="Synthesis agent capable of combining and analyzing information from multiple sources",
            provider="DAWN Research Suite",
            version="1.0.0"
        )
        
        # Add synthesis capabilities
        self._add_synthesis_capabilities()
        
        logger.info(f"Initialized {agent_name} with synthesis capabilities")
        
    def _add_synthesis_capabilities(self):
        """Add synthesis capabilities to the agent."""
        
        # Information synthesis capability
        synthesis_cap = Capability(
            capability_type="information_synthesis",
            name="Information Synthesis",
            description="Combine and analyze information from multiple sources",
            parameters={
                "sources": {
                    "type": "array",
                    "description": "Array of information sources to synthesize",
                    "required": True
                },
                "question": {
                    "type": "string",
                    "description": "Research question to focus the synthesis",
                    "required": True
                },
                "source_type": {
                    "type": "string",
                    "description": "Type of sources being synthesized",
                    "default": "mixed"
                }
            }
        )
        self.add_capability(synthesis_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"A2ASynthesisAgent invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id or cap.name == capability_id or cap.type == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "information_synthesis":
                return self._handle_synthesis(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in capability {capability_id}: {e}")
            return {"error": str(e)}
    
    def _handle_synthesis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize information from multiple sources."""
        sources = inputs.get("sources", [])
        question = inputs.get("question", "")
        source_type = inputs.get("source_type", "mixed")
        
        if not sources:
            return {"error": "No sources provided for synthesis"}
        if not question:
            return {"error": "No question provided for synthesis"}
        
        logger.info(f"Synthesizing information for question: {question}")
        logger.info(f"Using {len(sources)} sources of type: {source_type}")
        
        # If OpenAI is not configured, provide basic synthesis
        if not client:
            return self._basic_synthesis(sources, question, source_type)
        
        # Create context from sources
        context = f"Research Question: {question}\n\nSources:\n\n"
        
        for i, source in enumerate(sources, 1):
            if source_type == "github":
                context += f"Source {i} (GitHub Repository):\n"
                context += f"Name: {source.get('repo_name', 'Unknown')}\n"
                context += f"Languages: {', '.join(source.get('languages', []))}\n"
                context += f"Description: {source.get('description', 'No description')}\n"
                context += f"Stars: {source.get('stars', 0)}\n"
            elif source_type == "arxiv":
                context += f"Source {i} (arXiv Paper):\n"
                context += f"Title: {source.get('title', 'Unknown')}\n"
                context += f"Authors: {', '.join(source.get('authors', []))}\n"
                context += f"Summary: {source.get('summary', 'No summary')}\n"
            else:
                context += f"Source {i}:\n{str(source)}\n"
            context += "\n"
        
        try:
            # Call OpenAI for synthesis
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Based on the following sources, provide a comprehensive synthesis to answer the research question. Focus on key insights and patterns across sources.\n\n{context}"
                }],
                temperature=0.3,
                max_tokens=800
            )
            
            synthesis = response.choices[0].message.content.strip()
            
            return {
                "synthesis": synthesis,
                "question": question,
                "source_count": len(sources),
                "source_type": source_type,
                "success": True
            }
        except Exception as e:
            logger.error(f"OpenAI synthesis failed: {e}")
            return self._basic_synthesis(sources, question, source_type)
    
    def _basic_synthesis(self, sources: List[Dict], question: str, source_type: str) -> Dict[str, Any]:
        """Provide basic synthesis when OpenAI is not available."""
        synthesis = f"Research Question: {question}\n\n"
        synthesis += f"Based on {len(sources)} {source_type} sources:\n\n"
        
        if source_type == "github":
            languages = set()
            topics = set()
            for source in sources:
                languages.update(source.get('languages', []))
                topics.update(source.get('topics', []))
            
            synthesis += f"Common Technologies: {', '.join(list(languages)[:5])}\n"
            synthesis += f"Key Topics: {', '.join(list(topics)[:5])}\n"
            
        elif source_type == "arxiv":
            categories = set()
            for source in sources:
                categories.update(source.get('categories', []))
            
            synthesis += f"Research Areas: {', '.join(list(categories)[:5])}\n"
        
        synthesis += "\nNote: Basic synthesis provided. Enable OpenAI integration for detailed analysis."
        
        return {
            "synthesis": synthesis,
            "question": question,
            "source_count": len(sources),
            "source_type": source_type,
            "success": True
        }
    
    async def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Chat interface for natural language interaction with synthesis agent.
        
        Args:
            user_message: Natural language query from user
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Synthesis agent received chat message: {user_message}")
        
        # Simple intent detection for synthesis-related queries
        message_lower = user_message.lower()
        
        # Check if user is asking for synthesis/comparison
        if any(word in message_lower for word in ['synthesize', 'compare', 'analyze', 'combine', 'summary', 'insights']):
            # For demo purposes, we'll ask user to provide data or suggest they use the orchestrator
            return {
                "response": ("🧠 **Synthesis Agent Ready!**\n\n"
                           "I can synthesize information from multiple sources, but I need structured data to work with.\n\n"
                           "**For better results, try:**\n"
                           "• Use `@orchestrator` for multi-source research that I can then synthesize\n" 
                           "• Provide specific data sources you'd like me to analyze\n\n"
                           "**What I can synthesize:**\n"
                           "• GitHub repository data + arXiv papers\n"
                           "• Multiple research sources\n"
                           "• Comparative analysis across different domains"),
                "type": "synthesis_help",
                "success": True
            }
        else:
            # Suggest using orchestrator for complex queries
            return {
                "response": ("🤔 **I'm the synthesis specialist!**\n\n"
                           f"For the query '{user_message}', you might want to try:\n\n"
                           f"• `@orchestrator {user_message}` - Let the orchestrator gather data from multiple agents, then I'll synthesize it\n"
                           f"• `@github {user_message}` + `@arxiv {user_message}` - Get data from individual agents first\n\n"
                           "I'm here to combine and analyze information once you have multiple sources!"),
                "type": "synthesis_suggestion",
                "success": True
            }

    async def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Chat interface for natural language interaction with research orchestrator.
        This agent can delegate to specialized agents based on query analysis.
        
        Args:
            user_message: Natural language query from user
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Research orchestrator received chat message: {user_message}")
        
        # Analyze query to determine which agents to involve
        message_lower = user_message.lower()
        
        # Define agent URLs (these should match the running servers)
        agent_urls = {
            "github_agent": "http://localhost:8081",
            "arxiv_agent": "http://localhost:8082", 
            "synthesis_agent": "http://localhost:8083"
        }
        
        # Smart delegation logic
        needs_github = any(word in message_lower for word in [
            'github', 'repository', 'repos', 'code', 'implementation', 'library', 'framework'
        ])
        
        needs_arxiv = any(word in message_lower for word in [
            'paper', 'research', 'study', 'academic', 'arxiv', 'publication', 'theory', 'algorithm'
        ])
        
        needs_synthesis = any(word in message_lower for word in [
            'compare', 'comparison', 'versus', 'vs', 'difference', 'similar', 'analyze', 'synthesis'
        ])
        
        # If query mentions both domains or asks for comparison, use multi-agent workflow
        if (needs_github and needs_arxiv) or needs_synthesis or 'both' in message_lower:
            try:
                logger.info("🔄 Using multi-agent workflow for comprehensive research")
                
                result = await self._handle_multi_agent_workflow({
                    "research_question": user_message,
                    "agent_urls": agent_urls
                })
                
                if result.get("success"):
                    response = f"🎯 **Multi-Agent Research Results for: {user_message}**\n\n"
                    
                    # GitHub results
                    if result.get("github_results") and result["github_results"].get("success"):
                        github_count = len(result["github_results"].get("results", []))
                        response += f"🔍 **GitHub Research ({github_count} repositories):**\n"
                        for repo in result["github_results"]["results"][:3]:
                            response += f"• **{repo['name']}** ⭐ {repo['stars']} - {repo['description'][:100]}...\n"
                        response += "\n"
                    
                    # arXiv results  
                    if result.get("arxiv_results") and result["arxiv_results"].get("success"):
                        arxiv_count = len(result["arxiv_results"].get("results", []))
                        response += f"📚 **arXiv Research ({arxiv_count} papers):**\n"
                        for paper in result["arxiv_results"]["results"][:3]:
                            response += f"• **{paper['title']}** by {', '.join(paper['authors'][:2])}\n"
                        response += "\n"
                    
                    # Synthesis results
                    if result.get("synthesis_results") and result["synthesis_results"].get("success"):
                        synthesis = result["synthesis_results"].get("synthesis", "")
                        response += f"🧠 **Synthesis & Analysis:**\n{synthesis}\n\n"
                        
                        insights = result["synthesis_results"].get("key_insights", [])
                        if insights:
                            response += "💡 **Key Insights:**\n"
                            for insight in insights[:3]:
                                response += f"• {insight}\n"
                    
                    response += "\n✅ **Research completed using GitHub + arXiv + Synthesis agents**"
                    
                    return {
                        "response": response,
                        "type": "multi_agent_research",
                        "data": result,
                        "success": True,
                        "agents_used": ["github", "arxiv", "synthesis"]
                    }
                else:
                    errors = result.get("errors", ["Unknown error"])
                    return {
                        "response": f"❌ **Multi-agent research failed:**\n" + "\n".join(f"• {error}" for error in errors),
                        "type": "error",
                        "success": False
                    }
                    
            except Exception as e:
                logger.error(f"Multi-agent workflow error: {e}")
                return {
                    "response": f"❌ **Orchestration failed:** {str(e)}",
                    "type": "error", 
                    "success": False
                }
        
        # If primarily GitHub-focused, delegate to GitHub agent
        elif needs_github and not needs_arxiv:
            try:
                logger.info("➡️ Delegating to GitHub agent")
                github_agent_id = await self.connect_to_a2a_agent(agent_urls["github_agent"], "github_researcher")
                
                # Find search capability
                capabilities = await self.discover_remote_a2a_capabilities(github_agent_id)
                search_cap = None
                for cap in capabilities:
                    if "search" in cap.get("name", "").lower():
                        search_cap = cap
                        break
                
                if search_cap:
                    result = await self.invoke_remote_a2a_capability(
                        github_agent_id,
                        search_cap["name"],
                        {"query": user_message, "max_results": 5}
                    )
                    
                    if result.get("success"):
                        response = f"🔍 **GitHub Research (via orchestrator):** {user_message}\n\n"
                        for i, repo in enumerate(result["results"], 1):
                            response += f"**{i}. {repo['name']}** ⭐ {repo['stars']}\n"
                            response += f"   {repo['description'] or 'No description'}\n\n"
                        response += "✅ **Delegated to GitHub agent**"
                        
                        return {
                            "response": response,
                            "type": "delegated_github",
                            "data": result,
                            "success": True,
                            "agents_used": ["github"]
                        }
                
                return {
                    "response": "❌ Failed to find GitHub search capability",
                    "type": "error",
                    "success": False
                }
                
            except Exception as e:
                logger.error(f"GitHub delegation error: {e}")
                return {
                    "response": f"❌ **GitHub delegation failed:** {str(e)}",
                    "type": "error",
                    "success": False
                }
        
        # If primarily arXiv-focused, delegate to arXiv agent
        elif needs_arxiv and not needs_github:
            try:
                logger.info("➡️ Delegating to arXiv agent")
                arxiv_agent_id = await self.connect_to_a2a_agent(agent_urls["arxiv_agent"], "arxiv_researcher")
                
                # Find search capability
                capabilities = await self.discover_remote_a2a_capabilities(arxiv_agent_id)
                search_cap = None
                for cap in capabilities:
                    if "search" in cap.get("name", "").lower():
                        search_cap = cap
                        break
                
                if search_cap:
                    result = await self.invoke_remote_a2a_capability(
                        arxiv_agent_id,
                        search_cap["name"],
                        {"query": user_message, "max_results": 5}
                    )
                    
                    if result.get("success"):
                        response = f"📚 **arXiv Research (via orchestrator):** {user_message}\n\n"
                        for i, paper in enumerate(result["results"], 1):
                            response += f"**{i}. {paper['title']}**\n"
                            response += f"   👥 {', '.join(paper['authors'][:2])}\n"
                            response += f"   📅 {paper['published']}\n\n"
                        response += "✅ **Delegated to arXiv agent**"
                        
                        return {
                            "response": response,
                            "type": "delegated_arxiv",
                            "data": result,
                            "success": True,
                            "agents_used": ["arxiv"]
                        }
                
                return {
                    "response": "❌ Failed to find arXiv search capability",
                    "type": "error",
                    "success": False
                }
                
            except Exception as e:
                logger.error(f"arXiv delegation error: {e}")
                return {
                    "response": f"❌ **arXiv delegation failed:** {str(e)}",
                    "type": "error",
                    "success": False
                }
        
        # Default: provide guidance on using the orchestrator
        else:
            return {
                "response": ("🎯 **Research Orchestrator Ready!**\n\n"
                           f"I can help research '{user_message}' using multiple specialized agents:\n\n"
                           "**🔍 What I can do:**\n"
                           "• **GitHub research**: repositories, code implementations\n"
                           "• **arXiv research**: academic papers, theories\n" 
                           "• **Multi-agent synthesis**: compare and analyze across sources\n\n"
                           "**💡 Try asking:**\n"
                           "• 'Compare PyTorch and TensorFlow repositories and research papers'\n"
                           "• 'Find transformer architecture implementations and papers'\n"
                           "• 'Research reinforcement learning frameworks and theory'\n\n"
                           "I'll automatically coordinate the right agents for your query!"),
                "type": "orchestrator_help",
                "success": True
            }

    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


class A2AResearchOrchestrator(A2ACapableAgent):
    """
    A2A-capable research orchestrator that coordinates multiple research agents.
    This agent can delegate tasks to specialized research agents via A2A protocol.
    """
    
    def __init__(self, agent_name: str = "ResearchOrchestrator"):
        super().__init__(
            name=agent_name,
            description="Research orchestrator that coordinates multiple specialized research agents",
            provider="DAWN Research Suite",
            version="1.0.0"
        )
        
        # Add orchestration capabilities
        self._add_orchestration_capabilities()
        
        logger.info(f"Initialized {agent_name} with orchestration capabilities")
        
    def _add_orchestration_capabilities(self):
        """Add orchestration capabilities to the agent."""
        
        # Multi-agent research workflow
        workflow_cap = Capability(
            capability_type="multi_agent_research_workflow",
            name="Multi-Agent Research Workflow",
            description="Execute a comprehensive research workflow using multiple specialized agents",
            parameters={
                "research_question": {
                    "type": "string",
                    "description": "The research question to investigate",
                    "required": True
                },
                "agent_urls": {
                    "type": "object",
                    "description": "URLs of specialized research agents",
                    "properties": {
                        "github_agent": {"type": "string"},
                        "arxiv_agent": {"type": "string"},
                        "synthesis_agent": {"type": "string"}
                    },
                    "required": ["github_agent", "arxiv_agent", "synthesis_agent"]
                }
            }
        )
        self.add_capability(workflow_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        # All orchestration capabilities require async execution
        return {"error": "This capability requires async execution. Use invoke_async instead."}
            
    async def invoke_async(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of invoke for capabilities that require remote calls."""
        logger.info(f"A2AResearchOrchestrator async invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id or cap.name == capability_id or cap.type == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "multi_agent_research_workflow":
                return await self._handle_multi_agent_workflow(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in async capability {capability_id}: {e}")
            return {"error": str(e)}
    
    async def _handle_multi_agent_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-agent research workflow."""
        research_question = inputs.get("research_question", "")
        agent_urls = inputs.get("agent_urls", {})
        
        if not research_question:
            return {"error": "No research question provided"}
        
        required_agents = ["github_agent", "arxiv_agent", "synthesis_agent"]
        for agent_type in required_agents:
            if agent_type not in agent_urls:
                return {"error": f"Missing {agent_type} URL"}
        
        logger.info(f"Starting multi-agent research workflow for: {research_question}")
        
        results = {
            "research_question": research_question,
            "github_results": None,
            "arxiv_results": None,
            "synthesis_results": None,
            "errors": []
        }
        
        # Step 1: Connect to GitHub research agent and get repository data
        try:
            logger.info("Connecting to GitHub research agent...")
            github_agent_id = await self.connect_to_a2a_agent(
                agent_urls["github_agent"], 
                "github_researcher"
            )
            
            # Discover GitHub capabilities
            github_capabilities = await self.discover_remote_a2a_capabilities(github_agent_id)
            
            # Find search capability
            search_cap = None
            for cap in github_capabilities:
                if "search" in cap.get("name", "").lower() or "github_search" in cap.get("type", ""):
                    search_cap = cap
                    break
            
            if search_cap:
                github_result = await self.invoke_remote_a2a_capability(
                    github_agent_id,
                    search_cap["name"],
                    {
                        "query": research_question,
                        "max_results": 3
                    }
                )
                results["github_results"] = github_result
                logger.info("Successfully retrieved GitHub research data")
            else:
                results["errors"].append("GitHub search capability not found")
                
        except Exception as e:
            error_msg = f"GitHub agent error: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # Step 2: Connect to arXiv research agent and get paper data
        try:
            logger.info("Connecting to arXiv research agent...")
            arxiv_agent_id = await self.connect_to_a2a_agent(
                agent_urls["arxiv_agent"], 
                "arxiv_researcher"
            )
            
            # Discover arXiv capabilities
            arxiv_capabilities = await self.discover_remote_a2a_capabilities(arxiv_agent_id)
            
            # Find search capability
            search_cap = None
            for cap in arxiv_capabilities:
                if "search" in cap.get("name", "").lower() or "arxiv_search" in cap.get("type", ""):
                    search_cap = cap
                    break
            
            if search_cap:
                arxiv_result = await self.invoke_remote_a2a_capability(
                    arxiv_agent_id,
                    search_cap["name"],
                    {
                        "query": research_question,
                        "max_results": 3
                    }
                )
                results["arxiv_results"] = arxiv_result
                logger.info("Successfully retrieved arXiv research data")
            else:
                results["errors"].append("arXiv search capability not found")
                
        except Exception as e:
            error_msg = f"arXiv agent error: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # Step 3: Connect to synthesis agent and synthesize results
        if results["github_results"] or results["arxiv_results"]:
            try:
                logger.info("Connecting to synthesis agent...")
                synthesis_agent_id = await self.connect_to_a2a_agent(
                    agent_urls["synthesis_agent"], 
                    "synthesis_agent"
                )
                
                # Discover synthesis capabilities
                synthesis_capabilities = await self.discover_remote_a2a_capabilities(synthesis_agent_id)
                
                # Find synthesis capability
                synthesis_cap = None
                for cap in synthesis_capabilities:
                    if "synthesis" in cap.get("name", "").lower():
                        synthesis_cap = cap
                        break
                
                if synthesis_cap:
                    # Prepare sources for synthesis
                    sources = []
                    if results["github_results"] and results["github_results"].get("success"):
                        for repo in results["github_results"].get("results", []):
                            sources.append({
                                "type": "github",
                                "repo_name": repo.get("name"),
                                "description": repo.get("description"),
                                "stars": repo.get("stars"),
                                "languages": []  # Would be filled from detailed analysis
                            })
                    
                    if results["arxiv_results"] and results["arxiv_results"].get("success"):
                        for paper in results["arxiv_results"].get("results", []):
                            sources.append({
                                "type": "arxiv",
                                "title": paper.get("title"),
                                "authors": paper.get("authors"),
                                "summary": paper.get("summary"),
                                "categories": paper.get("categories")
                            })
                    
                    if sources:
                        synthesis_result = await self.invoke_remote_a2a_capability(
                            synthesis_agent_id,
                            synthesis_cap["name"],
                            {
                                "sources": sources,
                                "question": research_question,
                                "source_type": "mixed"
                            }
                        )
                        results["synthesis_results"] = synthesis_result
                        logger.info("Successfully synthesized research results")
                    else:
                        results["errors"].append("No valid sources found for synthesis")
                else:
                    results["errors"].append("Synthesis capability not found")
                    
            except Exception as e:
                error_msg = f"Synthesis agent error: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        else:
            results["errors"].append("No research data available for synthesis")
        
        # Add success indicator
        results["success"] = len(results["errors"]) == 0 and (results["github_results"] or results["arxiv_results"])
        
        logger.info(f"Multi-agent research workflow completed with {len(results['errors'])} errors")
        return results
    
    async def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Chat interface for natural language interaction with GitHub research agent.
        
        Args:
            user_message: Natural language query from user
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"GitHub agent received chat message: {user_message}")
        
        # Simple intent detection for GitHub-related queries
        message_lower = user_message.lower()
        
        # Detect search intent
        if any(word in message_lower for word in ['find', 'search', 'look for', 'show me', 'repositories', 'repos']):
            # Extract query from natural language
            query = self._extract_search_query(user_message)
            
            result = self._handle_github_search({
                "query": query,
                "max_results": 5
            })
            
            if result.get("success"):
                response = f"🔍 **Found {len(result['results'])} GitHub repositories for '{query}':**\n\n"
                for i, repo in enumerate(result['results'], 1):
                    response += f"**{i}. {repo['name']}** ⭐ {repo['stars']}\n"
                    response += f"   {repo['description'] or 'No description'}\n"
                    response += f"   🔗 {repo['url']}\n\n"
                    
                return {
                    "response": response,
                    "type": "github_search_results",
                    "data": result,
                    "success": True
                }
            else:
                return {
                    "response": f"❌ Search failed: {result.get('error', 'Unknown error')}",
                    "type": "error",
                    "success": False
                }
                
        # Detect analysis intent
        elif any(word in message_lower for word in ['analyze', 'analysis', 'examine', 'details', 'about']):
            if 'github.com' in user_message:
                # Extract GitHub URL
                import re
                url_match = re.search(r'https?://github\.com/[^\s]+', user_message)
                if url_match:
                    repo_url = url_match.group()
                    result = self._handle_repo_analysis({"repo_url": repo_url})
                    
                    if result.get("success"):
                        repo = result['repository_info']
                        response = f"📊 **Analysis of {repo['name']}:**\n\n"
                        response += f"⭐ **Stars:** {repo['stars']}\n"
                        response += f"🍴 **Forks:** {repo['forks']}\n"
                        response += f"👥 **Contributors:** {repo['contributors_count']}\n"
                        response += f"📅 **Last Updated:** {repo['last_updated']}\n"
                        if repo['languages']:
                            response += f"💻 **Languages:** {', '.join(repo['languages'])}\n"
                        response += f"\n📝 **Description:** {repo['description'] or 'No description'}\n"
                        
                        return {
                            "response": response,
                            "type": "github_analysis",
                            "data": result,
                            "success": True
                        }
                    else:
                        return {
                            "response": f"❌ Analysis failed: {result.get('error', 'Unknown error')}",
                            "type": "error", 
                            "success": False
                        }
            else:
                return {
                    "response": "🤔 Please provide a GitHub repository URL for analysis (e.g., https://github.com/owner/repo)",
                    "type": "instruction",
                    "success": False
                }
        else:
            # Default: treat as search query
            query = user_message
            result = self._handle_github_search({
                "query": query,
                "max_results": 3
            })
            
            if result.get("success"):
                response = f"🔍 **GitHub search results for '{query}':**\n\n"
                for i, repo in enumerate(result['results'], 1):
                    response += f"**{i}. {repo['name']}** ⭐ {repo['stars']}\n"
                    response += f"   {repo['description'] or 'No description'}\n\n"
                    
                return {
                    "response": response,
                    "type": "github_search_results",
                    "data": result,
                    "success": True
                }
            else:
                return {
                    "response": f"❌ Search failed: {result.get('error', 'Unknown error')}",
                    "type": "error",
                    "success": False
                }
    
    def _extract_search_query(self, message: str) -> str:
        """Extract search query from natural language message."""
        # Simple extraction - remove common words and keep the core query
        message_lower = message.lower()
        
        # Remove common command words
        remove_words = ['find', 'search', 'look for', 'show me', 'tell me about', 'repositories', 'repos', 'projects']
        
        query = message
        for word in remove_words:
            query = query.replace(word, '').strip()
            
        # Clean up extra spaces
        query = ' '.join(query.split())
        
        return query if query else message

    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


# Interactive Multi-Agent Chat Demo
class InteractiveA2ADemo:
    """Interactive demo that starts all agents and provides chat interface."""
    
    def __init__(self):
        self.agents = {}
        self.running_servers = []
        
    async def start_all_agents(self):
        """Start all research agents as A2A servers in the background."""
        print("🚀 Starting all A2A research agents...")
        
        agents_config = [
            ("github", A2AGitHubResearchAgent("GitHubServer"), 8081),
            ("arxiv", A2AArXivResearchAgent("ArXivServer"), 8082),
            ("synthesis", A2ASynthesisAgent("SynthesisServer"), 8083),
            ("orchestrator", A2AResearchOrchestrator("OrchestratorServer"), 8080)
        ]
        
        for agent_name, agent_instance, port in agents_config:
            try:
                logger.info(f"Starting {agent_name} agent on port {port}...")
                server = await agent_instance.start_a2a_server(port=port)
                self.agents[agent_name] = agent_instance
                self.running_servers.append(server)
                print(f"✅ {agent_name.capitalize()} agent ready on port {port}")
                
                # Brief delay to allow server to fully start
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to start {agent_name} agent: {e}")
                print(f"❌ {agent_name.capitalize()} agent failed to start")
        
        print(f"\n🎉 Started {len(self.agents)} agents successfully!")
        
    async def stop_all_agents(self):
        """Stop all running A2A servers."""
        print("\n🛑 Stopping all agents...")
        for agent_name, agent in self.agents.items():
            try:
                await agent.stop_a2a_server()
                print(f"✅ Stopped {agent_name} agent")
            except Exception as e:
                logger.error(f"Error stopping {agent_name} agent: {e}")
        
    async def interactive_chat(self):
        """Run the interactive chat interface."""
        print("\n" + "="*70)
        print("🤖 INTERACTIVE MULTI-AGENT RESEARCH CHAT")
        print("="*70)
        print("\n📋 Available agents:")
        print("• @github      - Direct GitHub repository research")
        print("• @arxiv       - Direct arXiv academic paper research") 
        print("• @synthesis   - Direct synthesis and analysis")
        print("• @orchestrator - Multi-agent coordination & delegation")
        print("\n💡 Example queries:")
        print("• @github pytorch neural networks")
        print("• @arxiv transformer attention mechanisms") 
        print("• @orchestrator compare pytorch and tensorflow research")
        print("\n⌨️  Type your message (or 'quit' to exit):")
        print("-" * 70)
        
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not user_input:
                    continue
                    
                # Parse agent selection and message
                if user_input.startswith('@'):
                    parts = user_input.split(' ', 1)
                    if len(parts) < 2:
                        print("❌ Please provide a message after the agent name")
                        continue
                        
                    agent_name = parts[0][1:]  # Remove @
                    message = parts[1]
                    
                    if agent_name not in self.agents:
                        print(f"❌ Unknown agent: {agent_name}")
                        print(f"Available: {', '.join(self.agents.keys())}")
                        continue
                        
                    # Get agent and process chat
                    agent = self.agents[agent_name]
                    print(f"\n🤖 {agent_name.capitalize()} agent thinking...")
                    
                    try:
                        start_time = time.time()
                        result = await agent.chat(message)
                        end_time = time.time()
                        
                        print(f"\n{result['response']}")
                        
                        # Show metadata
                        agents_used = result.get('agents_used', [agent_name])
                        print(f"\n📊 Response time: {end_time - start_time:.2f}s | Agents used: {', '.join(agents_used)}")
                        
                        if not result.get('success'):
                            print(f"⚠️  Result type: {result.get('type', 'unknown')}")
                            
                    except Exception as e:
                        logger.error(f"Chat error with {agent_name}: {e}")
                        print(f"❌ Error communicating with {agent_name} agent: {str(e)}")
                        
                else:
                    print("❌ Please start your message with an agent name (e.g., @github, @orchestrator)")
                    print("Available agents: " + ", ".join(f"@{name}" for name in self.agents.keys()))
                    
            except (KeyboardInterrupt, EOFError):
                break
                
        print("\n👋 Thanks for using the A2A Research Chat Demo!")


async def run_canned_tests(demo):
    """Run a series of canned test commands to validate functionality."""
    print("\n" + "="*70)
    print("🧪 RUNNING CANNED TESTS")
    print("="*70)
    
    test_commands = [
        ("@github", "search for pytorch neural networks"),
        ("@arxiv", "find papers about transformer attention mechanisms"),
        ("@synthesis", "combine information about machine learning"),
        ("@orchestrator", "compare pytorch and tensorflow research trends"),
        ("@github", "analyze https://github.com/agntcy/oasf repository")
    ]
    
    for i, (agent_name, query) in enumerate(test_commands, 1):
        print(f"\n📋 Test {i}/5: {agent_name} {query}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            
            agent_key = agent_name[1:]  # Remove @ prefix
            if agent_key in demo.agents:
                response = await demo.agents[agent_key].chat(query)
            else:
                print(f"❌ Agent {agent_key} not found")
                continue
            
            end_time = time.time()
            
            if response.get('success'):
                print(f"✅ Success ({end_time - start_time:.1f}s)")
                print(f"📝 Response: {response['response'][:200]}...")
                if len(response['response']) > 200:
                    print("   [Response truncated for brevity]")
            else:
                print(f"❌ Failed ({end_time - start_time:.1f}s)")
                print(f"📝 Error: {response.get('response', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    print("\n" + "="*70)
    print("🏁 CANNED TESTS COMPLETED")
    print("="*70)

async def main():
    """Main entry point for the interactive demo."""
    import sys
    import time
    
    # Check for test mode
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    print("🎯 A2A Interactive Multi-Agent Research Demo")
    print("=" * 50)
    
    if test_mode:
        print("🧪 Running in TEST MODE - will execute canned commands and exit")
    
    demo = InteractiveA2ADemo()
    
    try:
        # Start all agents
        await demo.start_all_agents()
        
        if len(demo.agents) == 0:
            print("❌ No agents started successfully. Exiting...")
            return
            
        if test_mode:
            # Run canned tests
            await run_canned_tests(demo)
        else:
            # Run interactive chat
            await demo.interactive_chat()
        
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo error: {e}")
    finally:
        # Clean shutdown
        await demo.stop_all_agents()
        print("🏁 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())