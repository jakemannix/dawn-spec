"""
Research Agent Demo implementing DAWN architecture.

This demo implements a research agent that can answer R&D questions by orchestrating
specialized sub-agents that access GitHub and arXiv to gather information.
"""
import sys
import os
import json
import uuid
import re
import time
import textwrap
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interfaces import IAgent, IPrincipalAgent, IGatewayAgent
from src.config import APIConfig, AgentConfig, check_configuration

# Import OpenAI and other required libraries
import openai
import requests

# Import research API libraries
from github import Github, GithubException
import arxiv
from duckduckgo_search import DDGS

# Check if OpenAI is configured
if not APIConfig.is_openai_configured():
    print("Error: OpenAI API is not configured.")
    print("Please set OPENAI_API_KEY in .env file.")
    print("You can copy template.env to .env and update with your API key.")
    sys.exit(1)

# Configure OpenAI client
client = openai.OpenAI(
    api_key=APIConfig.OPENAI_API_KEY,
    organization=APIConfig.OPENAI_ORG_ID
)


class BaseAgent(IAgent):
    """Base implementation of the Agent interface."""
    
    def __init__(self, agent_id: str, name: str, description: str, capabilities: List[Dict[str, Any]]):
        self._id = agent_id
        self._name = name
        self._description = description
        self._capabilities = capabilities
        
    def get_info(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return self._capabilities
        
    def health_check(self) -> bool:
        return True
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement invoke method")


class GitHubResearchAgent(BaseAgent):
    """Agent that can search GitHub repositories for information using the real GitHub API."""
    
    def __init__(self):
        super().__init__(
            agent_id=f"github-agent-{str(uuid.uuid4())[:8]}",
            name="GitHub Research Agent",
            description="Searches GitHub repositories for code and documentation",
            capabilities=[
                {
                    "id": "github-search",
                    "type": "github_search",
                    "name": "GitHub Search",
                    "description": "Searches GitHub repositories for relevant information"
                },
                {
                    "id": "github-repo-analysis",
                    "type": "github_repo_analysis",
                    "name": "GitHub Repository Analysis",
                    "description": "Analyzes GitHub repositories for relevant information"
                }
            ]
        )
        # Initialize GitHub client with token if available (higher rate limits)
        self.github_token = APIConfig.GITHUB_TOKEN
        self.github = Github(self.github_token) if self.github_token else Github()
        
        # Print warning about rate limits if no token is provided
        if not self.github_token:
            print("[GitHubResearchAgent] Warning: No GitHub token provided. Rate limits will be strict (60 requests/hour).")
            print("[GitHubResearchAgent] Consider adding a GITHUB_TOKEN to your .env file for better performance.")
        else:
            print(f"[GitHubResearchAgent] GitHub token detected. Using higher rate limits.")
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a GitHub research capability."""
        if capability_id == "github-search":
            return self._search_github(inputs, config)
        elif capability_id == "github-repo-analysis":
            return self._analyze_repo(inputs, config)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
    
    def _search_github(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search GitHub for repositories matching a query using the GitHub API."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        if not query:
            return {"error": "No query provided for GitHub search"}
        
        print(f"[GitHubResearchAgent] Searching GitHub for: {query}")
        
        try:
            # Use PyGithub to search repositories
            repositories = self.github.search_repositories(query, sort="stars")
            results = []
            
            # Process results (limited by max_results)
            count = 0
            for repo in repositories:
                if count >= max_results:
                    break
                    
                results.append({
                    "name": repo.full_name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "updated_at": repo.updated_at.strftime("%Y-%m-%d") if repo.updated_at else "Unknown"
                })
                count += 1
                
            print(f"[GitHubResearchAgent] Found {len(results)} repositories matching '{query}'")
            return {
                "results": results,
                "count": len(results),
                "query": query
            }
        except GithubException as e:
            error_message = f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
            print(f"[GitHubResearchAgent] {error_message}")
            
            # Check for rate limiting
            if e.status == 403 and "rate limit" in str(e).lower():
                print("[GitHubResearchAgent] Rate limit exceeded. Consider adding a GitHub token to your .env file.")
                
            return {"error": error_message}
        except Exception as e:
            error_message = f"GitHub search failed: {str(e)}"
            print(f"[GitHubResearchAgent] {error_message}")
            return {"error": error_message}
    
    def _analyze_repo(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a specific GitHub repository using the GitHub API."""
        repo_url = inputs.get("repo_url", "")
        if not repo_url:
            return {"error": "No repository URL provided for analysis"}
        
        print(f"[GitHubResearchAgent] Analyzing GitHub repository: {repo_url}")
        
        try:
            # Extract owner and repo name from URL
            # Format: https://github.com/owner/repo
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 5 or parts[2] != 'github.com':
                return {"error": f"Invalid GitHub URL: {repo_url}"}
                
            owner = parts[3]
            repo_name = parts[4]
            
            # Fetch repository details
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Get languages
            languages_data = repo.get_languages()
            languages = list(languages_data.keys())
            
            # Get topics
            topics = repo.topics
            
            # Get readme if available
            readme_content = ""
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8')
                readme_content = readme_content[:2000] + "..." if len(readme_content) > 2000 else readme_content
            except:
                readme_content = "No README found"
            
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
            
            # Get top contributors
            contributors = []
            try:
                for contributor in repo.get_contributors()[:5]:
                    contributors.append(contributor.login)
            except:
                contributors = []
            
            # Get top directories/files
            top_contents = []
            try:
                contents = repo.get_contents("")
                for content in contents:
                    if len(top_contents) >= 10:
                        break
                    top_contents.append({
                        "name": content.name,
                        "type": content.type,
                        "path": content.path
                    })
            except:
                top_contents = []
            
            # Determine activity level
            activity_level = "Unknown"
            if recent_commits:
                latest_commit_date = datetime.strptime(recent_commits[0]["date"], "%Y-%m-%d")
                days_since_last_commit = (datetime.now() - latest_commit_date).days
                
                if days_since_last_commit < 7:
                    activity_level = "Very Active"
                elif days_since_last_commit < 30:
                    activity_level = "Active"
                elif days_since_last_commit < 90:
                    activity_level = "Moderately Active"
                else:
                    activity_level = "Less Active"
            
            # Compile analysis results
            analysis = {
                "languages": languages,
                "main_topics": topics,
                "contributors": contributors,
                "key_components": [f"{content['name']} ({content['type']})" for content in top_contents],
                "recent_commits": recent_commits,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "activity_level": activity_level,
                "last_commit": recent_commits[0]["date"] if recent_commits else "Unknown"
            }
            
            # Create a summary
            summary = f"Repository {owner}/{repo_name} has {repo.stargazers_count} stars and {repo.forks_count} forks. "
            summary += f"It primarily uses {', '.join(languages[:3])} and focuses on {', '.join(topics[:3])}. "
            summary += f"The repository is {activity_level.lower()} with the last commit on {analysis['last_commit']}."
            
            print(f"[GitHubResearchAgent] Successfully analyzed repository {owner}/{repo_name}")
            
            return {
                "repo_name": f"{owner}/{repo_name}",
                "analysis": analysis,
                "summary": summary,
                "readme_excerpt": readme_content
            }
            
        except GithubException as e:
            error_message = f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
            print(f"[GitHubResearchAgent] {error_message}")
            return {"error": error_message}
        except Exception as e:
            error_message = f"Repository analysis failed: {str(e)}"
            print(f"[GitHubResearchAgent] {error_message}")
            return {"error": error_message}


class ArXivResearchAgent(BaseAgent):
    """Agent that can search and retrieve papers from arXiv using the real arXiv API."""
    
    def __init__(self):
        super().__init__(
            agent_id=f"arxiv-agent-{str(uuid.uuid4())[:8]}",
            name="arXiv Research Agent",
            description="Searches and retrieves papers from arXiv",
            capabilities=[
                {
                    "id": "arxiv-search",
                    "type": "arxiv_search",
                    "name": "arXiv Search",
                    "description": "Searches arXiv for relevant papers"
                },
                {
                    "id": "arxiv-paper-summary",
                    "type": "arxiv_paper_summary",
                    "name": "arXiv Paper Summary",
                    "description": "Retrieves and summarizes papers from arXiv"
                }
            ]
        )
        print("[ArXivResearchAgent] Initialized. Using arXiv API with courtesy rate limit (~1 request/3 seconds).")
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke an arXiv research capability."""
        if capability_id == "arxiv-search":
            return self._search_arxiv(inputs, config)
        elif capability_id == "arxiv-paper-summary":
            return self._summarize_paper(inputs, config)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
    
    def _search_arxiv(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search arXiv for papers matching a query using the real arXiv API."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        if not query:
            return {"error": "No query provided for arXiv search"}
        
        print(f"[ArXivResearchAgent] Searching arXiv for: {query}")
        
        try:
            # Use the arxiv client to search papers
            # Respecting arXiv's courtesy rate limit with a small delay
            time.sleep(1)  # Be respectful to arXiv API
            
            # Create the search query
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                # Process paper details
                authors = [author.name for author in paper.authors]
                paper_id = paper.get_short_id()
                results.append({
                    "id": paper_id,
                    "title": paper.title,
                    "authors": authors,
                    "summary": paper.summary.replace("\n", " ")[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "categories": paper.categories
                })
            
            print(f"[ArXivResearchAgent] Found {len(results)} papers matching '{query}'")
            return {
                "results": results,
                "count": len(results),
                "query": query
            }
        except Exception as e:
            error_message = f"arXiv search failed: {str(e)}"
            print(f"[ArXivResearchAgent] {error_message}")
            return {"error": error_message}
    
    def _summarize_paper(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve and analyze a paper from arXiv using the real arXiv API."""
        paper_id = inputs.get("paper_id", "")
        
        if not paper_id:
            return {"error": "No paper ID provided for summarization"}
        
        print(f"[ArXivResearchAgent] Retrieving arXiv paper: {paper_id}")
        
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
            
            # Clean and format the summary
            summary = paper.summary.replace("\n", " ")
            
            # Extract key sentences from the summary to use as key findings
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            key_sentences = []
            
            # Choose important sentences (first, last, and ones with key phrases)
            key_phrases = ["propose", "present", "introduce", "novel", "approach", 
                          "method", "result", "demonstrate", "show", "conclude", "contribution"]
            
            # Always include the first sentence
            if sentences and len(sentences) > 0:
                key_sentences.append(sentences[0])
            
            # Look for sentences with key phrases
            for sentence in sentences[1:-1]:  # Skip first and last which we handle separately
                for phrase in key_phrases:
                    if phrase in sentence.lower() and sentence not in key_sentences and len(key_sentences) < 5:
                        key_sentences.append(sentence)
                        break
            
            # Include the last sentence if there's room
            if sentences and len(sentences) > 1 and len(key_sentences) < 5:
                key_sentences.append(sentences[-1])
            
            # If we still don't have enough key sentences, add more from the beginning
            idx = 1
            while len(key_sentences) < min(3, len(sentences)) and idx < len(sentences) - 1:
                if sentences[idx] not in key_sentences:
                    key_sentences.append(sentences[idx])
                idx += 1
            
            # Return the structured paper information
            result = {
                "paper_id": paper_id,
                "title": paper.title,
                "authors": authors,
                "summary": summary,
                "key_findings": key_sentences,
                "categories": categories,
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                "url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": paper.pdf_url
            }
            
            print(f"[ArXivResearchAgent] Successfully retrieved and analyzed paper: {paper.title}")
            return result
            
        except Exception as e:
            error_message = f"Paper retrieval failed: {str(e)}"
            print(f"[ArXivResearchAgent] {error_message}")
            return {"error": error_message}


class WebSearchAgent(BaseAgent):
    """Agent that performs web searches using DuckDuckGo (no API key required)."""
    
    def __init__(self):
        super().__init__(
            agent_id=f"web-search-agent-{str(uuid.uuid4())[:8]}",
            name="Web Search Agent",
            description="Searches the web for relevant information",
            capabilities=[
                {
                    "id": "web-search",
                    "type": "web_search",
                    "name": "Web Search",
                    "description": "Searches the web for relevant information"
                }
            ]
        )
        print("[WebSearchAgent] Initialized. Using DuckDuckGo for web searches (no API key required).")
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a web search capability."""
        if capability_id == "web-search":
            return self._search_web(inputs, config)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
    
    def _search_web(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search the web for information using DuckDuckGo."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 10)
        
        if not query:
            return {"error": "No query provided for web search"}
        
        print(f"[WebSearchAgent] Searching the web for: {query}")
        
        try:
            # Use DuckDuckGo search
            ddgs = DDGS()
            
            # Keep track of time to avoid timeouts
            start_time = time.time()
            timeout = 15  # seconds
            
            # Get results
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "No title"),
                    "body": r.get("body", "No description"),
                    "url": r.get("href", "No URL"),
                    "source": r.get("source", "Unknown")
                })
                
                # Check if we've been searching too long
                if time.time() - start_time > timeout:
                    print(f"[WebSearchAgent] Search timed out after {timeout} seconds")
                    break
                    
                # Check if we have enough results
                if len(results) >= max_results:
                    break
            
            print(f"[WebSearchAgent] Found {len(results)} web results for '{query}'")
            return {
                "results": results,
                "count": len(results),
                "query": query
            }
        except Exception as e:
            error_message = f"Web search failed: {str(e)}"
            print(f"[WebSearchAgent] {error_message}")
            return {"error": error_message}


class SynthesisAgent(BaseAgent):
    """Agent that synthesizes information from multiple sources using LLMs."""
    
    def __init__(self, model: str = None):
        super().__init__(
            agent_id=f"synthesis-agent-{str(uuid.uuid4())[:8]}",
            name="Research Synthesis Agent",
            description="Synthesizes information from multiple sources to answer research questions",
            capabilities=[
                {
                    "id": "information-synthesis",
                    "type": "information_synthesis",
                    "name": "Information Synthesis",
                    "description": "Combines and analyzes information from multiple sources"
                },
                {
                    "id": "research-report-generation",
                    "type": "research_report_generation",
                    "name": "Research Report Generation",
                    "description": "Generates comprehensive research reports"
                }
            ]
        )
        self._model = model or AgentConfig.DEFAULT_MODEL
        print(f"[SynthesisAgent] Using model: {self._model}")
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a synthesis capability."""
        if capability_id == "information-synthesis":
            return self._synthesize_information(inputs, config)
        elif capability_id == "research-report-generation":
            return self._generate_report(inputs, config)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
    
    def _synthesize_information(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synthesize information from multiple sources."""
        sources = inputs.get("sources", [])
        question = inputs.get("question", "")
        
        if not sources:
            return {"error": "No sources provided for synthesis"}
        if not question:
            return {"error": "No question provided for synthesis"}
        
        print(f"[SynthesisAgent] Synthesizing information for question: {question}")
        print(f"[SynthesisAgent] Using {len(sources)} sources")
        
        # Format sources as context
        context = "Sources:\n\n"
        for i, source in enumerate(sources, 1):
            source_type = source.get("type", "unknown")
            if source_type == "github":
                context += f"Source {i} (GitHub Repository):\n"
                context += f"Name: {source.get('repo_name', 'Unknown')}\n"
                if "analysis" in source:
                    analysis = source["analysis"]
                    context += f"Languages: {', '.join(analysis.get('languages', []))}\n"
                    context += f"Topics: {', '.join(analysis.get('main_topics', []))}\n"
                    if "key_components" in analysis:
                        context += "Key Components:\n"
                        for component in analysis["key_components"]:
                            context += f"- {component}\n"
                if "summary" in source:
                    context += f"Summary: {source['summary']}\n"
            elif source_type == "arxiv":
                context += f"Source {i} (arXiv Paper):\n"
                context += f"Title: {source.get('title', 'Unknown')}\n"
                context += f"Paper ID: {source.get('paper_id', 'Unknown')}\n"
                if "summary" in source:
                    context += f"Summary: {source['summary']}\n"
                if "key_findings" in source:
                    context += "Key Findings:\n"
                    for finding in source["key_findings"]:
                        context += f"- {finding}\n"
                if "implications" in source:
                    context += f"Implications: {source['implications']}\n"
            else:
                context += f"Source {i} (Other):\n"
                context += json.dumps(source, indent=2) + "\n"
            context += "\n"
        
        # Create the prompt for synthesis
        prompt = f"""You are a research assistant tasked with synthesizing information from multiple sources to answer a research question.

Question: {question}

{context}

Based on the provided sources, please synthesize a comprehensive answer to the question. 
Focus on key insights, patterns across sources, and important distinctions. 
If the sources provide conflicting information, note the contradictions.
If the sources are insufficient to fully answer the question, indicate what additional information would be needed.

Your synthesis should be well-structured, clear, and directly address the question.
"""
        
        try:
            # Call the OpenAI API for synthesis
            print(f"[SynthesisAgent] Calling {self._model} for synthesis...")
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            synthesis = response.choices[0].message.content.strip()
            
            return {
                "synthesis": synthesis,
                "question": question,
                "source_count": len(sources),
                "model_used": self._model
            }
        except Exception as e:
            print(f"[SynthesisAgent] Error during synthesis: {str(e)}")
            return {"error": f"Synthesis failed: {str(e)}"}
    
    def _generate_report(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a comprehensive research report."""
        synthesis = inputs.get("synthesis", "")
        question = inputs.get("question", "")
        sources = inputs.get("sources", [])
        
        if not synthesis:
            return {"error": "No synthesis provided for report generation"}
        
        print(f"[SynthesisAgent] Generating research report for question: {question}")
        
        try:
            # Create the prompt for report generation
            prompt = f"""You are a research assistant tasked with creating a comprehensive research report.

Question: {question}

Research Synthesis:
{synthesis}

Create a well-structured research report that includes:
1. Executive Summary
2. Background and Context
3. Key Findings
4. Implications and Applications
5. Conclusion and Future Directions
6. References (cite the sources provided)

The report should be professional, informative, and directly address the research question.
"""
            
            # Call the OpenAI API for report generation
            print(f"[SynthesisAgent] Calling {self._model} for report generation...")
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            report = response.choices[0].message.content.strip()
            
            # Generate a report name
            report_name = f"Research_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            return {
                "report": report,
                "report_name": report_name,
                "question": question,
                "source_count": len(sources),
                "model_used": self._model
            }
        except Exception as e:
            print(f"[SynthesisAgent] Error during report generation: {str(e)}")
            return {"error": f"Report generation failed: {str(e)}"}


class ResearchPrincipalAgent(IPrincipalAgent):
    """Principal agent that orchestrates research tasks across specialized agents."""
    
    def __init__(self, model: str = None):
        self._id = f"research-principal-agent-{str(uuid.uuid4())[:8]}"
        self._name = "Research Principal Agent"
        self._description = "Orchestrates research tasks across specialized agents"
        self._capabilities = [
            {
                "id": "research-orchestration",
                "type": "research_orchestration",
                "name": "Research Orchestration",
                "description": "Orchestrates research tasks across multiple agents"
            },
            {
                "id": "research-planning",
                "type": "research_planning",
                "name": "Research Planning",
                "description": "Plans research strategies based on questions"
            }
        ]
        
        # Store specialized agents
        self._github_agent = GitHubResearchAgent()
        self._arxiv_agent = ArXivResearchAgent()
        self._web_agent = WebSearchAgent()
        self._synthesis_agent = SynthesisAgent(model=model)
        
        # Configuration
        self._model = model or AgentConfig.DEFAULT_MODEL
        print(f"[ResearchPrincipalAgent] Using model: {self._model}")
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return self._capabilities
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if capability_id == "research-orchestration":
            return self._orchestrate_research(inputs, config)
        elif capability_id == "research-planning":
            return self._plan_research(inputs, config)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
    
    def health_check(self) -> bool:
        try:
            # Check if the API is working
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Are you operational? Reply with 'yes' or 'no'."}],
                max_tokens=5
            )
            content = response.choices[0].message.content.strip().lower()
            return "yes" in content
        except Exception as e:
            print(f"[ResearchPrincipalAgent] Health check failed: {str(e)}")
            return False
    
    def _plan_research(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan a research strategy based on a question."""
        question = inputs.get("question", "")
        if not question:
            return {"error": "No question provided for research planning"}
        
        print(f"[ResearchPrincipalAgent] Planning research for question: {question}")
        
        try:
            # Create a prompt for planning
            prompt = f"""You are a research planning assistant tasked with creating a strategy to answer the following research question:

Question: {question}

Your task is to:
1. Analyze the key components of the question
2. Identify the main topics that need to be researched
3. Determine specific search queries for:
   - GitHub (code/repositories)
   - arXiv (academic papers)
   - Web search (general information)
4. Outline how the information should be synthesized

Format your response as follows:
- Key Aspects: [List the key aspects of the question]
- GitHub Search Queries: [List 2-3 specific search queries for GitHub]
- arXiv Search Queries: [List 2-3 specific search queries for arXiv]
- Web Search Queries: [List 2-3 specific search queries for web search]
- Synthesis Approach: [Describe how to combine and analyze the information]
"""
            
            # Call the OpenAI API for planning
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Parse the plan
            plan = self._parse_research_plan(plan_text)
            
            return {
                "question": question,
                "plan": plan,
                "raw_plan": plan_text
            }
        except Exception as e:
            print(f"[ResearchPrincipalAgent] Planning error: {str(e)}")
            return {"error": f"Research planning failed: {str(e)}"}
    
    def _parse_research_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parse the research plan from text to structured format."""
        plan = {
            "key_aspects": [],
            "github_queries": [],
            "arxiv_queries": [],
            "web_queries": [],
            "synthesis_approach": ""
        }
        
        # Simple parsing - in production this would be more robust
        sections = re.split(r'\n\s*-\s*|\n#\s*|\n##\s*', plan_text)
        current_section = None
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if "Key Aspects:" in section:
                current_section = "key_aspects"
                section = section.replace("Key Aspects:", "").strip()
            elif "GitHub Search Queries:" in section:
                current_section = "github_queries"
                section = section.replace("GitHub Search Queries:", "").strip()
            elif "arXiv Search Queries:" in section:
                current_section = "arxiv_queries"
                section = section.replace("arXiv Search Queries:", "").strip()
            elif "Web Search Queries:" in section:
                current_section = "web_queries"
                section = section.replace("Web Search Queries:", "").strip()
            elif "Synthesis Approach:" in section:
                current_section = "synthesis_approach"
                section = section.replace("Synthesis Approach:", "").strip()
            
            if current_section == "key_aspects":
                aspects = [item.strip() for item in re.split(r'\n\s*\d+\.\s*|\n\s*-\s*', section) if item.strip()]
                plan["key_aspects"].extend(aspects)
            elif current_section == "github_queries":
                queries = [item.strip() for item in re.split(r'\n\s*\d+\.\s*|\n\s*-\s*', section) if item.strip()]
                plan["github_queries"].extend(queries)
            elif current_section == "arxiv_queries":
                queries = [item.strip() for item in re.split(r'\n\s*\d+\.\s*|\n\s*-\s*', section) if item.strip()]
                plan["arxiv_queries"].extend(queries)
            elif current_section == "synthesis_approach":
                plan["synthesis_approach"] += section + " "
        
        # Clean up synthesis approach
        plan["synthesis_approach"] = plan["synthesis_approach"].strip()
        
        return plan
    
    def _orchestrate_research(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Orchestrate the research process across specialized agents."""
        question = inputs.get("question", "")
        if not question:
            return {"error": "No question provided for research"}
        
        print(f"[ResearchPrincipalAgent] Orchestrating research for question: {question}")
        
        # Step 1: Create a research plan
        print("\n=== Step 1: Research Planning ===")
        plan_result = self._plan_research({"question": question})
        
        if "error" in plan_result:
            return plan_result
        
        plan = plan_result.get("plan", {})
        print(f"Research plan created:")
        print(f"- Key Aspects: {', '.join(plan.get('key_aspects', []))}")
        print(f"- GitHub Queries: {', '.join(plan.get('github_queries', []))}")
        print(f"- arXiv Queries: {', '.join(plan.get('arxiv_queries', []))}")
        print(f"- Web Queries: {', '.join(plan.get('web_queries', []))}")
        
        # Step 2: Collect GitHub information
        print("\n=== Step 2: GitHub Research ===")
        github_results = []
        for query in plan.get("github_queries", []):
            print(f"Searching GitHub for: {query}")
            search_result = self._github_agent.invoke("github-search", {"query": query})
            
            if "error" not in search_result and "results" in search_result:
                # For the top result, get more detailed information
                if search_result["results"]:
                    top_result = search_result["results"][0]
                    print(f"Analyzing top repository: {top_result.get('name', 'Unknown')}")
                    
                    analysis_result = self._github_agent.invoke(
                        "github-repo-analysis", 
                        {"repo_url": top_result.get("url", "")}
                    )
                    
                    if "error" not in analysis_result:
                        analysis_result["type"] = "github"
                        github_results.append(analysis_result)
        
        # Step 3: Collect arXiv information
        print("\n=== Step 3: arXiv Research ===")
        arxiv_results = []
        for query in plan.get("arxiv_queries", []):
            print(f"Searching arXiv for: {query}")
            search_result = self._arxiv_agent.invoke("arxiv-search", {"query": query, "max_results": 3})
            
            if "error" not in search_result and "results" in search_result:
                # For each relevant paper, get a summary
                for paper in search_result["results"][:1]:  # Limit to top paper for demo
                    paper_id = paper.get("id", "")
                    if paper_id:
                        print(f"Summarizing paper: {paper.get('title', 'Unknown')} ({paper_id})")
                        
                        summary_result = self._arxiv_agent.invoke(
                            "arxiv-paper-summary", 
                            {"paper_id": paper_id}
                        )
                        
                        if "error" not in summary_result:
                            summary_result["type"] = "arxiv"
                            arxiv_results.append(summary_result)
        
        # Step 4: Collect web search information
        print("\n=== Step 4: Web Research ===")
        web_results = []
        for query in plan.get("web_queries", []):
            print(f"Searching the web for: {query}")
            search_result = self._web_agent.invoke("web-search", {"query": query, "max_results": 5})
            
            if "error" not in search_result and "results" in search_result:
                # Process and add the web results
                processed_result = {
                    "type": "web",
                    "query": query,
                    "search_results": search_result["results"][:3]  # Limit to top 3 results
                }
                web_results.append(processed_result)
        
        # Step 5: Synthesize the information
        print("\n=== Step 5: Information Synthesis ===")
        all_sources = github_results + arxiv_results + web_results
        
        if not all_sources:
            return {"error": "No useful sources found for the research question"}
        
        synthesis_result = self._synthesis_agent.invoke(
            "information-synthesis",
            {"sources": all_sources, "question": question}
        )
        
        if "error" in synthesis_result:
            return synthesis_result
        
        # Step 6: Generate a report
        print("\n=== Step 6: Report Generation ===")
        report_result = self._synthesis_agent.invoke(
            "research-report-generation",
            {
                "synthesis": synthesis_result.get("synthesis", ""),
                "question": question,
                "sources": all_sources
            }
        )
        
        # Compile final result
        result = {
            "question": question,
            "plan": plan,
            "github_sources": github_results,
            "arxiv_sources": arxiv_results,
            "web_sources": web_results,
            "synthesis": synthesis_result.get("synthesis", ""),
            "report": report_result.get("report", ""),
            "report_name": report_result.get("report_name", "")
        }
        
        return result
    
    # IPrincipalAgent interface methods
    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down a research task into subtasks."""
        question = task.get("description", "")
        if not question:
            return []
        
        # Plan the research
        plan_result = self._plan_research({"question": question})
        plan = plan_result.get("plan", {})
        
        # Create subtasks
        subtasks = []
        
        # Add GitHub research subtasks
        for i, query in enumerate(plan.get("github_queries", []), 1):
            subtasks.append({
                "id": f"github-search-{i}",
                "description": f"Search GitHub for '{query}'",
                "inputs": {"query": query},
                "required_capabilities": [{"type": "github_search"}],
                "dependencies": []
            })
            
            # Add repository analysis task (depends on search)
            subtasks.append({
                "id": f"github-analysis-{i}",
                "description": f"Analyze top repository from GitHub search '{query}'",
                "inputs": {"repo_url": "@{github-search-" + str(i) + ".results[0].url}"},
                "required_capabilities": [{"type": "github_repo_analysis"}],
                "dependencies": [f"github-search-{i}"]
            })
        
        # Add arXiv research subtasks
        for i, query in enumerate(plan.get("arxiv_queries", []), 1):
            subtasks.append({
                "id": f"arxiv-search-{i}",
                "description": f"Search arXiv for '{query}'",
                "inputs": {"query": query, "max_results": 3},
                "required_capabilities": [{"type": "arxiv_search"}],
                "dependencies": []
            })
            
            # Add paper summary task (depends on search)
            subtasks.append({
                "id": f"arxiv-summary-{i}",
                "description": f"Summarize top paper from arXiv search '{query}'",
                "inputs": {"paper_id": "@{arxiv-search-" + str(i) + ".results[0].id}"},
                "required_capabilities": [{"type": "arxiv_paper_summary"}],
                "dependencies": [f"arxiv-search-{i}"]
            })
            
        # Add web search subtasks
        for i, query in enumerate(plan.get("web_queries", []), 1):
            subtasks.append({
                "id": f"web-search-{i}",
                "description": f"Search the web for '{query}'",
                "inputs": {"query": query, "max_results": 5},
                "required_capabilities": [{"type": "web_search"}],
                "dependencies": []
            })
        
        # Add synthesis subtask (depends on all research)
        synthesis_dependencies = [
            f"github-analysis-{i}" for i in range(1, len(plan.get("github_queries", [])) + 1)
        ] + [
            f"arxiv-summary-{i}" for i in range(1, len(plan.get("arxiv_queries", [])) + 1)
        ] + [
            f"web-search-{i}" for i in range(1, len(plan.get("web_queries", [])) + 1)
        ]
        
        # Construct sources array for synthesis
        sources_expr = "["
        for dep in synthesis_dependencies:
            sources_expr += f"@{{{dep}}}, "
        if sources_expr.endswith(", "):
            sources_expr = sources_expr[:-2]
        sources_expr += "]"
        
        subtasks.append({
            "id": "synthesis",
            "description": "Synthesize research findings",
            "inputs": {
                "sources": sources_expr,
                "question": question
            },
            "required_capabilities": [{"type": "information_synthesis"}],
            "dependencies": synthesis_dependencies
        })
        
        # Add report generation subtask (depends on synthesis)
        subtasks.append({
            "id": "report",
            "description": "Generate research report",
            "inputs": {
                "synthesis": "@{synthesis.synthesis}",
                "question": question,
                "sources": sources_expr
            },
            "required_capabilities": [{"type": "research_report_generation"}],
            "dependencies": ["synthesis"]
        })
        
        return subtasks
    
    def discover_agents(self, capability_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find agents with capabilities matching requirements."""
        result = {}
        
        for requirement in capability_requirements:
            capability_type = requirement.get("type")
            if not capability_type:
                continue
            
            if capability_type == "github_search" or capability_type == "github_repo_analysis":
                result[capability_type] = [self._github_agent.get_info()["id"]]
            elif capability_type == "arxiv_search" or capability_type == "arxiv_paper_summary":
                result[capability_type] = [self._arxiv_agent.get_info()["id"]]
            elif capability_type == "information_synthesis" or capability_type == "research_report_generation":
                result[capability_type] = [self._synthesis_agent.get_info()["id"]]
        
        return result
    
    def create_execution_plan(self, subtasks: List[Dict[str, Any]], available_agents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Create a plan for executing subtasks with selected agents."""
        execution_plan = []
        
        for subtask in subtasks:
            # Find an agent for each required capability
            agent_assignments = {}
            for capability in subtask.get("required_capabilities", []):
                capability_type = capability.get("type")
                if capability_type in available_agents and available_agents[capability_type]:
                    agent_id = available_agents[capability_type][0]
                    agent_assignments[capability_type] = agent_id
            
            execution_step = {
                "task_id": subtask["id"],
                "description": subtask["description"],
                "agent_assignments": agent_assignments,
                "inputs": subtask["inputs"],
                "dependencies": subtask.get("dependencies", [])
            }
            
            execution_plan.append(execution_step)
        
        return execution_plan
    
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the research plan and return aggregated results."""
        # Implementation would follow the _orchestrate_research method
        # For simplicity, we'll return a dummy response
        return {"message": "Plan execution not fully implemented in this demo"}
    
    def handle_error(self, error: Dict[str, Any], execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle errors during plan execution."""
        # Simple error handling for demo purposes
        return {"error_handled": False, "original_error": error}


def run_demo():
    """Run the research agent demo."""
    print("=== Research Agent Demo ===\n")
    
    # Check the OpenAI API configuration
    model_to_use = AgentConfig.DEFAULT_MODEL
    
    print(f"Using model: {model_to_use}")
    
    # Create the research principal agent
    principal_agent = ResearchPrincipalAgent(model=model_to_use)
    
    # Check agent health (temporarily disabled due to rate limiting)
    # health_status = principal_agent.health_check()
    # print(f"\nAgent health status: {'✅ Healthy' if health_status else '❌ Unhealthy'}")
    
    # if not health_status:
    #     print("Agent health check failed. Exiting.")
    #     return
    health_status = True  # Assume healthy for now
    
    # Define a research question
    questions = [
        "What are the latest advancements in agent interoperability frameworks?",
        "How are LLMs being used to coordinate multi-agent systems?",
        "What are the key challenges in implementing distributed agent architectures?",
        "How do DAWN and AGNTCY specifications compare to other agent frameworks?"
    ]
    
    print("\nAvailable research questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    # Let user select a question or enter their own
    try:
        choice = input("\nSelect a question number or type your own research question: ")
        if choice.isdigit() and 1 <= int(choice) <= len(questions):
            question = questions[int(choice) - 1]
        else:
            question = choice
    except Exception:
        # Default to the first question if input fails
        question = questions[0]
    
    print(f"\nResearching: {question}")
    
    # Orchestrate the research process
    try:
        result = principal_agent.invoke("research-orchestration", {"question": question})
        
        if "error" in result:
            print(f"\nResearch error: {result['error']}")
            return
        
        # Print the synthesis
        print("\n=== Research Synthesis ===\n")
        print(result.get("synthesis", "No synthesis available"))
        
        # Save the report to a file
        report = result.get("report", "")
        report_name = result.get("report_name", f"Research_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_name, "w") as f:
            f.write(report)
        
        print(f"\nFull research report saved to: {report_name}")
        
    except Exception as e:
        print(f"\nAn error occurred during the research process: {str(e)}")


if __name__ == "__main__":
    import sys
    
    run_demo()