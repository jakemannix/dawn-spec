#!/usr/bin/env python3
"""
LangGraph arXiv Research Agent

An arXiv research agent powered by LangGraph that provides:
- Paper search with semantic understanding
- PDF metadata and abstract fetching
- Paper summarization using LLM
- Multi-protocol support (A2A, ACP, MCP)
"""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import arxiv

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from .base_langgraph_agent import MultiProtocolLangGraphAgent

logger = logging.getLogger(__name__)


class ArXivSearchTool(BaseTool):
    """Tool for searching arXiv papers."""
    
    name: str = "arxiv_search"
    description: str = "Search arXiv for academic papers by query. Returns papers with metadata."
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute arXiv search."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                results.append({
                    "id": paper.entry_id.split('/')[-1],
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d"),
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "abs_url": paper.entry_id
                })
            
            return f"Found {len(results)} papers: {results}"
            
        except Exception as e:
            return f"arXiv search failed: {str(e)}"
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Async version of search."""
        return self._run(query, max_results)


class ArXivFetchPaperTool(BaseTool):
    """Tool for fetching detailed paper information from arXiv."""
    
    name: str = "arxiv_fetch_paper"
    description: str = "Fetch detailed information about a specific arXiv paper by ID (e.g., 1706.03762)."
    
    def _run(self, paper_id: str) -> str:
        """Fetch paper details from arXiv."""
        try:
            # Clean the paper ID
            if paper_id.startswith("arXiv:"):
                paper_id = paper_id[6:]
            
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            
            details = {
                "id": paper.entry_id.split('/')[-1],
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d"),
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "comment": paper.comment,
                "journal_ref": paper.journal_ref,
                "doi": paper.doi,
                "pdf_url": paper.pdf_url,
                "abs_url": paper.entry_id
            }
            
            return f"Paper details for {paper_id}: {details}"
            
        except StopIteration:
            return f"Paper not found: {paper_id}"
        except Exception as e:
            return f"Failed to fetch paper: {str(e)}"
    
    async def _arun(self, paper_id: str) -> str:
        """Async version of fetch."""
        return self._run(paper_id)


class ArXivSummarizeTool(BaseTool):
    """Tool for summarizing arXiv papers using LLM."""
    
    name: str = "arxiv_summarize"
    description: str = "Summarize an arXiv paper's key contributions and findings. Requires paper ID."
    llm: Any = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, llm) -> None:
        super().__init__(llm=llm)
    
    def _run(self, paper_id: str) -> str:
        """Summarize an arXiv paper."""
        try:
            # Clean the paper ID
            if paper_id.startswith("arXiv:"):
                paper_id = paper_id[6:]
            
            # Fetch paper details
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            
            # Create prompt for summarization
            prompt = f"""Please provide a comprehensive summary of this arXiv paper:

Title: {paper.title}
Authors: {', '.join([author.name for author in paper.authors])}
Published: {paper.published.strftime("%Y-%m-%d")}
Categories: {', '.join(paper.categories)}

Abstract:
{paper.summary}

Please include:
1. Main research question/problem
2. Key methodology or approach
3. Major findings/contributions
4. Potential impact or applications
5. Limitations or future work mentioned

Keep the summary concise but informative."""

            # Get LLM summary
            response = self.llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            return f"Summary of arXiv:{paper_id} - {paper.title}:\n\n{summary}"
            
        except StopIteration:
            return f"Paper not found: {paper_id}"
        except Exception as e:
            return f"Summarization failed: {str(e)}"
    
    async def _arun(self, paper_id: str) -> str:
        """Async version of summarize."""
        return self._run(paper_id)


class LangGraphArXivAgent(MultiProtocolLangGraphAgent):
    """
    arXiv research agent powered by LangGraph with real API integration.
    
    This agent provides:
    - Semantic paper search
    - Paper metadata retrieval
    - LLM-powered paper summarization
    - Multi-protocol support
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "arXiv Research Agent",
        port: int = 8082
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="LangGraph-powered arXiv research agent with real API integration",
            version="1.0.0",
            port=port
        )
        
        logger.info(f"Initialized {name} with arXiv API access")
    
    def get_tools(self) -> List[BaseTool]:
        """Return arXiv-specific tools."""
        return [
            ArXivSearchTool(),
            ArXivFetchPaperTool(),
            ArXivSummarizeTool(self.llm)  # Pass the LLM for summarization
        ]
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return OASF capability definitions."""
        return [
            {
                "id": "arxiv_search",
                "type": "research.arxiv.search",
                "name": "arXiv Paper Search",
                "description": "Search arXiv for academic papers with semantic understanding",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query for papers",
                        "required": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "metadata": {
                    "examples": ["search for transformer attention mechanisms", "find papers on neural networks"],
                    "tags": ["arxiv", "search", "research", "papers"]
                }
            },
            {
                "id": "arxiv_fetch_paper",
                "type": "research.arxiv.fetch",
                "name": "arXiv Paper Fetcher",
                "description": "Fetch detailed information about a specific arXiv paper",
                "parameters": {
                    "paper_id": {
                        "type": "string",
                        "description": "arXiv paper ID (e.g., 1706.03762)",
                        "required": True
                    }
                },
                "metadata": {
                    "examples": ["fetch 1706.03762", "get details for 2010.11929"],
                    "tags": ["arxiv", "fetch", "paper", "metadata"]
                }
            },
            {
                "id": "arxiv_summarize",
                "type": "research.arxiv.summarize",
                "name": "arXiv Paper Summarizer",
                "description": "Summarize an arXiv paper's key contributions using LLM",
                "parameters": {
                    "paper_id": {
                        "type": "string",
                        "description": "arXiv paper ID to summarize",
                        "required": True
                    }
                },
                "metadata": {
                    "examples": ["summarize 1706.03762", "explain paper 2010.11929"],
                    "tags": ["arxiv", "summarize", "llm", "analysis"]
                }
            }
        ]
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """
        Natural language chat interface for the arXiv agent.
        
        This method provides a user-friendly interface that:
        - Understands natural language queries about papers
        - Uses LangGraph ReACT for reasoning
        - Returns formatted responses
        """
        logger.info(f"arXiv agent received message: {message}")
        
        # Process with LangGraph
        result = await self.process_message(message)
        
        if result.get("success"):
            # Format the response nicely
            response = f"📚 **arXiv Research Results**\n\n{result['response']}"
            
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
                "type": "arxiv_research",
                "data": result,
                "success": True
            }
        else:
            return {
                "response": f"❌ arXiv research failed: {result.get('error', 'Unknown error')}",
                "type": "error",
                "success": False
            }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = LangGraphArXivAgent()
        
        # Test queries
        queries = [
            "Search for recent papers on transformer attention mechanisms",
            "Fetch details about paper 1706.03762",
            "Summarize the Attention Is All You Need paper (1706.03762)"
        ]
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            result = await agent.chat(query)
            print(result["response"])
    
    asyncio.run(test_agent()) 