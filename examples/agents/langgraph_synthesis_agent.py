#!/usr/bin/env python3
"""
LangGraph Synthesis Agent

A synthesis agent powered by LangGraph that:
- Combines research from GitHub and arXiv agents
- Provides comprehensive analysis and insights
- Maintains full conversation context
- Multi-protocol support (A2A, ACP, MCP)
"""

import os
import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from .base_langgraph_agent import MultiProtocolLangGraphAgent

logger = logging.getLogger(__name__)


class SynthesizeTool(BaseTool):
    """Tool for synthesizing information from multiple sources."""
    
    name: str = "synthesize_research"
    description: str = "Synthesize and analyze information from multiple research sources (GitHub, arXiv, etc.)"
    llm: Any = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, llm) -> None:
        super().__init__(llm=llm)
    
    def _run(self, sources: List[Dict[str, Any]], question: str) -> str:
        """Synthesize information from multiple sources."""
        try:
            # Build context from sources
            context = "Research Sources:\n\n"
            
            for i, source in enumerate(sources, 1):
                source_type = source.get("type", "unknown")
                
                if source_type == "github":
                    context += f"Source {i} (GitHub Repository):\n"
                    if "analysis" in source:
                        analysis = source["analysis"]
                        context += f"- Repository: {analysis.get('name', 'Unknown')}\n"
                        context += f"- Description: {analysis.get('description', 'N/A')}\n"
                        context += f"- Stars: {analysis.get('stars', 0)}\n"
                        context += f"- Languages: {', '.join(analysis.get('languages', []))}\n"
                        context += f"- Recent activity: {analysis.get('updated_at', 'Unknown')}\n"
                    elif "results" in source:
                        context += f"- Search results: {len(source['results'])} repositories found\n"
                        for repo in source["results"][:3]:
                            context += f"  • {repo['name']} ({repo['stars']} stars)\n"
                
                elif source_type == "arxiv":
                    context += f"Source {i} (arXiv Paper):\n"
                    if "paper" in source:
                        paper = source["paper"]
                        context += f"- Title: {paper.get('title', 'Unknown')}\n"
                        context += f"- Authors: {', '.join(paper.get('authors', []))}\n"
                        context += f"- Published: {paper.get('published', 'Unknown')}\n"
                        context += f"- Summary: {paper.get('summary', 'N/A')[:300]}...\n"
                    elif "results" in source:
                        context += f"- Search results: {len(source['results'])} papers found\n"
                        for paper in source["results"][:3]:
                            context += f"  • {paper['title']} ({paper['published']})\n"
                
                else:
                    context += f"Source {i} ({source_type}):\n"
                    context += json.dumps(source, indent=2)[:500] + "...\n"
                
                context += "\n"
            
            # Create synthesis prompt
            prompt = f"""{context}

Research Question: {question}

Please provide a comprehensive synthesis that:
1. Identifies key themes and patterns across sources
2. Highlights important findings and insights
3. Notes any contradictions or gaps
4. Provides actionable recommendations
5. Suggests areas for further research

Format the response in a clear, structured manner."""

            # Get LLM synthesis
            response = self.llm.invoke(prompt)
            synthesis = response.content if hasattr(response, 'content') else str(response)
            
            return synthesis
            
        except Exception as e:
            return f"Synthesis failed: {str(e)}"
    
    async def _arun(self, sources: List[Dict[str, Any]], question: str) -> str:
        """Async version of synthesis."""
        return self._run(sources, question)


class CompareTool(BaseTool):
    """Tool for comparing different technologies or approaches."""
    
    name: str = "compare_technologies"
    description: str = "Compare and contrast different technologies, frameworks, or research approaches"
    llm: Any = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, llm) -> None:
        super().__init__(llm=llm)
    
    def _run(self, items: List[str], criteria: Optional[List[str]] = None) -> str:
        """Compare multiple items based on criteria."""
        try:
            if not criteria:
                criteria = ["features", "performance", "ease of use", "community support", "documentation"]
            
            prompt = f"""Please provide a detailed comparison of the following items:

Items to compare: {', '.join(items)}

Evaluation criteria:
{chr(10).join(f'- {criterion}' for criterion in criteria)}

Please structure your comparison as:
1. Overview of each item
2. Detailed comparison table
3. Strengths and weaknesses of each
4. Recommendations for different use cases
5. Overall conclusion

Be objective and cite specific examples where possible."""

            response = self.llm.invoke(prompt)
            comparison = response.content if hasattr(response, 'content') else str(response)
            
            return comparison
            
        except Exception as e:
            return f"Comparison failed: {str(e)}"
    
    async def _arun(self, items: List[str], criteria: Optional[List[str]] = None) -> str:
        """Async version of compare."""
        return self._run(items, criteria)


class TrendAnalysisTool(BaseTool):
    """Tool for analyzing research trends and patterns."""
    
    name: str = "analyze_trends"
    description: str = "Analyze trends and patterns in research data from multiple sources"
    llm: Any = Field(exclude=True)  # Exclude from serialization
    
    def __init__(self, llm) -> None:
        super().__init__(llm=llm)
    
    def _run(self, data: Dict[str, Any], time_period: Optional[str] = None) -> str:
        """Analyze trends in research data."""
        try:
            prompt = f"""Analyze the following research data for trends and patterns:

Data: {json.dumps(data, indent=2)[:2000]}...

Time period: {time_period or 'All available data'}

Please provide:
1. Key trends identified
2. Emerging topics or technologies
3. Declining areas of interest
4. Geographic or institutional patterns
5. Predictions for future developments
6. Recommendations for researchers or developers

Use specific examples from the data to support your analysis."""

            response = self.llm.invoke(prompt)
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return analysis
            
        except Exception as e:
            return f"Trend analysis failed: {str(e)}"
    
    async def _arun(self, data: Dict[str, Any], time_period: Optional[str] = None) -> str:
        """Async version of trend analysis."""
        return self._run(data, time_period)


class LangGraphSynthesisAgent(MultiProtocolLangGraphAgent):
    """
    Synthesis agent powered by LangGraph that combines and analyzes research.
    
    This agent provides:
    - Multi-source synthesis
    - Technology comparison
    - Trend analysis
    - Full context awareness
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Synthesis Agent",
        port: int = 8083
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="LangGraph-powered synthesis agent for combining and analyzing research",
            version="1.0.0",
            port=port
        )
        
        # Store conversation context
        self.conversation_context = []
        
        logger.info(f"Initialized {name} with synthesis capabilities")
    
    def get_tools(self) -> List[BaseTool]:
        """Return synthesis-specific tools."""
        return [
            SynthesizeTool(self.llm),
            CompareTool(self.llm),
            TrendAnalysisTool(self.llm)
        ]
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return OASF capability definitions."""
        return [
            {
                "id": "synthesize_research",
                "type": "synthesis.research.combine",
                "name": "Research Synthesis",
                "description": "Synthesize information from multiple research sources",
                "parameters": {
                    "sources": {
                        "type": "array",
                        "description": "List of research sources to synthesize",
                        "items": {"type": "object"},
                        "required": True
                    },
                    "question": {
                        "type": "string",
                        "description": "Research question or synthesis focus",
                        "required": True
                    }
                },
                "metadata": {
                    "examples": ["synthesize GitHub and arXiv findings on transformers"],
                    "tags": ["synthesis", "analysis", "research"]
                }
            },
            {
                "id": "compare_technologies",
                "type": "synthesis.comparison",
                "name": "Technology Comparison",
                "description": "Compare and contrast different technologies or approaches",
                "parameters": {
                    "items": {
                        "type": "array",
                        "description": "Items to compare",
                        "items": {"type": "string"},
                        "required": True
                    },
                    "criteria": {
                        "type": "array",
                        "description": "Comparison criteria",
                        "items": {"type": "string"},
                        "default": ["features", "performance", "ease of use"]
                    }
                },
                "metadata": {
                    "examples": ["compare PyTorch vs TensorFlow"],
                    "tags": ["comparison", "analysis", "evaluation"]
                }
            },
            {
                "id": "analyze_trends",
                "type": "synthesis.trends",
                "name": "Trend Analysis",
                "description": "Analyze trends and patterns in research data",
                "parameters": {
                    "data": {
                        "type": "object",
                        "description": "Research data to analyze",
                        "required": True
                    },
                    "time_period": {
                        "type": "string",
                        "description": "Time period for analysis",
                        "default": "all"
                    }
                },
                "metadata": {
                    "examples": ["analyze ML framework trends over past year"],
                    "tags": ["trends", "analysis", "patterns"]
                }
            }
        ]
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Override to maintain conversation context."""
        # Add to conversation history
        self.conversation_context.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "role": "user"
        })
        
        # Include conversation context
        if context is None:
            context = {}
        context["conversation_history"] = self.conversation_context[-10:]  # Last 10 messages
        
        # Process with base class
        result = await super().process_message(message, context)
        
        # Add response to history
        if result.get("success"):
            self.conversation_context.append({
                "timestamp": datetime.now().isoformat(),
                "message": result["response"],
                "role": "assistant",
                "tools_used": result.get("tools_used", [])
            })
        
        return result
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """
        Natural language chat interface for the synthesis agent.
        
        This method provides a user-friendly interface that:
        - Maintains full conversation context
        - Uses LangGraph ReACT for reasoning
        - Returns comprehensive synthesis
        """
        logger.info(f"Synthesis agent received message: {message}")
        
        # Process with LangGraph
        result = await self.process_message(message)
        
        if result.get("success"):
            # Format the response nicely
            response = f"🧪 **Synthesis Results**\n\n{result['response']}"
            
            # Add tool usage information
            if result.get("tools_used"):
                response += f"\n\n📊 **Analysis tools used:** {', '.join(result['tools_used'])}"
            
            # Add reasoning trace if available
            if result.get("reasoning_trace") and len(result["reasoning_trace"]) > 1:
                response += "\n\n🧠 **Reasoning process:**"
                for step in result["reasoning_trace"]:
                    response += f"\n• {step}"
            
            # Add context awareness note
            if len(self.conversation_context) > 1:
                response += f"\n\n💭 **Context:** Drawing from {len(self.conversation_context)} previous exchanges"
            
            return {
                "response": response,
                "type": "synthesis",
                "data": result,
                "success": True,
                "context_length": len(self.conversation_context)
            }
        else:
            return {
                "response": f"❌ Synthesis failed: {result.get('error', 'Unknown error')}",
                "type": "error",
                "success": False
            }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = LangGraphSynthesisAgent()
        
        # Test queries
        queries = [
            "Compare PyTorch and TensorFlow for deep learning research",
            "Synthesize recent trends in transformer architectures",
            "What are the key patterns you see in modern ML frameworks?"
        ]
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            result = await agent.chat(query)
            print(result["response"])
    
    asyncio.run(test_agent()) 