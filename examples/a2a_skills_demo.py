#!/usr/bin/env python3
"""
A2A Skills Demo - Properly implementing the three-layer model.

This demonstrates:
1. OASF Capabilities: Schema definitions and semantic tags
2. A2A Skills: Concrete functions exposed over the wire  
3. MCP Tools: LLM-facing interfaces (simulated via chat methods)

Architecture:
- Each agent has OASF capabilities (schema/metadata)
- Each agent exposes A2A skills that implement those capabilities
- Each agent has a chat interface that LLMs would use (simulating MCP Tools)
- Agents can call each other's A2A skills over the network
"""

import asyncio
import logging
import sys
import argparse
from typing import Dict, Any, List, Optional
import json

# Add src to path
sys.path.insert(0, 'src')

from src.a2a_integration_v2 import DawnA2ASkillAdapter, A2ASkill, A2AMessage, A2ATask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockGitHubAgent:
    """Mock GitHub research agent with proper OASF capabilities and A2A skills."""
    
    def __init__(self, agent_id: str = "github_agent"):
        self.id = agent_id
        self.name = "GitHub Research Agent"
        self.description = "Specializes in GitHub repository research and analysis"
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent information."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": "1.0.0"
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return OASF capabilities (schema definitions)."""
        return [
            {
                "id": "github_search",
                "type": "research.github.search",
                "name": "GitHub Repository Search",
                "description": "Search GitHub repositories by keywords and criteria",
                "parameters": {
                    "query": {"type": "string", "required": True, "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5, "description": "Maximum results to return"}
                },
                "metadata": {
                    "examples": ["Search for machine learning repositories", "Find Python web frameworks"],
                    "tags": ["github", "search", "research"]
                }
            },
            {
                "id": "github_analyze",
                "type": "research.github.analysis",
                "name": "GitHub Repository Analysis", 
                "description": "Analyze a specific GitHub repository in detail",
                "parameters": {
                    "repository_url": {"type": "string", "required": True, "description": "GitHub repository URL"}
                },
                "metadata": {
                    "examples": ["Analyze pytorch/pytorch repository"],
                    "tags": ["github", "analysis", "research"]
                }
            }
        ]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a capability (implements the OASF capability contract)."""
        if capability_id == "github_search":
            return self._handle_github_search(inputs)
        elif capability_id == "github_analyze":
            return self._handle_github_analyze(inputs)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
            
    def _handle_github_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub search capability."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        # Mock search results
        mock_results = [
            {"name": "pytorch/pytorch", "description": "Tensors and Dynamic neural networks", "stars": 45000, "language": "Python"},
            {"name": "tensorflow/tensorflow", "description": "An Open Source Machine Learning Framework", "stars": 85000, "language": "C++"},
            {"name": "scikit-learn/scikit-learn", "description": "Machine learning in Python", "stars": 52000, "language": "Python"}
        ]
        
        return {
            "success": True,
            "results": mock_results[:max_results],
            "query": query,
            "capability_metadata": {
                "capability_id": "github_search",
                "capability_type": "research.github.search"
            }
        }
        
    def _handle_github_analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub repository analysis."""
        repo_url = inputs.get("repository_url", "")
        
        # Mock analysis
        return {
            "success": True,
            "analysis": {
                "repo_url": repo_url,
                "name": "pytorch/pytorch",
                "description": "Deep learning framework",
                "stars": 45000,
                "forks": 12000,
                "language": "Python",
                "topics": ["machine-learning", "deep-learning", "pytorch"],
                "readme_summary": "PyTorch is a Python package that provides Tensor computation with strong GPU acceleration"
            },
            "capability_metadata": {
                "capability_id": "github_analyze", 
                "capability_type": "research.github.analysis"
            }
        }
        
    async def chat(self, message: str) -> Dict[str, Any]:
        """Chat interface (simulates what an MCP Tool would expose to LLMs)."""
        logger.info(f"GitHub agent chat: {message}")
        
        # Simple intent detection
        message_lower = message.lower()
        
        if "search" in message_lower or "find" in message_lower:
            # Extract search query (simple approach)
            query = message.replace("search for", "").replace("find", "").strip()
            if not query:
                query = "machine learning"
                
            result = self.invoke("github_search", {"query": query, "max_results": 3})
            
            if result.get("success"):
                response = f"🔍 **Found {len(result['results'])} repositories:**\n\n"
                for repo in result["results"][:3]:
                    response += f"• **{repo['name']}** ({repo['language']}) - ⭐ {repo['stars']}\n"
                    response += f"  {repo['description']}\n\n"
                return {"response": response, "data": result}
            else:
                return {"response": f"❌ Search failed: {result.get('error', 'Unknown error')}", "data": result}
                
        elif "analyz" in message_lower or "repository" in message_lower:
            # Default to analyzing PyTorch
            repo_url = "https://github.com/pytorch/pytorch"
            if "github.com" in message:
                # Extract URL if provided
                words = message.split()
                for word in words:
                    if "github.com" in word:
                        repo_url = word
                        break
                        
            result = self.invoke("github_analyze", {"repository_url": repo_url})
            
            if result.get("success"):
                analysis = result["analysis"]
                response = f"🔍 **Repository Analysis:**\n\n"
                response += f"**Name:** {analysis['name']}\n"
                response += f"**Description:** {analysis['description']}\n"
                response += f"**Stars:** ⭐ {analysis['stars']} | **Forks:** 🍴 {analysis['forks']}\n"
                response += f"**Language:** {analysis['language']}\n"
                response += f"**Topics:** {', '.join(analysis['topics'])}\n\n"
                response += f"**README Summary:** {analysis['readme_summary']}\n"
                return {"response": response, "data": result}
            else:
                return {"response": f"❌ Analysis failed: {result.get('error', 'Unknown error')}", "data": result}
        else:
            return {
                "response": "🤖 I can help you **search** GitHub repositories or **analyze** specific repositories. Try:\n• 'search for python web frameworks'\n• 'analyze pytorch repository'",
                "data": {"suggestion": True}
            }


class MockArxivAgent:
    """Mock arXiv research agent with proper OASF capabilities and A2A skills."""
    
    def __init__(self, agent_id: str = "arxiv_agent"):
        self.id = agent_id
        self.name = "arXiv Research Agent"
        self.description = "Specializes in academic paper research and summarization"
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent information."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": "1.0.0"
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return OASF capabilities (schema definitions)."""
        return [
            {
                "id": "arxiv_search",
                "type": "research.arxiv.search",
                "name": "arXiv Paper Search",
                "description": "Search arXiv for academic papers by keywords",
                "parameters": {
                    "query": {"type": "string", "required": True, "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5, "description": "Maximum results to return"}
                },
                "metadata": {
                    "examples": ["Search for transformer papers", "Find neural network research"],
                    "tags": ["arxiv", "search", "papers", "research"]
                }
            },
            {
                "id": "arxiv_summarize",
                "type": "research.arxiv.summarization",
                "name": "arXiv Paper Summarization",
                "description": "Summarize an arXiv paper given its ID or URL",
                "parameters": {
                    "paper_id": {"type": "string", "required": True, "description": "arXiv paper ID (e.g., 1706.03762)"}
                },
                "metadata": {
                    "examples": ["Summarize arXiv:1706.03762 (Attention Is All You Need)"],
                    "tags": ["arxiv", "summarization", "papers"]
                }
            }
        ]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a capability."""
        if capability_id == "arxiv_search":
            return self._handle_arxiv_search(inputs)
        elif capability_id == "arxiv_summarize":
            return self._handle_arxiv_summarize(inputs)
        else:
            return {"error": f"Unknown capability: {capability_id}"}
            
    def _handle_arxiv_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle arXiv search capability."""
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        # Mock search results
        mock_results = [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "published": "2017-06-12",
                "url": "https://arxiv.org/abs/1706.03762",
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
            },
            {
                "id": "2010.11929", 
                "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
                "authors": ["Alexey Dosovitskiy", "Lucas Beyer"],
                "summary": "While the Transformer architecture has become the de-facto standard for natural language processing tasks...",
                "published": "2020-10-22",
                "url": "https://arxiv.org/abs/2010.11929",
                "pdf_url": "https://arxiv.org/pdf/2010.11929.pdf"
            }
        ]
        
        return {
            "success": True,
            "results": mock_results[:max_results],
            "query": query,
            "capability_metadata": {
                "capability_id": "arxiv_search",
                "capability_type": "research.arxiv.search"
            }
        }
        
    def _handle_arxiv_summarize(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle arXiv paper summarization."""
        paper_id = inputs.get("paper_id", "")
        
        # Mock summarization
        return {
            "success": True,
            "summary": {
                "paper_id": paper_id,
                "title": "Attention Is All You Need",
                "key_contributions": [
                    "Introduced the Transformer architecture",
                    "Eliminated recurrence and convolutions entirely",
                    "Achieved state-of-the-art results on machine translation"
                ],
                "methodology": "Self-attention mechanism with multi-head attention",
                "results": "BLEU scores of 28.4 on WMT 2014 English-to-German translation",
                "significance": "Foundational work for modern NLP models like BERT and GPT"
            },
            "capability_metadata": {
                "capability_id": "arxiv_summarize",
                "capability_type": "research.arxiv.summarization"
            }
        }
        
    async def chat(self, message: str) -> Dict[str, Any]:
        """Chat interface (simulates MCP Tool)."""
        logger.info(f"arXiv agent chat: {message}")
        
        message_lower = message.lower()
        
        if "search" in message_lower or "find" in message_lower or "paper" in message_lower:
            # Extract search query
            query = message.replace("search for", "").replace("find", "").replace("papers about", "").strip()
            if not query:
                query = "transformer"
                
            result = self.invoke("arxiv_search", {"query": query, "max_results": 3})
            
            if result.get("success"):
                response = f"📚 **Found {len(result['results'])} papers:**\n\n"
                for paper in result["results"][:3]:
                    response += f"• **{paper['title']}** ({paper['published']})\n"
                    response += f"  Authors: {', '.join(paper['authors'])}\n"
                    response += f"  arXiv:{paper['id']} | [PDF]({paper['pdf_url']})\n\n"
                return {"response": response, "data": result}
            else:
                return {"response": f"❌ Search failed: {result.get('error', 'Unknown error')}", "data": result}
                
        elif "summari" in message_lower or "1706.03762" in message:
            # Default to summarizing attention paper
            paper_id = "1706.03762"
            if "arxiv:" in message_lower:
                # Extract paper ID if provided
                parts = message_lower.split("arxiv:")
                if len(parts) > 1:
                    paper_id = parts[1].split()[0]
                    
            result = self.invoke("arxiv_summarize", {"paper_id": paper_id})
            
            if result.get("success"):
                summary = result["summary"]
                response = f"📄 **Paper Summary:**\n\n"
                response += f"**Title:** {summary['title']}\n"
                response += f"**Paper ID:** arXiv:{summary['paper_id']}\n\n"
                response += f"**Key Contributions:**\n"
                for contrib in summary['key_contributions']:
                    response += f"• {contrib}\n"
                response += f"\n**Methodology:** {summary['methodology']}\n"
                response += f"**Results:** {summary['results']}\n"
                response += f"**Significance:** {summary['significance']}\n"
                return {"response": response, "data": result}
            else:
                return {"response": f"❌ Summarization failed: {result.get('error', 'Unknown error')}", "data": result}
        else:
            return {
                "response": "📚 I can help you **search** arXiv papers or **summarize** specific papers. Try:\n• 'search for transformer papers'\n• 'summarize arXiv:1706.03762'",
                "data": {"suggestion": True}
            }


class A2ASkillsDemo:
    """
    Demo showcasing the three-layer model:
    1. OASF Capabilities (schema definitions) 
    2. A2A Skills (concrete wire functions)
    3. MCP Tools (LLM chat interfaces)
    """
    
    def __init__(self):
        self.agents = {}
        self.a2a_adapters = {}
        self.running = False
        
    async def setup_agents(self) -> None:
        """Set up agents with A2A skill adapters."""
        # Create mock agents
        github_agent = MockGitHubAgent()
        arxiv_agent = MockArxivAgent()
        
        # Create A2A skill adapters for each agent
        github_adapter = DawnA2ASkillAdapter(github_agent, port=8081, host="localhost")
        arxiv_adapter = DawnA2ASkillAdapter(arxiv_agent, port=8082, host="localhost")
        
        # Store references
        self.agents["github"] = github_agent
        self.agents["arxiv"] = arxiv_agent
        self.a2a_adapters["github"] = github_adapter
        self.a2a_adapters["arxiv"] = arxiv_adapter
        
        logger.info("Set up 2 agents with A2A skill adapters")
        
    async def start_a2a_servers(self) -> None:
        """Start A2A servers for all agents."""
        for agent_name, adapter in self.a2a_adapters.items():
            try:
                await adapter.start_a2a_server()
                logger.info(f"Started A2A server for {agent_name}")
            except Exception as e:
                logger.error(f"Failed to start A2A server for {agent_name}: {e}")
                
        print(f"🎉 Started {len(self.a2a_adapters)} A2A skill servers!")
        self.running = True
        
    async def stop_a2a_servers(self) -> None:
        """Stop all A2A servers."""
        for agent_name, adapter in self.a2a_adapters.items():
            try:
                await adapter.stop_a2a_server()
                logger.info(f"Stopped A2A server for {agent_name}")
            except Exception as e:
                logger.error(f"Error stopping A2A server for {agent_name}: {e}")
        
        self.running = False
        print("🏁 Stopped all A2A servers")
        
    def show_architecture_info(self) -> None:
        """Display information about the three-layer architecture."""
        print("\n" + "="*80)
        print("🏗️  THREE-LAYER ARCHITECTURE DEMO")
        print("="*80)
        
        for agent_name, agent in self.agents.items():
            adapter = self.a2a_adapters[agent_name]
            
            print(f"\n🤖 **{agent.name}** ({agent_name})")
            print(f"   Description: {agent.description}")
            
            print(f"\n   📋 **OASF Capabilities** (Schema Definitions):")
            for cap in agent.get_capabilities():
                print(f"      • {cap['type']} - {cap['name']}")
                
            print(f"\n   ⚡ **A2A Skills** (Wire Functions):")
            for skill in adapter.get_skills():
                print(f"      • {skill.id} - {skill.name}")
                print(f"        Tags: {', '.join(skill.tags)}")
                
            print(f"\n   💬 **Chat Interface** (Simulates MCP Tools):")
            print(f"      • chat() method available for LLM interaction")
            
        print(f"\n🌐 **A2A Network:**")
        for agent_name, adapter in self.a2a_adapters.items():
            agent_card = adapter.get_agent_card()
            print(f"   • {agent_name}: {agent_card['url']}")
            print(f"     Skills: {len(agent_card['skills'])} exposed")
            
    async def run_test_sequence(self) -> None:
        """Run automated test sequence."""
        print("\n" + "="*60)
        print("🧪 RUNNING TEST SEQUENCE")
        print("="*60)
        
        test_commands = [
            ("github", "search for machine learning frameworks"),
            ("arxiv", "find papers about transformers"),
            ("github", "analyze pytorch repository"),
            ("arxiv", "summarize arXiv:1706.03762")
        ]
        
        for agent_name, command in test_commands:
            print(f"\n▶️  Testing {agent_name} agent: '{command}'")
            print("-" * 50)
            
            try:
                agent = self.agents[agent_name]
                result = await agent.chat(command)
                print(f"🤖 {agent.name} response:")
                print(result.get("response", "No response"))
                
                # Show underlying data
                data = result.get("data", {})
                if data.get("capability_metadata"):
                    metadata = data["capability_metadata"]
                    print(f"\n📊 Invoked capability: {metadata.get('capability_type', 'unknown')}.{metadata.get('capability_id', 'unknown')}")
                    
            except Exception as e:
                print(f"❌ Error testing {agent_name}: {e}")
                logger.error(f"Test error for {agent_name}: {e}")
                
        print(f"\n✅ Test sequence completed!")
        
    async def interactive_chat(self) -> None:
        """Run interactive chat mode."""
        print("\n" + "="*60)
        print("🗣️  INTERACTIVE CHAT MODE")
        print("="*60)
        print("Available agents: @github, @arxiv")
        print("Type 'quit' to exit, 'info' for architecture details")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if user_input.lower() == 'info':
                    self.show_architecture_info()
                    continue
                    
                # Parse agent selection
                if user_input.startswith('@'):
                    parts = user_input.split(' ', 1)
                    agent_name = parts[0][1:]  # Remove @
                    message = parts[1] if len(parts) > 1 else ""
                    
                    if agent_name in self.agents and message:
                        print(f"\n🤖 {self.agents[agent_name].name} thinking...")
                        
                        try:
                            result = await self.agents[agent_name].chat(message)
                            print(result.get("response", "No response"))
                        except Exception as e:
                            print(f"❌ Error: {e}")
                    else:
                        print("❌ Usage: @agent_name your message")
                        print("Available agents: @github, @arxiv")
                else:
                    print("💡 Use @agent_name to direct your message. Example: @github search for pytorch")
                    
            except (KeyboardInterrupt, EOFError):
                break
                
        print("\n👋 Chat session ended")


async def main():
    """Main entry point with test and interactive modes."""
    parser = argparse.ArgumentParser(description="A2A Skills Demo - Three Layer Architecture")
    parser.add_argument("--test", action="store_true", help="Run automated test sequence and exit")
    parser.add_argument("--info", action="store_true", help="Show architecture info and exit")
    args = parser.parse_args()
    
    demo = A2ASkillsDemo()
    
    try:
        # Setup
        await demo.setup_agents()
        await demo.start_a2a_servers()
        
        if args.info:
            demo.show_architecture_info()
            return
            
        if args.test:
            await demo.run_test_sequence()
        else:
            demo.show_architecture_info()
            await demo.interactive_chat()
            
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo error: {e}")
    finally:
        await demo.stop_a2a_servers()
        print("🏁 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 