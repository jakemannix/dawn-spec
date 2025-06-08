#!/usr/bin/env python3
"""
Pluggable Agents Demo

This demo showcases the new pluggable agent architecture that allows
seamless switching between different agent implementations:

1. Text Matching (original string-based intent detection)
2. LangGraph (LLM-powered ReACT and CoALA patterns)  
3. Future: OpenAI SDK, AutoGen, etc.

All implementations share the same MCP tools and A2A skills,
ensuring consistency across different agent backends.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent_factory import AgentFactory, create_default_orchestrator
from src.agent_core import AgentImplementationType


class PluggableAgentsDemo:
    """Demo class showcasing pluggable agent architecture"""
    
    def __init__(self):
        self.orchestrator = None
        self.factory = None
    
    async def initialize(self):
        """Initialize the agent system"""
        print("🚀 Initializing Pluggable Agent System...")
        
        self.factory = AgentFactory()
        self.orchestrator = await self.factory.create_orchestrator()
        
        # Register some mock MCP tools and A2A skills for demo
        await self._setup_demo_tools()
        
        print("✅ Agent system initialized successfully!")
        return True
    
    async def _setup_demo_tools(self):
        """Set up demo MCP tools and A2A skills"""
        from src.mcp_integration import MCPTool
        from src.a2a_integration_v2 import A2ASkill, A2AMessage
        
        # Mock MCP tools
        class MockGitHubTool(MCPTool):
            def __init__(self):
                super().__init__("github_search", "Search GitHub repositories")
            
            async def call(self, arguments):
                query = arguments.get("query", "")
                return f"Mock GitHub results for '{query}': [langchain/langchain, microsoft/vscode]"
        
        class MockArxivTool(MCPTool):
            def __init__(self):
                super().__init__("arxiv_search", "Search arXiv papers")
            
            async def call(self, arguments):
                query = arguments.get("query", "")
                return f"Mock arXiv results for '{query}': [Paper 1: Advanced AI, Paper 2: Language Models]"
        
        # Register MCP tools
        self.orchestrator.register_mcp_tool(MockGitHubTool())
        self.orchestrator.register_mcp_tool(MockArxivTool())
        
        # Mock A2A skills
        async def mock_synthesis_handler(inputs):
            return "Mock synthesis: Combined research from multiple sources shows emerging trends in AI."
        
        synthesis_skill = A2ASkill(
            id="research.synthesis.synthesize",
            name="Research Synthesis",
            description="Synthesize research findings",
            tags=["research", "synthesis", "ai"],
            handler=mock_synthesis_handler
        )
        
        # Register A2A skills
        self.orchestrator.register_a2a_skill(synthesis_skill)
    
    async def run_demo(self):
        """Run the complete pluggable agents demo"""
        if not await self.initialize():
            return
        
        print("\n" + "="*60)
        print("🎯 PLUGGABLE AGENTS ARCHITECTURE DEMO")
        print("="*60)
        
        # Show system architecture
        await self._show_architecture()
        
        # Demo 1: Text Matching Agent
        await self._demo_text_matching()
        
        # Demo 2: LangGraph Agent (if available)
        await self._demo_langgraph()
        
        # Demo 3: Implementation Switching
        await self._demo_switching()
        
        # Demo 4: Health Monitoring
        await self._demo_health_monitoring()
        
        print("\n🎉 Demo completed! The pluggable architecture allows:")
        print("   • Easy switching between agent implementations")
        print("   • Shared MCP tools and A2A skills across all agents")
        print("   • Consistent APIs regardless of underlying technology")
        print("   • Simple configuration management")
        print("   • Runtime implementation switching")
    
    async def _show_architecture(self):
        """Show the system architecture"""
        print("\n📐 SYSTEM ARCHITECTURE")
        print("-" * 30)
        
        health = await self.orchestrator.health_check()
        print(f"Active Implementation: {health['active_implementation']}")
        print("Available Implementations:")
        for impl, status in health['implementations'].items():
            status_icon = "✅" if status['status'] == 'healthy' else "❌"
            print(f"  {status_icon} {impl}: {status['status']}")
        
        print("\n🔧 Three-Layer Architecture:")
        print("  📡 A2A Skills (Wire Protocol)")
        print("  🏷️  OASF Capabilities (Schema/Validation)")  
        print("  🛠️  MCP Tools (LLM Integration)")
    
    async def _demo_text_matching(self):
        """Demo the text matching implementation"""
        print("\n🎯 TEXT MATCHING AGENT DEMO")
        print("-" * 35)
        
        # Ensure we're using text matching
        await self.orchestrator.set_active_implementation(
            AgentImplementationType.TEXT_MATCHING
        )
        
        test_queries = [
            "search github for langchain",
            "find papers about transformers on arxiv",
            "synthesize research on neural networks",
            "help"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            response = await self.orchestrator.process_request(query)
            print(f"💬 Response: {response.response_text}")
            if response.tools_used:
                print(f"🛠️  Tools used: {response.tools_used}")
            if response.skills_invoked:
                print(f"📡 Skills invoked: {response.skills_invoked}")
    
    async def _demo_langgraph(self):
        """Demo the LangGraph implementation"""
        print("\n🧠 LANGGRAPH AGENT DEMO")
        print("-" * 30)
        
        try:
            await self.orchestrator.set_active_implementation(
                AgentImplementationType.LANGGRAPH
            )
            
            print("✅ LangGraph implementation active")
            
            # Test with a complex query that benefits from LLM reasoning
            query = "I need to find recent papers about large language models and also look for related GitHub repositories. Can you help synthesize the findings?"
            
            print(f"\n📝 Complex Query: {query}")
            response = await self.orchestrator.process_request(query)
            
            print(f"💬 Response: {response.response_text}")
            if response.reasoning_trace:
                print("🧭 Reasoning Trace:")
                for i, step in enumerate(response.reasoning_trace, 1):
                    print(f"   {i}. {step}")
            
            print(f"🎯 Confidence: {response.confidence_score:.2f}")
            
        except Exception as e:
            print(f"❌ LangGraph not available: {e}")
            print("💡 To enable LangGraph, install dependencies:")
            print("   pip install langgraph langchain-openai langchain-anthropic")
    
    async def _demo_switching(self):
        """Demo runtime implementation switching"""
        print("\n🔄 RUNTIME IMPLEMENTATION SWITCHING")
        print("-" * 45)
        
        query = "search github for AI projects"
        
        # Test with each available implementation
        implementations = [
            AgentImplementationType.TEXT_MATCHING,
            AgentImplementationType.LANGGRAPH
        ]
        
        for impl_type in implementations:
            try:
                await self.orchestrator.set_active_implementation(impl_type)
                print(f"\n🔧 Switched to: {impl_type.value}")
                
                response = await self.orchestrator.process_request(query)
                print(f"💬 Response: {response.response_text[:100]}...")
                
                impl_used = response.metadata.get("implementation_used", "unknown")
                print(f"✅ Confirmed using: {impl_used}")
                
            except Exception as e:
                print(f"❌ {impl_type.value} failed: {e}")
    
    async def _demo_health_monitoring(self):
        """Demo health monitoring and diagnostics"""
        print("\n🏥 HEALTH MONITORING & DIAGNOSTICS")
        print("-" * 40)
        
        health = await self.orchestrator.health_check()
        
        print("📊 System Status:")
        print(f"   Active: {health['active_implementation']}")
        print("   Implementations:")
        
        for impl, status in health['implementations'].items():
            status_icon = "🟢" if status['status'] == 'healthy' else "🔴"
            print(f"     {status_icon} {impl}: {status['status']}")
        
        # Show configuration
        config = self.factory.get_config()
        print(f"\n⚙️  Configuration:")
        print(f"   Default: {config.get('default_implementation', 'unknown')}")
        print(f"   Enabled: {[k for k, v in config.get('implementations', {}).items() if v.get('enabled')]}")


async def interactive_demo():
    """Run interactive demo with user input"""
    demo = PluggableAgentsDemo()
    await demo.initialize()
    
    print("\n🎮 INTERACTIVE MODE")
    print("Commands:")
    print("  @text     - Switch to text matching")
    print("  @lang     - Switch to LangGraph") 
    print("  @health   - Show system health")
    print("  @quit     - Exit")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n💭 You: ").strip()
            
            if user_input == "@quit":
                break
            elif user_input == "@text":
                await demo.orchestrator.set_active_implementation(
                    AgentImplementationType.TEXT_MATCHING
                )
                print("✅ Switched to text matching")
                continue
            elif user_input == "@lang":
                try:
                    await demo.orchestrator.set_active_implementation(
                        AgentImplementationType.LANGGRAPH
                    )
                    print("✅ Switched to LangGraph")
                except Exception as e:
                    print(f"❌ LangGraph not available: {e}")
                continue
            elif user_input == "@health":
                health = await demo.orchestrator.health_check()
                print(f"📊 Active: {health['active_implementation']}")
                for impl, status in health['implementations'].items():
                    print(f"  {impl}: {status['status']}")
                continue
            
            if user_input:
                response = await demo.orchestrator.process_request(user_input)
                impl_used = response.metadata.get("implementation_used", "unknown")
                print(f"🤖 Agent ({impl_used}): {response.response_text}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


async def main():
    """Main demo entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_demo()
    else:
        demo = PluggableAgentsDemo()
        await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 