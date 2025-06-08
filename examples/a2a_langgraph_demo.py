#!/usr/bin/env python3
"""
A2A LangGraph Demo - Multi-Protocol Research Agents

This demo showcases LangGraph-powered agents that support:
- Google A2A protocol
- AGNTCY ACP protocol
- MCP (Model Context Protocol)
- Real GitHub and arXiv API integration
- Interactive CLI with @agent mentions

Usage:
    uv run python examples/a2a_langgraph_demo.py [--test]
    uv run python examples/a2a_langgraph_demo.py --message "@arxiv summarize paper 1706.03762"
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our agents
from agents.langgraph_github_agent import LangGraphGitHubAgent
from agents.langgraph_arxiv_agent import LangGraphArXivAgent
from agents.langgraph_synthesis_agent import LangGraphSynthesisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiProtocolAgentDemo:
    """
    Demo class that manages multiple LangGraph agents with protocol support.
    """
    
    def __init__(self):
        self.agents = {}
        self.running_servers = []
        
    async def start_all_agents(self):
        """Initialize and start all agents with their protocol servers."""
        print("🚀 Starting Multi-Protocol LangGraph Agents...")
        print("-" * 60)
        
        # Check for LLM API keys
        has_llm = any([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY")
        ])
        
        if not has_llm:
            print("❌ No LLM API keys found!")
            print("\nPlease set at least one of the following in your .env file:")
            print("- OPENAI_API_KEY")
            print("- ANTHROPIC_API_KEY")
            print("- GOOGLE_API_KEY")
            print("\nExiting...")
            return False
        
        # Create agents
        agents_config = [
            ("github", LangGraphGitHubAgent, 8081),
            ("arxiv", LangGraphArXivAgent, 8082),
            ("synthesis", LangGraphSynthesisAgent, 8083)
        ]
        
        for agent_name, agent_class, port in agents_config:
            try:
                logger.info(f"Creating {agent_name} agent...")
                agent = agent_class(name=f"{agent_name.capitalize()} Agent", port=port)
                self.agents[agent_name] = agent
                
                # Start protocol servers
                await agent.start_all_protocols()
                
                print(f"✅ {agent_name.capitalize()} agent ready")
                
                # Show agent card info
                card = agent.get_agent_card()
                print(f"   ID: {card['id']}")
                print(f"   Capabilities: {len(card['capabilities'])}")
                print(f"   Protocols: A2A, ACP, MCP")
                
            except Exception as e:
                logger.error(f"Failed to start {agent_name} agent: {e}")
                print(f"❌ {agent_name.capitalize()} agent failed: {str(e)}")
                
                # If it's an LLM initialization error, show which LLM was tried
                if "No LLM API keys found" in str(e):
                    return False
        
        print(f"\n🎉 Started {len(self.agents)} agents successfully!")
        return True
    
    async def stop_all_agents(self):
        """Stop all running agents and their protocol servers."""
        print("\n🛑 Stopping all agents...")
        for agent_name, agent in self.agents.items():
            try:
                await agent.stop_all_protocols()
                print(f"✅ Stopped {agent_name} agent")
            except Exception as e:
                logger.error(f"Error stopping {agent_name} agent: {e}")
    
    async def interactive_chat(self):
        """Run the interactive chat interface."""
        print("\n" + "="*70)
        print("🤖 MULTI-PROTOCOL LANGGRAPH RESEARCH AGENTS")
        print("="*70)
        print("\n📋 Available agents:")
        print("• @github      - GitHub repository research (real API)")
        print("• @arxiv       - arXiv paper research (real API)")
        print("• @synthesis   - Multi-source synthesis and analysis")
        print("\n💡 Example queries:")
        print("• @github search for langchain repositories")
        print("• @github analyze agntcy/acp-sdk")
        print("• @arxiv find papers on transformer architectures")
        print("• @arxiv summarize 1706.03762")
        print("• @synthesis compare PyTorch and TensorFlow")
        print("\n🔧 Features:")
        print("• LangGraph ReACT reasoning")
        print("• Real API integration (not mocks)")
        print("• Multi-protocol support (A2A, ACP, MCP)")
        print("• Full conversation context")
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
                
                # Show agent cards
                if user_input.lower() == 'cards':
                    print("\n📇 Agent Cards:")
                    for name, agent in self.agents.items():
                        # Get tools directly from agent
                        tools = agent.get_tools()
                        capabilities = agent.get_capabilities()
                        
                        print(f"\n{name.upper()} AGENT:")
                        print(f"  🆔 ID: {agent.agent_id}")
                        print(f"  📝 Description: {agent.description}")
                        print(f"  🔧 Tools: {', '.join([tool.name for tool in tools])}")
                        print(f"  ⚡ Capabilities: {len(capabilities)} defined")
                        print(f"  🌐 Port: {agent.port}")
                        print(f"  🔗 Agent Card: http://localhost:{agent.port}/agent.json")
                        print(f"  🔗 Well-known: http://localhost:{agent.port}/.well-known/agent.json")
                        
                        # Show first few capabilities
                        if capabilities:
                            print(f"  📋 Sample capabilities:")
                            for cap in capabilities[:2]:
                                print(f"     • {cap.get('name', cap.get('id', 'Unknown'))}")
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
                        start_time = datetime.now()
                        result = await agent.chat(message)
                        end_time = datetime.now()
                        
                        print(f"\n{result['response']}")
                        
                        # Show timing
                        duration = (end_time - start_time).total_seconds()
                        print(f"\n⏱️  Response time: {duration:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Chat error with {agent_name}: {e}")
                        print(f"❌ Error: {str(e)}")
                        
                else:
                    print("❌ Please start your message with an agent name (e.g., @github, @arxiv, @synthesis)")
                    print("💡 Type 'cards' to see agent details")
                    
            except (KeyboardInterrupt, EOFError):
                break
        
        print("\n👋 Thanks for using the Multi-Protocol LangGraph Demo!")


async def run_single_message(demo: MultiProtocolAgentDemo, message: str) -> str:
    """Run a single message query and return the response."""
    print("\n" + "="*70)
    print("📨 RUNNING SINGLE MESSAGE")
    print("="*70)
    
    # Parse the message to extract agent and query
    if not message.startswith('@'):
        raise ValueError("Message must start with an agent name (e.g., @github, @arxiv, @synthesis)")
    
    parts = message.split(' ', 1)
    if len(parts) < 2:
        raise ValueError("Please provide a message after the agent name")
    
    agent_name = parts[0][1:]  # Remove @
    query = parts[1]
    
    if agent_name not in demo.agents:
        available = ', '.join(demo.agents.keys())
        raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
    
    print(f"🎯 Sending to @{agent_name}: {query}")
    print("-" * 70)
    
    try:
        agent = demo.agents[agent_name]
        
        start_time = datetime.now()
        response = await agent.chat(query)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        if response.get('success'):
            print(f"✅ Success ({duration:.1f}s)")
            response_text = response['response']
            print(f"📝 Response: {response_text}")
            return response_text
        else:
            error_text = response.get('response', 'Unknown error')
            print(f"❌ Failed ({duration:.1f}s)")
            print(f"📝 Error: {error_text}")
            return f"Error: {error_text}"
            
    except Exception as e:
        error_text = f"Exception: {str(e)}"
        print(f"❌ Exception: {error_text}")
        return error_text


async def run_test_sequence(demo: MultiProtocolAgentDemo):
    """Run automated test sequence."""
    print("\n" + "="*70)
    print("🧪 RUNNING TEST SEQUENCE")
    print("="*70)
    
    test_commands = [
        ("@github", "search for popular Python web frameworks"),
        ("@github", "analyze the agntcy/acp-sdk repository"),
        ("@arxiv", "find recent papers on large language models"),
        ("@arxiv", "summarize paper 1706.03762"),
        ("@synthesis", "compare FastAPI and Flask for building APIs")
    ]
    
    for i, (agent_cmd, query) in enumerate(test_commands, 1):
        print(f"\n📋 Test {i}/{len(test_commands)}: {agent_cmd} {query}")
        print("-" * 60)
        
        agent_name = agent_cmd[1:]  # Remove @
        
        try:
            if agent_name in demo.agents:
                agent = demo.agents[agent_name]
                
                start_time = datetime.now()
                response = await agent.chat(query)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                
                if response.get('success'):
                    print(f"✅ Success ({duration:.1f}s)")
                    # Show truncated response
                    response_text = response['response']
                    if len(response_text) > 500:
                        print(f"📝 Response: {response_text[:500]}...")
                        print("   [Response truncated for test output]")
                    else:
                        print(f"📝 Response: {response_text}")
                else:
                    print(f"❌ Failed ({duration:.1f}s)")
                    print(f"📝 Error: {response.get('response', 'Unknown error')}")
            else:
                print(f"❌ Agent {agent_name} not found")
                
        except Exception as e:
            print(f"❌ Test error: {str(e)}")
        
        # Brief pause between tests
        await asyncio.sleep(1)
    
    print("\n" + "="*70)
    print("🏁 TEST SEQUENCE COMPLETED")
    print("="*70)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Protocol LangGraph Research Agents Demo"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run automated test sequence"
    )
    parser.add_argument(
        "--message", "-m",
        type=str,
        help='Send a single message to an agent (e.g., "@arxiv summarize paper 1706.03762")'
    )
    args = parser.parse_args()
    
    print("🎯 Multi-Protocol LangGraph Research Agents")
    print("=" * 50)
    
    demo = MultiProtocolAgentDemo()
    captured_response = None
    agent_name = None
    
    try:
        # Start all agents
        success = await demo.start_all_agents()
        
        if not success:
            print("\n❌ Failed to start agents. Please check your configuration.")
            return
        
        if args.test:
            # Run test sequence
            await run_test_sequence(demo)
        elif args.message:
            # Run single message mode
            try:
                captured_response = await run_single_message(demo, args.message)
                # Extract agent name for final output
                agent_name = args.message.split(' ', 1)[0]  # Includes the @
            except ValueError as e:
                print(f"❌ Message format error: {e}")
                return
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
        
        # Print captured response if in single message mode
        if captured_response and agent_name:
            print(f"\nResponse from {agent_name}:")
            print(captured_response)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ This demo requires Python 3.10 or higher")
        sys.exit(1)
    
    # Run the demo
    asyncio.run(main()) 