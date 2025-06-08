"""
Base Demo Runner

This module provides the base class for running agent demonstrations
with different protocol configurations.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from dawn.core.agent import Agent
from dawn.protocols.base import ProtocolAdapter, ProtocolEndpoint
from dawn.agents.registry import registry
from dawn.utils.logging import get_logger, LogLevel, AgentContext, RequestContext


@dataclass
class RunningAgent:
    """Information about a running agent."""
    agent: Agent
    adapters: List[ProtocolAdapter]
    endpoints: List[ProtocolEndpoint]


class DemoRunner:
    """
    Simple demo runner for single agent/adapter combinations.
    
    This is used primarily for testing and simple demonstrations.
    """
    
    def __init__(
        self,
        agent: Agent,
        adapter: ProtocolAdapter,
        host: str = "localhost",
        port: int = 8080,
        show_ui: bool = True,
        message: Optional[str] = None
    ):
        """
        Initialize the demo runner.
        
        Args:
            agent: Agent instance to run
            adapter: Protocol adapter to use
            host: Host to bind to
            port: Port to bind to
            show_ui: Whether to show UI elements
            message: Optional message to send and exit
        """
        self.agent = agent
        self.adapter = adapter
        self.host = host
        self.port = port
        self.show_ui = show_ui
        self.message = message
        self._running = False
        self.logger = get_logger(__name__)
    
    async def start(self) -> None:
        """Start the agent and adapter."""
        self._running = True
        
        # Initialize agent
        await self.agent.initialize()
        
        # Start adapter
        endpoint = await self.adapter.start(self.agent, self.host, self.port)
        
        if self.show_ui:
            self._display_welcome()
            self._display_endpoint_info(endpoint)
            self._display_agent_card()
            self._display_instructions()
        
        # Run the main loop
        await self._run()
    
    async def stop(self) -> None:
        """Stop the agent and adapter."""
        self._running = False
        await self.adapter.stop()
        await self.agent.shutdown()
    
    async def send_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a message to the agent."""
        return await self.agent.process_message(message, context)
    
    async def run(self) -> None:
        """Run the demo (convenience method)."""
        await self.start()
    
    async def _run(self) -> None:
        """Main run loop."""
        if self.message:
            # Single message mode
            result = await self.send_message(self.message)
            if self.show_ui:
                if result.get("success"):
                    print(f"\n✅ Response: {result['response']}")
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            await self.stop()
        else:
            # Interactive mode
            await self._run_interactive_loop()
    
    async def _run_interactive_loop(self) -> None:
        """Run interactive chat loop."""
        while self._running:
            try:
                user_input = input("\n> ").strip()
                await self._handle_interactive_input(user_input)
            except (KeyboardInterrupt, EOFError):
                break
        
        await self.stop()
    
    async def _handle_interactive_input(self, user_input: str) -> None:
        """Handle interactive input."""
        if user_input.lower() in ['quit', 'exit', 'q']:
            self._running = False
            return
        
        if not user_input:
            return
        
        try:
            result = await self.send_message(user_input)
            if result.get("success"):
                print(f"\n🤖 {result['response']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            print(f"\n❌ Error: {str(e)}")
    
    def _display_welcome(self) -> None:
        """Display welcome message."""
        print("\n" + "="*70)
        print("🌅 Welcome to Dawn Agent Demo")
        print("="*70)
        print(f"Agent: {self.agent.name}")
        print(f"Protocol: {self.adapter.get_protocol_name()}")
        print("="*70)
    
    def _display_endpoint_info(self, endpoint: ProtocolEndpoint) -> None:
        """Display endpoint information."""
        print(f"\n🚀 Agent running at: {endpoint.base_url}")
    
    def _display_instructions(self) -> None:
        """Display usage instructions."""
        print("\n📋 Instructions:")
        print("- Type messages to chat with the agent")
        print("- Type 'quit' to exit")
        print("-" * 70)
    
    def _display_agent_card(self) -> None:
        """Display agent card."""
        if self.show_ui:
            print("\n🎴 Agent Card:")
            card = self.adapter.get_agent_card_for_protocol()
            print(f"  ID: {card.get('id', 'N/A')}")
            print(f"  Name: {card.get('name', 'N/A')}")
            print(f"  Description: {card.get('description', 'N/A')}")


class BaseDemoRunner:
    """
    Base class for demo runners with protocol injection.
    
    This provides a framework for running agents with various protocol
    combinations, handling lifecycle management, and providing interactive
    or automated demonstrations.
    """
    
    def __init__(
        self,
        agents: List[str],
        protocols: List[str],
        log_level: str = "INFO",
        base_port: int = 8080
    ):
        """
        Initialize the demo runner.
        
        Args:
            agents: List of agent names to run
            protocols: List of protocol names to enable
            log_level: Logging level (INFO, DEBUG, TRACE)
            base_port: Starting port number for services
        """
        self.agent_names = agents
        self.protocol_names = protocols
        self.base_port = base_port
        self.running_agents: Dict[str, RunningAgent] = {}
        
        # Set up logging
        self.logger = get_logger(__name__)
        level = getattr(LogLevel, log_level.upper(), LogLevel.INFO)
        self.logger.set_level(level)
        
        self.logger.info(
            f"Initialized demo runner with agents: {agents}, "
            f"protocols: {protocols}, log level: {log_level}"
        )
    
    def _get_next_port(self) -> int:
        """Get the next available port number."""
        port = self.base_port
        self.base_port += 1
        return port
    
    def _create_protocol_adapter(self, protocol: str) -> ProtocolAdapter:
        """
        Create a protocol adapter instance.
        
        This method should be overridden by subclasses to provide
        specific protocol adapter implementations.
        
        Args:
            protocol: Protocol name
            
        Returns:
            Protocol adapter instance
            
        Raises:
            NotImplementedError: If protocol is not supported
        """
        # Try to import the protocol adapter
        if protocol == "acp":
            try:
                from dawn.protocols.acp import ACPAdapter
                return ACPAdapter(logger=self.logger)
            except ImportError:
                raise ImportError(
                    f"Protocol '{protocol}' not available. "
                    f"Install with: pip install dawn-spec[{protocol}]"
                )
        elif protocol == "a2a":
            try:
                from dawn.protocols.a2a import A2AAdapter
                return A2AAdapter(logger=self.logger)
            except ImportError:
                raise ImportError(
                    f"Protocol '{protocol}' not available. "
                    f"Install with: pip install dawn-spec[{protocol}]"
                )
        elif protocol == "mcp":
            try:
                from dawn.protocols.mcp import MCPAdapter
                return MCPAdapter(logger=self.logger)
            except ImportError:
                raise ImportError(
                    f"Protocol '{protocol}' not available. "
                    f"Install with: pip install dawn-spec[{protocol}]"
                )
        else:
            raise NotImplementedError(f"Unknown protocol: {protocol}")
    
    async def start(self) -> None:
        """Start all agents with specified protocols."""
        self.logger.info("Starting agents...")
        
        for agent_name in self.agent_names:
            try:
                # Create agent
                agent = registry.create(agent_name)
                
                # Initialize agent
                await agent.initialize()
                
                # Log agent start
                with AgentContext(agent.agent_id):
                    self.logger.agent_start(
                        agent.agent_id,
                        agent_name,
                        self.protocol_names
                    )
                
                # Start protocol adapters
                adapters = []
                endpoints = []
                
                for protocol in self.protocol_names:
                    try:
                        adapter = self._create_protocol_adapter(protocol)
                        
                        # MCP doesn't use ports
                        if protocol == "mcp":
                            endpoint = await adapter.start(agent)
                        else:
                            port = self._get_next_port()
                            endpoint = await adapter.start(agent, port=port)
                        
                        adapters.append(adapter)
                        endpoints.append(endpoint)
                        
                        self.logger.info(
                            f"Started {protocol} for {agent_name} at {endpoint.url}"
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            f"Failed to start {protocol} for {agent_name}: {e}",
                            exception=e
                        )
                
                # Store running agent info
                self.running_agents[agent_name] = RunningAgent(
                    agent=agent,
                    adapters=adapters,
                    endpoints=endpoints
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to start agent {agent_name}: {e}",
                    exception=e
                )
        
        # Display summary
        self._display_summary()
    
    async def stop(self) -> None:
        """Stop all agents and protocols."""
        self.logger.info("Stopping agents...")
        
        for agent_name, info in self.running_agents.items():
            # Log agent stop
            with AgentContext(info.agent.agent_id):
                self.logger.agent_stop(info.agent.agent_id)
            
            # Stop protocol adapters
            for adapter in info.adapters:
                try:
                    await adapter.stop()
                except Exception as e:
                    self.logger.error(
                        f"Error stopping adapter: {e}",
                        exception=e
                    )
            
            # Shutdown agent
            try:
                await info.agent.shutdown()
            except Exception as e:
                self.logger.error(
                    f"Error shutting down agent: {e}",
                    exception=e
                )
        
        self.running_agents.clear()
        self.logger.info("All agents stopped")
    
    def _display_summary(self) -> None:
        """Display a summary of running agents."""
        print("\n" + "="*70)
        print("🚀 RUNNING AGENTS")
        print("="*70)
        
        for name, info in self.running_agents.items():
            print(f"\n{name.upper()}:")
            print(f"  ID: {info.agent.agent_id}")
            print(f"  Description: {info.agent.description}")
            
            if info.endpoints:
                print("  Endpoints:")
                for endpoint in info.endpoints:
                    print(f"    - {endpoint.protocol}: {endpoint.url}")
            
            tools = info.agent.get_tools()
            if tools:
                print(f"  Tools: {', '.join(tools)}")
        
        print("\n" + "="*70)
    
    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\n💬 Interactive Chat")
        print("Type '@agent_name message' to chat with an agent")
        print("Type 'quit' to exit")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Parse agent and message
                if user_input.startswith('@'):
                    parts = user_input.split(' ', 1)
                    if len(parts) < 2:
                        print("❌ Please provide a message after the agent name")
                        continue
                    
                    agent_name = parts[0][1:]  # Remove @
                    message = parts[1]
                    
                    if agent_name not in self.running_agents:
                        print(f"❌ Unknown agent: {agent_name}")
                        print(f"Available: {', '.join(self.running_agents.keys())}")
                        continue
                    
                    # Process with agent
                    await self._process_chat(agent_name, message)
                else:
                    print("❌ Please start with @agent_name")
                    
            except (KeyboardInterrupt, EOFError):
                break
        
        print("\n👋 Goodbye!")
    
    async def _process_chat(self, agent_name: str, message: str) -> None:
        """Process a chat message with an agent."""
        info = self.running_agents[agent_name]
        
        print(f"\n🤖 {agent_name} is thinking...")
        
        try:
            # Create request context
            with RequestContext(f"chat-{agent_name}"):
                result = await info.agent.process_message(message)
            
            if result.get("success"):
                print(f"\n{result['response']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Chat error: {e}", exception=e)
            print(f"\n❌ Error: {str(e)}")
    
    async def run_demo_sequence(self, commands: List[Tuple[str, str]]) -> None:
        """
        Run a sequence of demo commands.
        
        Args:
            commands: List of (agent_name, message) tuples
        """
        print("\n" + "="*70)
        print("🎭 DEMO SEQUENCE")
        print("="*70)
        
        for i, (agent_name, message) in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] @{agent_name}: {message}")
            print("-" * 60)
            
            if agent_name in self.running_agents:
                await self._process_chat(agent_name, message)
            else:
                print(f"❌ Agent {agent_name} not running")
            
            # Brief pause between commands
            await asyncio.sleep(1)
        
        print("\n" + "="*70)
        print("✅ Demo sequence completed")
        print("="*70) 