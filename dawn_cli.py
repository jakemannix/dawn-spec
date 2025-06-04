#!/usr/bin/env python
"""
CLI for the DAWN/AGNTCY implementation.

This command-line interface allows interaction with the DAWN implementation,
providing commands to list agents, invoke capabilities, and run demos.
"""
import argparse
import sys
import os
import json
import time
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dawn_cli")

# Import DAWN components
from src.config import APIConfig, AgentConfig, check_configuration
from src.interfaces import IAgent
from src.anthropic_agent import create_anthropic_agent

# Try to import Gemini, but make it optional
try:
    from src.gemini_agent import create_gemini_agent
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: Gemini agent not available (protobuf compatibility issue)")
    GEMINI_AVAILABLE = False


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="DAWN Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dawn_cli.py config                          # Show current configuration
  dawn_cli.py run-demo research               # Run the research agent demo
  dawn_cli.py run-demo anthropic              # Run the Anthropic agent demo
  dawn_cli.py agent-info anthropic            # Show information about the Anthropic agent
  dawn_cli.py invoke anthropic text-generation --prompt "Write a poem about AI agents"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    # run-demo command
    demo_parser = subparsers.add_parser("run-demo", help="Run a demo")
    demo_parser.add_argument(
        "demo",
        choices=["research", "anthropic", "gemini", "openai", "weather-email", "basic", "agp-gateway"],
        help="Demo to run"
    )
    
    # agent-info command
    agent_info_parser = subparsers.add_parser("agent-info", help="Show agent information")
    agent_info_parser.add_argument(
        "agent",
        choices=["anthropic", "gemini", "openai", "research"],
        help="Agent to show information for"
    )
    
    # invoke command
    invoke_parser = subparsers.add_parser("invoke", help="Invoke an agent capability")
    invoke_parser.add_argument(
        "agent",
        choices=["anthropic", "gemini", "openai", "research"],
        help="Agent to invoke"
    )
    invoke_parser.add_argument(
        "capability",
        help="Capability ID to invoke"
    )
    invoke_parser.add_argument(
        "--prompt",
        help="Prompt for text generation"
    )
    invoke_parser.add_argument(
        "--text",
        help="Text content for summarization, classification, etc."
    )
    invoke_parser.add_argument(
        "--question",
        help="Research question for research agent"
    )
    invoke_parser.add_argument(
        "--json-input",
        help="JSON string containing input parameters"
    )
    invoke_parser.add_argument(
        "--input-file",
        help="Path to JSON file containing input parameters"
    )
    
    return parser


def show_config():
    """Show the current configuration."""
    config = check_configuration()
    
    print("=== DAWN Configuration ===\n")
    
    print("Available API Providers:")
    for provider, available in config["providers"].items():
        status = "✅ Configured" if available else "❌ Not configured"
        print(f"- {provider}: {status}")
    
    print(f"\nPrincipal Agent: {config['principal_agent']}")
    print(f"ACP Server: {config['acp_server']}")
    print(f"Auth Enabled: {config['auth_enabled']}")
    
    print("\nDefault Settings:")
    print(f"- Model: {AgentConfig.DEFAULT_MODEL}")
    print(f"- Temperature: {AgentConfig.TEMPERATURE}")
    print(f"- Max Tokens: {AgentConfig.MAX_TOKENS}")


def get_agent(agent_type: str) -> Optional[IAgent]:
    """
    Get an agent instance by type.
    
    Args:
        agent_type: Type of agent to create
        
    Returns:
        Agent instance or None if creation fails
    """
    if agent_type == "anthropic":
        if not APIConfig.is_anthropic_configured():
            print("Error: Anthropic API is not configured.")
            print("Please set ANTHROPIC_API_KEY in your .env file.")
            return None
            
        return create_anthropic_agent(
            name="DAWN Claude Agent",
            description="Claude-powered agent implementing DAWN architecture"
        )
    elif agent_type == "gemini":
        if not GEMINI_AVAILABLE:
            print("Error: Gemini agent is not available due to protobuf compatibility issues.")
            return None
            
        if not APIConfig.is_gemini_configured():
            print("Error: Gemini API is not configured.")
            print("Please set GEMINI_API_KEY in your .env file.")
            return None
            
        return create_gemini_agent(
            name="DAWN Gemini Agent",
            description="Gemini-powered agent implementing DAWN architecture"
        )
    elif agent_type == "openai":
        # This would import and create an OpenAI-based agent
        if not APIConfig.is_openai_configured():
            print("Error: OpenAI API is not configured.")
            print("Please set OPENAI_API_KEY in your .env file.")
            return None
            
        print("OpenAI agent creation not implemented in this CLI version.")
        return None
    elif agent_type == "research":
        if not APIConfig.is_openai_configured():
            print("Error: The research agent requires OpenAI API to be configured.")
            print("Please set OPENAI_API_KEY in your .env file.")
            return None
            
        print("Research agent creation not implemented in this CLI version.")
        print("Please use the run-demo command to run the research agent demo.")
        return None
    else:
        print(f"Error: Unknown agent type: {agent_type}")
        return None


def show_agent_info(agent_type: str):
    """
    Show information about an agent.
    
    Args:
        agent_type: Type of agent to show information for
    """
    agent = get_agent(agent_type)
    if not agent:
        return
    
    info = agent.get_info()
    capabilities = agent.get_capabilities()
    
    print(f"=== {info['name']} ===\n")
    print(f"ID: {info['id']}")
    print(f"Description: {info['description']}")
    
    if "provider" in info:
        print(f"Provider: {info['provider']}")
    if "model" in info:
        print(f"Model: {info['model']}")
    
    print("\nCapabilities:")
    for capability in capabilities:
        print(f"- {capability['name']} (ID: {capability['id']})")
        print(f"  {capability['description']}")
    
    # Check health
    health = agent.health_check()
    print(f"\nHealth Check: {'✅ Healthy' if health else '❌ Unhealthy'}")


def run_demo(demo_name: str):
    """
    Run a specified demo.
    
    Args:
        demo_name: Name of the demo to run
    """
    if demo_name == "research":
        demo_module = "examples.research_agent_demo"
    elif demo_name == "anthropic":
        demo_module = "examples.anthropic_agent_demo"
    elif demo_name == "gemini":
        demo_module = "examples.gemini_agent_demo"
    elif demo_name == "openai":
        demo_module = "examples.openai_agent_demo"
    elif demo_name == "weather-email":
        demo_module = "examples.weather_email_demo"
    elif demo_name == "basic":
        demo_module = "examples.basic_agent_interaction"
    elif demo_name == "agp-gateway":
        demo_module = "examples.agp_gateway_demo"
    else:
        print(f"Error: Unknown demo: {demo_name}")
        return
    
    try:
        # Import the demo module
        __import__(demo_module)
        module = sys.modules[demo_module]
        
        # Run the demo
        if hasattr(module, "run_demo"):
            module.run_demo()
        else:
            print(f"Error: Demo module {demo_module} does not have a run_demo function.")
    except ModuleNotFoundError:
        print(f"Error: Demo module {demo_module} not found.")
    except Exception as e:
        print(f"Error running demo: {str(e)}")


def parse_input_parameters(args) -> Dict[str, Any]:
    """
    Parse input parameters from various sources.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of input parameters
    """
    inputs = {}
    
    # Add individual parameters
    if args.prompt:
        inputs["prompt"] = args.prompt
    if args.text:
        inputs["text"] = args.text
    if args.question:
        inputs["question"] = args.question
    
    # Parse JSON input if provided
    if args.json_input:
        try:
            json_inputs = json.loads(args.json_input)
            inputs.update(json_inputs)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON input: {str(e)}")
    
    # Read from input file if provided
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                file_inputs = json.load(f)
                inputs.update(file_inputs)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading input file: {str(e)}")
    
    return inputs


def invoke_capability(agent_type: str, capability_id: str, args):
    """
    Invoke an agent capability with the provided inputs.
    
    Args:
        agent_type: Type of agent to invoke
        capability_id: ID of the capability to invoke
        args: Command line arguments containing input parameters
    """
    agent = get_agent(agent_type)
    if not agent:
        return
    
    # Parse input parameters
    inputs = parse_input_parameters(args)
    
    if not inputs:
        print("Error: No input parameters provided.")
        return
    
    print(f"Invoking {agent_type} agent capability: {capability_id}")
    print(f"Inputs: {json.dumps(inputs, indent=2)}")
    
    try:
        start_time = time.time()
        result = agent.invoke(capability_id, inputs)
        end_time = time.time()
        
        print(f"\nResults (completed in {end_time - start_time:.2f} seconds):")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            if "retry_after" in result:
                print(f"Retry after: {result['retry_after']} seconds")
        else:
            # Print result in a formatted way
            if "content" in result:
                print("\n--- Content ---\n")
                print(result["content"])
                print("\n--- End Content ---\n")
                
                # Print other metadata
                metadata = {k: v for k, v in result.items() if k != "content"}
                if metadata:
                    print("Metadata:")
                    print(json.dumps(metadata, indent=2))
            else:
                # Just print the whole result
                print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error invoking capability: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "config":
        show_config()
    elif args.command == "run-demo":
        run_demo(args.demo)
    elif args.command == "agent-info":
        show_agent_info(args.agent)
    elif args.command == "invoke":
        invoke_capability(args.agent, args.capability, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()