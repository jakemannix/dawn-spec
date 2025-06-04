"""
Anthropic Claude Agent Demo implementing DAWN architecture.

This demo showcases the use of Anthropic Claude models within the DAWN framework.
It demonstrates various capabilities of the AnthropicAgent.
"""
import sys
import os
import json
from typing import Dict, Any
import time

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anthropic_agent import create_anthropic_agent, AnthropicAgent
from src.config import APIConfig, check_configuration


def display_divider(title: str = None):
    """Display a divider with an optional title."""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")


def check_environment():
    """Check if the environment is properly configured."""
    if not APIConfig.is_anthropic_configured():
        print("Error: Anthropic API is not configured.")
        print("Please set ANTHROPIC_API_KEY in your .env file.")
        print("You can copy template.env to .env and update with your API key.")
        return False
    return True


def text_generation_demo(agent: AnthropicAgent):
    """Demonstrate text generation capability."""
    display_divider("Text Generation Demo")
    
    prompt = "Write a short poem about artificial intelligence and collaboration."
    
    print(f"Prompt: {prompt}\n")
    
    result = agent.invoke("text-generation", {"prompt": prompt})
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("Generated Text:")
    print(result["content"])
    print(f"\nTokens: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")


def chat_demo(agent: AnthropicAgent):
    """Demonstrate chat capability."""
    display_divider("Chat Demo")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in explaining DAWN architecture concepts."},
        {"role": "user", "content": "What is the role of a Principal Agent in the DAWN architecture?"}
    ]
    
    print("Messages:")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        print(f"{role.capitalize()}: {content}")
    print()
    
    result = agent.invoke("chat", {"messages": messages})
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("Response:")
    print(result["content"])
    print(f"\nTokens: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")


def summarization_demo(agent: AnthropicAgent):
    """Demonstrate summarization capability."""
    display_divider("Summarization Demo")
    
    text = """
    The Distributed Agents in a Worldwide Network (DAWN) architecture is a framework
    for building distributed agent systems. It defines how specialized agents can work
    together to accomplish complex tasks through standardized protocols. The framework
    consists of Principal Agents that orchestrate tasks, Gateway Agents that manage
    discovery and registration, and various specialized Tool Agents that perform specific
    functions. Communication happens through the Agent Connect Protocol (ACP) and Agent
    Gateway Protocol (AGP), which standardize how agents discover each other and exchange
    messages. This architecture enables the creation of flexible, scalable networks of
    cooperative AI systems that can leverage each other's capabilities.
    """
    
    print(f"Original Text: {text}\n")
    
    result = agent.invoke("summarization", {"text": text, "max_length": 50})
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("Summary:")
    print(result["content"])
    print(f"\nTokens: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")


def classification_demo(agent: AnthropicAgent):
    """Demonstrate classification capability."""
    display_divider("Classification Demo")
    
    text = "The Principal Agent coordinates task execution across specialized agents, breaking down complex requests into manageable subtasks and assigning them to the most suitable agents."
    categories = ["Architecture Component", "Protocol Definition", "Implementation Detail", "Theoretical Concept"]
    
    print(f"Text to classify: {text}")
    print(f"Categories: {categories}\n")
    
    result = agent.invoke("classification", {"text": text, "categories": categories})
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("Classification:")
    print(result["content"])
    print(f"\nTokens: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")


def extraction_demo(agent: AnthropicAgent):
    """Demonstrate information extraction capability."""
    display_divider("Information Extraction Demo")
    
    text = """
    DAWN architecture was proposed by Cisco in 2023. The Principal Agent is responsible for task orchestration,
    while Gateway Agents handle discovery and registration. Tool Agents implement specific capabilities
    that can be invoked by Principal Agents. The ACP protocol defines message formats for agent-to-agent
    communication, and the AGP protocol standardizes gateway interactions.
    """
    
    schema = {
        "components": {
            "type": "array",
            "description": "List of architecture components mentioned"
        },
        "protocols": {
            "type": "array",
            "description": "List of protocols mentioned"
        },
        "year_proposed": {
            "type": "integer",
            "description": "Year the architecture was proposed"
        },
        "proposing_organization": {
            "type": "string",
            "description": "Organization that proposed the architecture"
        }
    }
    
    print(f"Text for extraction: {text}")
    print(f"Schema: {json.dumps(schema, indent=2)}\n")
    
    result = agent.invoke("extraction", {"text": text, "schema": schema})
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("Extracted Information:")
    print(result["content"])
    print(f"\nTokens: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")


def run_demo():
    """Run the Anthropic agent demo."""
    print("=== Anthropic Claude Agent Demo (DAWN Framework) ===\n")
    
    # Check environment configuration
    if not check_environment():
        sys.exit(1)
    
    # Display configuration
    config = check_configuration()
    print("Configuration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # Create an Anthropic agent
    agent = create_anthropic_agent(
        name="DAWN Claude Agent",
        description="Claude-powered agent implementing DAWN architecture",
        model="claude-3-sonnet-20240229",  # You can change this to other Claude models
        temperature=0.3,
        max_tokens=1024,
        system_prompt="You are a helpful AI assistant in the DAWN framework. Provide concise, accurate responses."
    )
    
    # Check agent health
    print("\nChecking agent health...")
    health_status = agent.health_check()
    print(f"Agent health: {'✅ Healthy' if health_status else '❌ Unhealthy'}")
    
    if not health_status:
        print("Agent health check failed. Please check your Anthropic API configuration.")
        sys.exit(1)
    
    # Run all demos
    try:
        text_generation_demo(agent)
        time.sleep(1)  # Brief pause between API calls
        
        chat_demo(agent)
        time.sleep(1)
        
        summarization_demo(agent)
        time.sleep(1)
        
        classification_demo(agent)
        time.sleep(1)
        
        extraction_demo(agent)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during the demo: {str(e)}")
    
    print("\nDemo completed.")


if __name__ == "__main__":
    run_demo()