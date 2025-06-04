"""
Agent Gateway Protocol (AGP) Demo implementing DAWN architecture.

This demo showcases a gateway agent using the AGP protocol for agent registration
and discovery, which is a key component of the DAWN architecture.
"""
import sys
import os
import time
import threading
import json
from typing import Dict, Any

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agp_gateway import AgpGatewayAgent
from src.anthropic_agent import create_anthropic_agent
# Skip importing Gemini to avoid protobuf conflicts
# from src.gemini_agent import create_gemini_agent
from src.config import APIConfig, check_configuration


def print_divider(title=None):
    """Display a divider with an optional title."""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")


def start_gateway_in_thread(gateway):
    """Start the gateway server in a background thread."""
    gateway_thread = threading.Thread(target=gateway.serve_forever)
    gateway_thread.daemon = True
    gateway_thread.start()
    return gateway_thread


def create_mock_agent(name, description, capabilities):
    """Create a mock agent for demonstration purposes."""
    return {
        "name": name,
        "description": description,
        "capabilities": capabilities,
        "provider": "mock",
        "metadata": {
            "type": "demo"
        }
    }


def register_agents(gateway):
    """Register several agents with the gateway."""
    print_divider("Registering Agents")
    
    # Create and register a few agents
    agents = []
    
    # Mock text generation agent
    text_agent = create_mock_agent(
        name="Mock Text Generator",
        description="Generates text based on prompts",
        capabilities=[
            {
                "id": "text-generation",
                "type": "text_generation",
                "name": "Text Generation",
                "description": "Generate text based on a prompt"
            }
        ]
    )
    agent_id = gateway.register_agent(text_agent)
    print(f"Registered Text Generator Agent: {agent_id}")
    agents.append(agent_id)
    
    # Mock image recognition agent
    image_agent = create_mock_agent(
        name="Mock Image Recognizer",
        description="Recognizes objects in images",
        capabilities=[
            {
                "id": "image-recognition",
                "type": "image_recognition",
                "name": "Image Recognition",
                "description": "Recognize objects in images"
            }
        ]
    )
    agent_id = gateway.register_agent(image_agent)
    print(f"Registered Image Recognition Agent: {agent_id}")
    agents.append(agent_id)
    
    # Mock multi-capability agent
    multi_agent = create_mock_agent(
        name="Mock Multi-capability Agent",
        description="Agent with multiple capabilities",
        capabilities=[
            {
                "id": "text-generation",
                "type": "text_generation",
                "name": "Text Generation",
                "description": "Generate text based on a prompt"
            },
            {
                "id": "summarization",
                "type": "summarization",
                "name": "Summarization",
                "description": "Summarize long text"
            }
        ]
    )
    agent_id = gateway.register_agent(multi_agent)
    print(f"Registered Multi-capability Agent: {agent_id}")
    agents.append(agent_id)
    
    # Register real agents if APIs are configured
    if APIConfig.is_anthropic_configured():
        try:
            anthropic_agent = create_anthropic_agent()
            agent_info = anthropic_agent.get_info()
            agent_id = gateway.register_agent(agent_info)
            print(f"Registered Anthropic Agent: {agent_id}")
            agents.append(agent_id)
        except Exception as e:
            print(f"Error registering Anthropic agent: {str(e)}")
    
    # Skip Gemini registration to avoid protobuf conflicts
    if False and APIConfig.is_gemini_configured():
        try:
            # Commented out to avoid protobuf conflicts
            # gemini_agent = create_gemini_agent()
            # agent_info = gemini_agent.get_info()
            # agent_id = gateway.register_agent(agent_info)
            # print(f"Registered Gemini Agent: {agent_id}")
            # agents.append(agent_id)
            pass
        except Exception as e:
            print(f"Error registering Gemini agent: {str(e)}")
    
    return agents


def query_gateway(gateway, agent_ids):
    """Perform various queries on the gateway."""
    print_divider("Querying Gateway")
    
    # List all agents
    all_agents = gateway.list_agents()
    print(f"All agents ({len(all_agents)}):")
    for i, agent in enumerate(all_agents, 1):
        print(f"{i}. {agent['name']} ({agent['id']})")
    print()
    
    # Find agents by capability type
    text_agents = gateway.find_agents_by_capability("text_generation")
    print(f"Agents with text generation capability ({len(text_agents)}):")
    for i, agent in enumerate(text_agents, 1):
        print(f"{i}. {agent['name']} ({agent['id']})")
    print()
    
    image_agents = gateway.find_agents_by_capability("image_recognition")
    print(f"Agents with image recognition capability ({len(image_agents)}):")
    for i, agent in enumerate(image_agents, 1):
        print(f"{i}. {agent['name']} ({agent['id']})")
    print()
    
    # Get specific agent
    if agent_ids:
        agent_id = agent_ids[0]
        agent = gateway.get_agent(agent_id)
        print(f"Agent details for {agent_id}:")
        print(f"Name: {agent['name']}")
        print(f"Description: {agent['description']}")
        print(f"Capabilities:")
        for capability in agent['capabilities']:
            print(f"- {capability['name']} ({capability['id']}): {capability['description']}")
        print()
    
    # Validate an agent
    if agent_ids:
        agent_id = agent_ids[0]
        validation = gateway.validate_agent(agent_id)
        print(f"Validation for {agent_id}: {'Valid' if validation['valid'] else 'Invalid'}")
        if not validation['valid']:
            print(f"Reason: {validation.get('reason')}")
        print()


def unregister_agents(gateway, agent_ids):
    """Unregister agents from the gateway."""
    print_divider("Unregistering Agents")
    
    for agent_id in agent_ids:
        success = gateway.unregister_agent(agent_id)
        if success:
            print(f"Successfully unregistered agent: {agent_id}")
        else:
            print(f"Failed to unregister agent: {agent_id}")
    
    # Verify agents are unregistered
    remaining_agents = gateway.list_agents()
    print(f"\nRemaining agents ({len(remaining_agents)}):")
    for i, agent in enumerate(remaining_agents, 1):
        print(f"{i}. {agent['name']} ({agent['id']})")


def run_demo():
    """Run the AGP gateway demo."""
    print("=== Agent Gateway Protocol (AGP) Demo ===\n")
    
    # Create a gateway agent
    gateway = AgpGatewayAgent(
        name="DAWN Demo Gateway",
        description="Demonstration gateway for the DAWN architecture",
        host="localhost",
        port=50051
    )
    
    print(f"Created gateway: {gateway._name} ({gateway._id})")
    print(f"Gateway will be available at {gateway._host}:{gateway._port}")
    
    # Start the gateway server in a background thread
    try:
        # For demo purposes, we'll just interact with the gateway directly
        # without actually starting the gRPC server
        # gateway_thread = start_gateway_in_thread(gateway)
        
        # Register agents
        agent_ids = register_agents(gateway)
        
        # Wait a moment
        time.sleep(1)
        
        # Query the gateway
        query_gateway(gateway, agent_ids)
        
        # Wait a moment
        time.sleep(1)
        
        # Unregister agents
        unregister_agents(gateway, agent_ids)
        
        print("\nDemo completed.")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        # If we started the gateway server, stop it
        # gateway.stop_server()
        pass


if __name__ == "__main__":
    run_demo()