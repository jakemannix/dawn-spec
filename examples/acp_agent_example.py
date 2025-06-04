"""
Example demonstrating the use of the ACP server with AGNTCY-compatible agents.
"""
import sys
import os
import json
import requests
from typing import Dict, Any

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import Agent, Capability, Message
from src.registry import Registry
from src.acp_server import ACPServer


def run_example():
    """Run the ACP server example."""
    print("Initializing ACP server example...")
    
    # Create a registry
    registry = Registry("Example ACP Registry", "Registry for demonstrating ACP")
    
    # Create the ACP server
    server = ACPServer(registry)
    
    # Create some agents with capabilities
    assistant_agent = Agent(
        name="Assistant",
        description="A helpful assistant agent",
        provider="DAWN Example",
        version="1.0.0",
        metadata={"domain": "general", "environment": "example"}
    )
    
    # Add capabilities to the assistant
    text_gen_capability = Capability(
        capability_type="text_generation",
        name="Text Generation",
        description="Generates text responses to user queries",
        parameters={
            "prompt": {"type": "string", "description": "Input prompt"},
            "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"}
        },
        metadata={"model": "example-model"}
    )
    assistant_agent.add_capability(text_gen_capability)
    
    # Create a tool agent
    calculator_agent = Agent(
        name="Calculator",
        description="An agent that performs mathematical calculations",
        provider="DAWN Example",
        version="1.0.0",
        metadata={"domain": "mathematics", "environment": "example"}
    )
    
    # Add capabilities to the calculator
    calc_capability = Capability(
        capability_type="calculation",
        name="Calculation",
        description="Performs mathematical calculations",
        parameters={
            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
        },
        metadata={"operations": ["add", "subtract", "multiply", "divide"]}
    )
    calculator_agent.add_capability(calc_capability)
    
    # Register the agents
    assistant_id = registry.register(assistant_agent)
    calculator_id = registry.register(calculator_agent)
    
    print(f"Registered Assistant Agent with ID: {assistant_id}")
    print(f"Registered Calculator Agent with ID: {calculator_id}")
    
    # List all registered agents
    print("\nAll registered agents:")
    for idx, agent_info in enumerate(registry.list_agents(), 1):
        print(f"{idx}. {agent_info['name']} ({agent_info['id']}): {agent_info['description']}")
    
    # Find agents by capability
    calculation_agents = registry.find_agents_by_capability("calculation")
    print("\nAgents with calculation capability:")
    for agent in calculation_agents:
        print(f"- {agent.name} ({agent.id})")
    
    # Create a message between agents
    message = Message(
        sender_id=assistant_id,
        recipient_id=calculator_id,
        content={"expression": "2 + 2"},
        message_type="request",
        metadata={"operation": "calculation"}
    )
    
    print("\nMessage sent:")
    print(json.dumps(message.to_dict(), indent=2))
    
    # Simulate response
    response = Message(
        sender_id=calculator_id,
        recipient_id=assistant_id,
        content={"result": 4},
        conversation_id=message.conversation_id,
        message_type="response",
        metadata={"operation": "calculation", "status": "success"}
    )
    
    print("\nResponse received:")
    print(json.dumps(response.to_dict(), indent=2))
    
    # In a real application, we would start the server with:
    # server.run(host="0.0.0.0", port=8000)
    print("\nTo start the ACP server, run:")
    print("  from src.acp_server import ACPServer")
    print("  server = ACPServer()")
    print("  server.run(host='0.0.0.0', port=8000)")
    
    # and then we would make HTTP requests to it:
    print("\nExample HTTP request to register an agent:")
    print("""
    import requests
    
    response = requests.post(
        "http://localhost:8000/agents",
        json={
            "name": "Assistant Agent",
            "description": "A helpful assistant agent",
            "provider": "Example Corp",
            "version": "1.0.0",
            "capabilities": [
                {
                    "type": "text_generation",
                    "name": "Text Generator",
                    "description": "Generates text based on a prompt",
                    "parameters": {
                        "prompt": {"type": "string", "description": "Input prompt"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"}
                    },
                    "metadata": {"model": "example-model"}
                }
            ],
            "metadata": {"domain": "general"}
        }
    )
    
    agent_id = response.json()["agent_id"]
    print(f"Agent registered with ID: {agent_id}")
    """)


if __name__ == "__main__":
    run_example()