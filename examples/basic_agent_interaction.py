"""
Example demonstrating basic agent interaction using the DAWN/AGNTCY implementation.
"""
import sys
import os

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import Agent, Message
from src.registry import Registry

# Create a registry
registry = Registry("Example Registry")

# Create some agents
assistant_agent = Agent(
    name="Assistant",
    description="A helpful assistant agent"
)
assistant_agent.add_capability({
    "type": "text_generation",
    "description": "Can generate text responses to queries"
})

tool_agent = Agent(
    name="Calculator",
    description="An agent that can perform mathematical calculations"
)
tool_agent.add_capability({
    "type": "calculation",
    "description": "Can perform basic arithmetic operations"
})

# Register the agents
registry.register(assistant_agent)
registry.register(tool_agent)

# List all registered agents
print("All registered agents:")
for agent_info in registry.list_agents():
    print(f"- {agent_info['name']} ({agent_info['id']}): {agent_info['description']}")

# Find agents with specific capabilities
calculation_agents = registry.find_agents_by_capability("calculation")
print("\nAgents that can perform calculations:")
for agent in calculation_agents:
    print(f"- {agent.name} ({agent.id})")

# Simulate a message exchange
message = Message(
    sender_id=assistant_agent.id,
    recipient_id=tool_agent.id,
    content="Calculate 2 + 2",
    message_type="request",
    metadata={"operation": "addition"}
)

print("\nMessage sent:")
print(f"From: {message.sender_id} (Assistant)")
print(f"To: {message.recipient_id} (Calculator)")
print(f"Content: {message.content}")
print(f"Type: {message.message_type}")
print(f"Metadata: {message.metadata}")

# Simulate a response
response = Message(
    sender_id=tool_agent.id,
    recipient_id=assistant_agent.id,
    content="4",
    message_type="response",
    metadata={"operation": "addition", "status": "success"}
)

print("\nResponse received:")
print(f"From: {response.sender_id} (Calculator)")
print(f"To: {response.recipient_id} (Assistant)")
print(f"Content: {response.content}")
print(f"Type: {response.message_type}")
print(f"Metadata: {response.metadata}")