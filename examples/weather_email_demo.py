"""
Demonstration of a simple DAWN workflow for the weather email example.

This module implements a mock version of the weather email workflow
described in the demo plan, using the interfaces defined in the project.
"""
import sys
import os
import uuid
import json
from typing import Dict, List, Optional, Any, Union

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interfaces import IAgent, IPrincipalAgent, IGatewayAgent, IMessage, ITask


class Message:
    """Implementation of the IMessage interface."""
    
    def __init__(
        self, 
        sender_id: str, 
        recipient_id: str, 
        content: Any, 
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self._id = str(uuid.uuid4())
        self._sender_id = sender_id
        self._recipient_id = recipient_id
        self._content = content
        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._metadata = metadata or {}
        
    def get_id(self) -> str:
        return self._id
        
    def get_sender_id(self) -> str:
        return self._sender_id
        
    def get_recipient_id(self) -> str:
        return self._recipient_id
        
    def get_content(self) -> Any:
        return self._content
        
    def get_conversation_id(self) -> str:
        return self._conversation_id
        
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "sender_id": self._sender_id,
            "recipient_id": self._recipient_id,
            "content": self._content,
            "conversation_id": self._conversation_id,
            "metadata": self._metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        instance = cls(
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            content=data["content"],
            conversation_id=data.get("conversation_id"),
            metadata=data.get("metadata", {})
        )
        instance._id = data["id"]
        return instance


class Task:
    """Implementation of the ITask interface."""
    
    def __init__(
        self,
        description: str,
        inputs: Dict[str, Any],
        output_schema: Dict[str, Any],
        required_capabilities: List[Dict[str, Any]],
        dependencies: Optional[List[str]] = None
    ):
        self._id = str(uuid.uuid4())
        self._description = description
        self._inputs = inputs
        self._output_schema = output_schema
        self._required_capabilities = required_capabilities
        self._dependencies = dependencies or []
        self._status = "pending"
        self._result = None
        
    def get_id(self) -> str:
        return self._id
        
    def get_description(self) -> str:
        return self._description
        
    def get_inputs(self) -> Dict[str, Any]:
        return self._inputs
        
    def get_output_schema(self) -> Dict[str, Any]:
        return self._output_schema
        
    def get_required_capabilities(self) -> List[Dict[str, Any]]:
        return self._required_capabilities
        
    def get_dependencies(self) -> List[str]:
        return self._dependencies
        
    def get_status(self) -> str:
        return self._status
        
    def set_status(self, status: str) -> None:
        self._status = status
        
    def set_result(self, result: Dict[str, Any]) -> None:
        self._result = result
        self._status = "completed"
        
    def get_result(self) -> Optional[Dict[str, Any]]:
        return self._result


class BaseAgent:
    """Base implementation of the IAgent interface."""
    
    def __init__(self, agent_id: str, name: str, description: str, capabilities: List[Dict[str, Any]]):
        self._id = agent_id
        self._name = name
        self._description = description
        self._capabilities = capabilities
        
    def get_info(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return self._capabilities
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement invoke method")
        
    def health_check(self) -> bool:
        return True


class MockWeatherAgent(BaseAgent):
    """Mock implementation of a weather data agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="weather-agent-123",
            name="Weather Data Agent",
            description="Provides weather data for locations",
            capabilities=[
                {
                    "id": "weather-cap-1",
                    "type": "weather_data",
                    "name": "Get Weather",
                    "description": "Retrieves current weather data for a location"
                }
            ]
        )
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"[WeatherAgent] Retrieving weather data for {inputs.get('location', 'Unknown')}")
        # Mock weather data
        return {
            "temperature": 72,
            "conditions": "Sunny",
            "humidity": 65
        }


class MockTextAgent(BaseAgent):
    """Mock implementation of a text generation agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="text-agent-456",
            name="Text Generation Agent",
            description="Generates text based on prompts and data",
            capabilities=[
                {
                    "id": "text-cap-1",
                    "type": "text_generation",
                    "name": "Generate Text",
                    "description": "Generates text based on a prompt and data"
                }
            ]
        )
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"[TextAgent] Generating text for prompt: {inputs.get('prompt', 'Unknown')}")
        # Mock text generation
        location = inputs.get('location', 'San Francisco')
        data = inputs.get('data', {})
        
        return {
            "text": f"Today in {location}: It's a {data.get('conditions', 'sunny')} day with a temperature of {data.get('temperature', 72)}°F and {data.get('humidity', 65)}% humidity."
        }


class MockEmailAgent(BaseAgent):
    """Mock implementation of an email sending agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="email-agent-789",
            name="Email Agent",
            description="Sends email messages",
            capabilities=[
                {
                    "id": "email-cap-1",
                    "type": "email_sending",
                    "name": "Send Email",
                    "description": "Sends an email to a recipient"
                }
            ]
        )
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"[EmailAgent] Sending email to {inputs.get('to', 'Unknown')}")
        print(f"  Subject: {inputs.get('subject', 'No Subject')}")
        print(f"  Body: {inputs.get('body', 'No Body')}")
        
        # Mock email sending
        return {
            "success": True,
            "message_id": "abc123"
        }


class GatewayAgent(BaseAgent, IGatewayAgent):
    """Implementation of the Gateway Agent interface."""
    
    def __init__(self):
        super().__init__(
            agent_id="gateway-agent-001",
            name="Gateway Agent",
            description="Manages agent registry and discovery",
            capabilities=[
                {
                    "id": "gateway-cap-1",
                    "type": "agent_registry",
                    "name": "Agent Registry",
                    "description": "Manages a registry of available agents"
                }
            ]
        )
        self._agents = {}
        
    def register_agent(self, agent_info: Dict[str, Any]) -> str:
        agent_id = agent_info.get("id", str(uuid.uuid4()))
        self._agents[agent_id] = agent_info
        print(f"[GatewayAgent] Registered agent: {agent_info.get('name')} ({agent_id})")
        return agent_id
        
    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            agent_name = self._agents[agent_id].get("name", "Unknown Agent")
            del self._agents[agent_id]
            print(f"[GatewayAgent] Unregistered agent: {agent_name} ({agent_id})")
            return True
        return False
        
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self._agents.get(agent_id)
        
    def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not filters:
            return list(self._agents.values())
            
        result = []
        for agent in self._agents.values():
            match = True
            for key, value in filters.items():
                if key not in agent or agent[key] != value:
                    match = False
                    break
            if match:
                result.append(agent)
                
        return result
        
    def find_agents_by_capability(self, capability_type: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        result = []
        for agent in self._agents.values():
            capabilities = agent.get("capabilities", [])
            for capability in capabilities:
                if capability.get("type") == capability_type:
                    # Check parameters if provided
                    if parameters:
                        params_match = True
                        for key, value in parameters.items():
                            if key not in capability or capability[key] != value:
                                params_match = False
                                break
                        if not params_match:
                            continue
                    result.append(agent)
                    break
                    
        print(f"[GatewayAgent] Found {len(result)} agents with capability: {capability_type}")
        return result
        
    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._agents:
            return {"valid": False, "reason": "Agent not found"}
            
        # In a real implementation, we would check if the agent is accessible
        return {"valid": True}


class PrincipalAgent(BaseAgent, IPrincipalAgent):
    """Implementation of the Principal Agent interface."""
    
    def __init__(self, gateway_agent: GatewayAgent):
        super().__init__(
            agent_id="principal-agent-001",
            name="Principal Agent",
            description="Orchestrates complex tasks across multiple agents",
            capabilities=[
                {
                    "id": "principal-cap-1",
                    "type": "task_orchestration",
                    "name": "Task Orchestration",
                    "description": "Breaks down and orchestrates complex tasks"
                }
            ]
        )
        self._gateway = gateway_agent
        
    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        task_description = task.get("description", "")
        print(f"[PrincipalAgent] Decomposing task: {task_description}")
        
        # For the weather email demo, we'll hard-code the decomposition
        if "weather" in task_description.lower() and "email" in task_description.lower():
            location = "San Francisco"  # Extract from task or use default
            if "location" in task:
                location = task["location"]
                
            # Create subtasks with actual UUIDs that we can track
            weather_task_id = str(uuid.uuid4())
            text_task_id = str(uuid.uuid4())
            email_task_id = str(uuid.uuid4())
            
            subtasks = [
                {
                    "id": weather_task_id,
                    "description": f"Get weather data for {location}",
                    "inputs": {"location": location},
                    "output_schema": {"temperature": "number", "conditions": "string", "humidity": "number"},
                    "required_capabilities": [{"type": "weather_data"}],
                    "dependencies": []
                },
                {
                    "id": text_task_id,
                    "description": "Generate weather summary text",
                    "inputs": {"prompt": "Write a short weather summary", "location": location, "data": f"@{{{weather_task_id}}}"},
                    "output_schema": {"text": "string"},
                    "required_capabilities": [{"type": "text_generation"}],
                    "dependencies": [weather_task_id]
                },
                {
                    "id": email_task_id,
                    "description": "Send email with weather summary",
                    "inputs": {"to": task.get("email", "user@example.com"), "subject": f"Weather Summary for {location}", "body": f"@{{{text_task_id}.text}}"},
                    "output_schema": {"success": "boolean", "message_id": "string"},
                    "required_capabilities": [{"type": "email_sending"}],
                    "dependencies": [text_task_id]
                }
            ]
            
            # No need for these updates anymore as we're setting them directly above
            
            return subtasks
            
        return []
        
    def discover_agents(self, capability_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        print(f"[PrincipalAgent] Discovering agents for {len(capability_requirements)} capability requirements")
        
        result = {}
        for req in capability_requirements:
            capability_type = req.get("type")
            agents = self._gateway.find_agents_by_capability(capability_type)
            result[capability_type] = [agent["id"] for agent in agents]
            
        return result
        
    def create_execution_plan(self, subtasks: List[Dict[str, Any]], available_agents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        print(f"[PrincipalAgent] Creating execution plan for {len(subtasks)} subtasks")
        
        execution_plan = []
        for subtask in subtasks:
            # Find an agent for each required capability
            agent_assignments = {}
            for capability in subtask["required_capabilities"]:
                capability_type = capability["type"]
                if capability_type in available_agents and available_agents[capability_type]:
                    # Just use the first available agent for this demo
                    agent_id = available_agents[capability_type][0]
                    agent_assignments[capability_type] = agent_id
                    
            execution_step = {
                "task_id": subtask["id"],
                "description": subtask["description"],
                "agent_assignments": agent_assignments,
                "inputs": subtask["inputs"],
                "dependencies": subtask["dependencies"]
            }
            
            execution_plan.append(execution_step)
            
        return execution_plan
        
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"[PrincipalAgent] Executing plan with {len(execution_plan)} steps")
        
        # Initialize results storage
        results = {}
        
        # Execute steps in order (respecting dependencies)
        for step in execution_plan:
            step_id = step["task_id"]
            description = step["description"]
            agent_assignments = step["agent_assignments"]
            inputs = step["inputs"]
            dependencies = step["dependencies"]
            
            print(f"\n[PrincipalAgent] Executing step: {description}")
            
            # Check if dependencies are satisfied
            dependencies_satisfied = True
            for dep_id in dependencies:
                if dep_id not in results:
                    dependencies_satisfied = False
                    print(f"[PrincipalAgent] Dependency {dep_id} not satisfied")
                    break
                    
            if not dependencies_satisfied:
                return {"status": "error", "message": f"Dependencies not satisfied for step {step_id}"}
                
            # Process input references
            processed_inputs = self._process_input_references(inputs, results)
            
            # For simplicity, we'll just use the first capability type and agent
            if len(agent_assignments) == 0:
                return {"status": "error", "message": f"No agents assigned for step {step_id}"}
                
            capability_type = list(agent_assignments.keys())[0]
            agent_id = agent_assignments[capability_type]
            
            # Get the agent
            agent_info = self._gateway.get_agent(agent_id)
            if not agent_info:
                return {"status": "error", "message": f"Agent {agent_id} not found"}
                
            print(f"[PrincipalAgent] Invoking agent: {agent_info['name']} ({agent_id})")
            
            # Find the actual agent instance (in a real implementation, this would be an API call)
            agent_instance = None
            if agent_id == "weather-agent-123":
                agent_instance = MockWeatherAgent()
            elif agent_id == "text-agent-456":
                agent_instance = MockTextAgent()
            elif agent_id == "email-agent-789":
                agent_instance = MockEmailAgent()
                
            if not agent_instance:
                return {"status": "error", "message": f"Could not find actual agent for {agent_id}"}
                
            # Find capability ID
            capability_id = None
            for capability in agent_info["capabilities"]:
                if capability["type"] == capability_type:
                    capability_id = capability["id"]
                    break
                    
            if not capability_id:
                return {"status": "error", "message": f"Could not find capability ID for {capability_type}"}
                
            # Invoke the agent
            try:
                result = agent_instance.invoke(capability_id, processed_inputs)
                results[step_id] = result
                print(f"[PrincipalAgent] Step {step_id} completed successfully")
            except Exception as e:
                print(f"[PrincipalAgent] Error executing step {step_id}: {str(e)}")
                return {"status": "error", "message": f"Error executing step {step_id}: {str(e)}"}
                
        # Return final result
        return {
            "status": "completed",
            "summary": "Task completed successfully",
            "details": results
        }
        
    def _process_input_references(self, inputs: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Process input references to task results."""
        processed = {}
        
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("@{") and value.endswith("}"):
                # Extract reference
                ref = value[2:-1]
                parts = ref.split(".")
                
                if len(parts) >= 1:  # Changed from 2 to 1
                    task_id = parts[0]
                    if task_id in results:
                        # Navigate through the result structure
                        current = results[task_id]
                        for i in range(1, len(parts)):
                            if isinstance(current, dict) and parts[i] in current:
                                current = current[parts[i]]
                            else:
                                # Debug print
                                print(f"[PrincipalAgent] Warning: Could not resolve reference path {parts[i]} in {current}")
                                current = {}  # Changed from None to empty dict
                                break
                        processed[key] = current
                    else:
                        # Debug print
                        print(f"[PrincipalAgent] Warning: Referenced task ID {task_id} not found in results")
                        processed[key] = {}  # Changed from None to empty dict
            else:
                processed[key] = value
                
        return processed
        
    def handle_error(self, error: Dict[str, Any], execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"[PrincipalAgent] Handling error: {error.get('message', 'Unknown error')}")
        
        # In a real implementation, we might retry or try alternative agents
        return {
            "status": "error",
            "message": error.get("message", "Unknown error"),
            "handled": False
        }


def run_demo():
    """Run the weather email demo scenario."""
    print("=== Starting DAWN Weather Email Demo ===\n")
    
    # Create the gateway agent
    gateway = GatewayAgent()
    
    # Create specialized agents
    weather_agent = MockWeatherAgent()
    text_agent = MockTextAgent()
    email_agent = MockEmailAgent()
    
    # Register the agents
    gateway.register_agent(weather_agent.get_info())
    gateway.register_agent(text_agent.get_info())
    gateway.register_agent(email_agent.get_info())
    
    # Create the principal agent
    principal = PrincipalAgent(gateway)
    
    # Submit a task to the principal agent
    task = {
        "description": "Get the current weather for San Francisco and send me an email summary",
        "email": "user@example.com",
        "location": "San Francisco"
    }
    
    print("=== Task Decomposition ===\n")
    subtasks = principal.decompose_task(task)
    
    print("\n=== Agent Discovery ===\n")
    capability_requirements = []
    for subtask in subtasks:
        capability_requirements.extend(subtask["required_capabilities"])
    available_agents = principal.discover_agents(capability_requirements)
    
    print("\n=== Execution Planning ===\n")
    execution_plan = principal.create_execution_plan(subtasks, available_agents)
    
    print("\n=== Plan Execution ===\n")
    result = principal.execute_plan(execution_plan)
    
    print("\n=== Final Result ===\n")
    print(json.dumps(result, indent=2))
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    run_demo()