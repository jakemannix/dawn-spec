"""
Unit tests for the core interfaces defined in the DAWN specification.

These tests verify that the interfaces are correctly defined and that
implementations properly follow the required contracts.
"""
import pytest
import uuid
from typing import Dict, List, Optional, Any, Union

from src.interfaces import IAgent, IPrincipalAgent, IGatewayAgent, IMessage, ITask


class MockAgent(IAgent):
    """Mock implementation of IAgent for testing."""
    
    def __init__(self, agent_id: str = None, capabilities: List[Dict[str, Any]] = None):
        self._id = agent_id or f"mock-agent-{str(uuid.uuid4())[:8]}"
        self._name = "Mock Agent"
        self._description = "Mock agent for testing"
        self._capabilities = capabilities or [
            {
                "id": "test-capability",
                "type": "test",
                "name": "Test Capability",
                "description": "A test capability"
            }
        ]
        self._invoked = False
        self._capability_id = None
        self._inputs = None
        self._config = None
    
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return self._capabilities
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs and configuration."""
        self._invoked = True
        self._capability_id = capability_id
        self._inputs = inputs
        self._config = config
        
        if capability_id not in [c["id"] for c in self._capabilities]:
            return {"error": f"Unknown capability: {capability_id}"}
        
        return {"result": "mock_result", "capability_id": capability_id}
    
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


class MockPrincipalAgent(IPrincipalAgent):
    """Mock implementation of IPrincipalAgent for testing."""
    
    def __init__(self, agent_id: str = None, capabilities: List[Dict[str, Any]] = None):
        self._id = agent_id or f"mock-principal-{str(uuid.uuid4())[:8]}"
        self._name = "Mock Principal Agent"
        self._description = "Mock principal agent for testing"
        self._capabilities = capabilities or [
            {
                "id": "orchestration",
                "type": "orchestration",
                "name": "Task Orchestration",
                "description": "Orchestrate tasks across agents"
            }
        ]
    
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return self._capabilities
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs and configuration."""
        if capability_id not in [c["id"] for c in self._capabilities]:
            return {"error": f"Unknown capability: {capability_id}"}
        
        return {"result": "mock_result", "capability_id": capability_id}
    
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True
    
    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down a complex task into subtasks."""
        subtasks = [
            {
                "id": f"subtask-1-{str(uuid.uuid4())[:8]}",
                "description": "Mock subtask 1",
                "inputs": {"input1": "value1"},
                "required_capabilities": [{"type": "test"}],
                "dependencies": []
            },
            {
                "id": f"subtask-2-{str(uuid.uuid4())[:8]}",
                "description": "Mock subtask 2",
                "inputs": {"input2": "value2"},
                "required_capabilities": [{"type": "test"}],
                "dependencies": []
            }
        ]
        return subtasks
    
    def discover_agents(self, capability_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find agents with capabilities matching requirements."""
        results = {}
        for req in capability_requirements:
            req_type = req.get("type", "")
            if req_type:
                results[req_type] = [f"mock-agent-{req_type}"]
        return results
    
    def create_execution_plan(self, subtasks: List[Dict[str, Any]], available_agents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Create a plan for executing subtasks with selected agents."""
        plan = []
        for task in subtasks:
            agent_assignments = {}
            for cap in task.get("required_capabilities", []):
                cap_type = cap.get("type", "")
                if cap_type in available_agents and available_agents[cap_type]:
                    agent_assignments[cap_type] = available_agents[cap_type][0]
            
            plan.append({
                "task_id": task["id"],
                "description": task["description"],
                "agent_assignments": agent_assignments,
                "inputs": task["inputs"],
                "dependencies": task.get("dependencies", [])
            })
        return plan
    
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the plan and return aggregated results."""
        return {
            "status": "success",
            "results": {
                task["task_id"]: {"status": "completed"} for task in execution_plan
            }
        }
    
    def handle_error(self, error: Dict[str, Any], execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle errors during plan execution."""
        return {"status": "error_handled", "original_error": error}


class MockGatewayAgent(IGatewayAgent):
    """Mock implementation of IGatewayAgent for testing."""
    
    def __init__(self, agent_id: str = None, capabilities: List[Dict[str, Any]] = None):
        self._id = agent_id or f"mock-gateway-{str(uuid.uuid4())[:8]}"
        self._name = "Mock Gateway Agent"
        self._description = "Mock gateway agent for testing"
        self._capabilities = capabilities or [
            {
                "id": "registration",
                "type": "registration",
                "name": "Agent Registration",
                "description": "Register and manage agents"
            }
        ]
        self._registry = {}
    
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return self._capabilities
    
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs and configuration."""
        if capability_id not in [c["id"] for c in self._capabilities]:
            return {"error": f"Unknown capability: {capability_id}"}
        
        return {"result": "mock_result", "capability_id": capability_id}
    
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True
    
    def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """Register an agent in the registry."""
        agent_id = agent_info.get("id", str(uuid.uuid4()))
        self._registry[agent_id] = agent_info
        return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._registry:
            del self._registry[agent_id]
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        return self._registry.get(agent_id)
    
    def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List agents matching optional filters."""
        if not filters:
            return list(self._registry.values())
        
        results = []
        for agent in self._registry.values():
            match = True
            for key, value in filters.items():
                if key not in agent or agent[key] != value:
                    match = False
                    break
            if match:
                results.append(agent)
        
        return results
    
    def find_agents_by_capability(self, capability_type: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find agents with a specific capability type and parameters."""
        results = []
        for agent in self._registry.values():
            for capability in agent.get("capabilities", []):
                if capability.get("type") == capability_type:
                    # If parameters are specified, check if they match
                    if parameters:
                        capability_params = capability.get("parameters", {})
                        if all(capability_params.get(k) == v for k, v in parameters.items()):
                            results.append(agent)
                            break
                    else:
                        results.append(agent)
                        break
        
        return results
    
    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Validate an agent's capabilities and accessibility."""
        if agent_id not in self._registry:
            return {"valid": False, "reason": "Agent not found"}
        
        return {"valid": True, "agent_id": agent_id}


class MockMessage(IMessage):
    """Mock implementation of IMessage for testing."""
    
    def __init__(
        self,
        message_id: str = None,
        sender_id: str = None,
        recipient_id: str = None,
        content: Any = None,
        conversation_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        self._id = message_id or str(uuid.uuid4())
        self._sender_id = sender_id or f"sender-{str(uuid.uuid4())[:8]}"
        self._recipient_id = recipient_id or f"recipient-{str(uuid.uuid4())[:8]}"
        self._content = content or "Mock message content"
        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._metadata = metadata or {}
    
    def get_id(self) -> str:
        """Get unique message identifier."""
        return self._id
    
    def get_sender_id(self) -> str:
        """Get sender agent identifier."""
        return self._sender_id
    
    def get_recipient_id(self) -> str:
        """Get recipient agent identifier."""
        return self._recipient_id
    
    def get_content(self) -> Any:
        """Get message content."""
        return self._content
    
    def get_conversation_id(self) -> str:
        """Get conversation identifier for related messages."""
        return self._conversation_id
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get message metadata."""
        return self._metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self._id,
            "sender_id": self._sender_id,
            "recipient_id": self._recipient_id,
            "content": self._content,
            "conversation_id": self._conversation_id,
            "metadata": self._metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MockMessage':
        """Create message from dictionary representation."""
        return MockMessage(
            message_id=data.get("id"),
            sender_id=data.get("sender_id"),
            recipient_id=data.get("recipient_id"),
            content=data.get("content"),
            conversation_id=data.get("conversation_id"),
            metadata=data.get("metadata", {})
        )


class MockTask(ITask):
    """Mock implementation of ITask for testing."""
    
    def __init__(
        self,
        task_id: str = None,
        description: str = None,
        inputs: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        required_capabilities: List[Dict[str, Any]] = None,
        dependencies: List[str] = None
    ):
        self._id = task_id or str(uuid.uuid4())
        self._description = description or "Mock task description"
        self._inputs = inputs or {}
        self._output_schema = output_schema or {}
        self._required_capabilities = required_capabilities or []
        self._dependencies = dependencies or []
        self._status = "pending"
        self._result = None
    
    def get_id(self) -> str:
        """Get unique task identifier."""
        return self._id
    
    def get_description(self) -> str:
        """Get task description."""
        return self._description
    
    def get_inputs(self) -> Dict[str, Any]:
        """Get task inputs."""
        return self._inputs
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get schema for expected task output."""
        return self._output_schema
    
    def get_required_capabilities(self) -> List[Dict[str, Any]]:
        """Get capabilities required for task execution."""
        return self._required_capabilities
    
    def get_dependencies(self) -> List[str]:
        """Get IDs of dependent tasks."""
        return self._dependencies
    
    def get_status(self) -> str:
        """Get task status."""
        return self._status
    
    def set_status(self, status: str) -> None:
        """Update task status."""
        self._status = status
    
    def set_result(self, result: Dict[str, Any]) -> None:
        """Set task result."""
        self._result = result
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get task result."""
        return self._result


# Tests for IAgent interface
def test_agent_interface():
    """Test that the IAgent interface can be implemented and used correctly."""
    agent = MockAgent()
    
    # Test get_info
    info = agent.get_info()
    assert isinstance(info, dict)
    assert "id" in info
    assert "name" in info
    assert "description" in info
    assert "capabilities" in info
    assert isinstance(info["capabilities"], list)
    
    # Test get_capabilities
    capabilities = agent.get_capabilities()
    assert isinstance(capabilities, list)
    assert len(capabilities) > 0
    assert isinstance(capabilities[0], dict)
    assert "id" in capabilities[0]
    assert "type" in capabilities[0]
    assert "name" in capabilities[0]
    assert "description" in capabilities[0]
    
    # Test invoke
    result = agent.invoke("test-capability", {"test": "value"})
    assert isinstance(result, dict)
    assert "result" in result
    assert result["capability_id"] == "test-capability"
    assert agent._invoked is True
    assert agent._capability_id == "test-capability"
    assert agent._inputs == {"test": "value"}
    
    # Test invoke with invalid capability
    error_result = agent.invoke("invalid-capability", {})
    assert "error" in error_result
    
    # Test health_check
    health = agent.health_check()
    assert isinstance(health, bool)
    assert health is True


# Tests for IPrincipalAgent interface
def test_principal_agent_interface():
    """Test that the IPrincipalAgent interface can be implemented and used correctly."""
    agent = MockPrincipalAgent()
    
    # Test inheritance from IAgent
    assert isinstance(agent, IAgent)
    
    # Test the additional methods
    task = {"description": "Test task", "inputs": {"input": "value"}}
    subtasks = agent.decompose_task(task)
    assert isinstance(subtasks, list)
    assert len(subtasks) > 0
    assert "id" in subtasks[0]
    assert "description" in subtasks[0]
    assert "inputs" in subtasks[0]
    assert "required_capabilities" in subtasks[0]
    
    # Test discover_agents
    capability_requirements = [
        {"type": "test1"},
        {"type": "test2"}
    ]
    discovered = agent.discover_agents(capability_requirements)
    assert isinstance(discovered, dict)
    assert "test1" in discovered
    assert "test2" in discovered
    assert len(discovered["test1"]) > 0
    
    # Test create_execution_plan
    available_agents = {
        "test": ["mock-agent-1", "mock-agent-2"]
    }
    plan = agent.create_execution_plan(subtasks, available_agents)
    assert isinstance(plan, list)
    assert len(plan) == len(subtasks)
    assert "task_id" in plan[0]
    assert "agent_assignments" in plan[0]
    
    # Test execute_plan
    execution_result = agent.execute_plan(plan)
    assert isinstance(execution_result, dict)
    assert "status" in execution_result
    assert execution_result["status"] == "success"
    
    # Test handle_error
    error = {"error": "Test error"}
    error_handling = agent.handle_error(error, plan)
    assert isinstance(error_handling, dict)
    assert "status" in error_handling
    assert "original_error" in error_handling


# Tests for IGatewayAgent interface
def test_gateway_agent_interface():
    """Test that the IGatewayAgent interface can be implemented and used correctly."""
    agent = MockGatewayAgent()
    
    # Test inheritance from IAgent
    assert isinstance(agent, IAgent)
    
    # Test register_agent
    test_agent = {
        "id": "test-agent-1",
        "name": "Test Agent",
        "description": "Agent for testing",
        "capabilities": [
            {
                "id": "test-cap",
                "type": "test_capability",
                "name": "Test Capability",
                "description": "A capability for testing"
            }
        ]
    }
    agent_id = agent.register_agent(test_agent)
    assert agent_id == "test-agent-1"
    
    # Test get_agent
    retrieved = agent.get_agent("test-agent-1")
    assert retrieved == test_agent
    
    # Test list_agents
    agents = agent.list_agents()
    assert isinstance(agents, list)
    assert len(agents) == 1
    assert agents[0] == test_agent
    
    # Test list_agents with filter
    filtered_agents = agent.list_agents({"name": "Test Agent"})
    assert len(filtered_agents) == 1
    assert filtered_agents[0] == test_agent
    
    # Test find_agents_by_capability
    capability_agents = agent.find_agents_by_capability("test_capability")
    assert len(capability_agents) == 1
    assert capability_agents[0] == test_agent
    
    # Test validate_agent
    validation = agent.validate_agent("test-agent-1")
    assert validation["valid"] is True
    
    # Test unregister_agent
    unregistered = agent.unregister_agent("test-agent-1")
    assert unregistered is True
    assert agent.get_agent("test-agent-1") is None


# Tests for IMessage interface
def test_message_interface():
    """Test that the IMessage interface can be implemented and used correctly."""
    message = MockMessage(
        sender_id="test-sender",
        recipient_id="test-recipient",
        content="Test message content",
        conversation_id="test-conversation"
    )
    
    # Test getters
    assert message.get_sender_id() == "test-sender"
    assert message.get_recipient_id() == "test-recipient"
    assert message.get_content() == "Test message content"
    assert message.get_conversation_id() == "test-conversation"
    assert isinstance(message.get_metadata(), dict)
    
    # Test to_dict
    message_dict = message.to_dict()
    assert isinstance(message_dict, dict)
    assert message_dict["sender_id"] == "test-sender"
    assert message_dict["recipient_id"] == "test-recipient"
    assert message_dict["content"] == "Test message content"
    
    # Test from_dict
    new_message = MockMessage.from_dict(message_dict)
    assert new_message.get_sender_id() == message.get_sender_id()
    assert new_message.get_recipient_id() == message.get_recipient_id()
    assert new_message.get_content() == message.get_content()


# Tests for ITask interface
def test_task_interface():
    """Test that the ITask interface can be implemented and used correctly."""
    task = MockTask(
        task_id="test-task",
        description="Test task description",
        inputs={"input1": "value1"},
        required_capabilities=[{"type": "test_capability"}]
    )
    
    # Test getters
    assert task.get_id() == "test-task"
    assert task.get_description() == "Test task description"
    assert task.get_inputs() == {"input1": "value1"}
    assert len(task.get_required_capabilities()) == 1
    assert task.get_required_capabilities()[0]["type"] == "test_capability"
    assert task.get_dependencies() == []
    assert task.get_status() == "pending"
    assert task.get_result() is None
    
    # Test setters
    task.set_status("in_progress")
    assert task.get_status() == "in_progress"
    
    task.set_result({"output": "test_output"})
    assert task.get_result() == {"output": "test_output"}
    
    # Test status transitions
    task.set_status("completed")
    assert task.get_status() == "completed"