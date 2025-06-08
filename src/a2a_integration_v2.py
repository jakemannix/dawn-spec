"""
A2A SDK integration for DAWN agents - Version 2.

This module provides proper integration between DAWN agents and Google's A2A protocol,
implementing the three-layer model:
1. OASF Capabilities: Schema definitions and semantic tags
2. A2A Skills: Concrete functions exposed over the wire to other agents  
3. MCP Tools: LLM-facing interfaces (handled separately in mcp_integration.py)

This focuses on the A2A Skills layer that implements OASF Capability contracts.
"""
from typing import Any, Dict, List, Optional, Callable, Union
import asyncio
import logging
import uuid
import json
from dataclasses import dataclass, field

# For now, still using mock A2A components until real SDK is stable
# TODO: Replace with real A2A SDK when available

logger = logging.getLogger(__name__)


@dataclass
class A2ASkill:
    """
    Represents a concrete function that can be invoked over A2A protocol.
    
    Skills are externally callable functions you invoke over the wire.
    Each skill implements one or more OASF Capability contracts.
    """
    id: str
    name: str
    description: str
    tags: List[str]  # Semantic tags for categorization
    examples: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["application/json"])
    output_modes: List[str] = field(default_factory=lambda: ["application/json"])
    implementing_capabilities: List[str] = field(default_factory=list)  # OASF capability IDs this skill implements
    handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"skill_{uuid.uuid4().hex[:8]}"
    
    def to_a2a_skill_dict(self) -> Dict[str, Any]:
        """Convert to A2A AgentSkill format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
            "inputModes": self.input_modes,
            "outputModes": self.output_modes
        }


@dataclass  
class A2AMessage:
    """Represents an A2A message with parts."""
    message_id: str
    role: str  # 'user' or 'agent'
    parts: List[Dict[str, Any]]
    context_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"msg_{uuid.uuid4().hex[:8]}"
    
    def to_a2a_message_dict(self) -> Dict[str, Any]:
        """Convert to A2A Message format."""
        return {
            "messageId": self.message_id,
            "kind": "message",
            "role": self.role,
            "parts": self.parts,
            "contextId": self.context_id,
            "taskId": self.task_id,
            "metadata": self.metadata
        }


@dataclass
class A2ATask:
    """Represents an A2A Task."""
    id: str
    context_id: str
    status: Dict[str, Any]
    history: List[A2AMessage] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"task_{uuid.uuid4().hex[:8]}"
        if not self.context_id:
            self.context_id = f"ctx_{uuid.uuid4().hex[:8]}"
    
    def to_a2a_task_dict(self) -> Dict[str, Any]:
        """Convert to A2A Task format."""
        return {
            "id": self.id,
            "kind": "task", 
            "contextId": self.context_id,
            "status": self.status,
            "history": [msg.to_a2a_message_dict() for msg in self.history],
            "artifacts": self.artifacts,
            "metadata": self.metadata
        }


class MockA2ASkillServer:
    """Mock A2A server that exposes skills (not tools) to other agents."""
    
    def __init__(self, port: int = 8080, host: str = "localhost"):
        self.port = port
        self.host = host
        self.skills: Dict[str, A2ASkill] = {}
        self.tasks: Dict[str, A2ATask] = {}
        self.running = False
        
    def register_skill(self, skill: A2ASkill) -> None:
        """Register an A2A skill that other agents can invoke."""
        self.skills[skill.id] = skill
        logger.info(f"Registered A2A skill: {skill.id} ({skill.name})")
        
    async def start(self) -> None:
        """Start the A2A server."""
        self.running = True
        logger.info(f"Mock A2A skill server started on {self.host}:{self.port}")
        print(f"Mock A2A skill server started on {self.host}:{self.port}")
        
    async def stop(self) -> None:
        """Stop the A2A server."""
        self.running = False
        logger.info("Mock A2A skill server stopped")
        print("Mock A2A skill server stopped")
        
    def get_agent_card(self) -> Dict[str, Any]:
        """Return A2A AgentCard with skills."""
        return {
            "name": "DAWN Agent",
            "description": "DAWN agent exposing skills via A2A protocol",
            "version": "1.0.0",
            "url": f"http://{self.host}:{self.port}",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False
            },
            "skills": [skill.to_a2a_skill_dict() for skill in self.skills.values()],
            "defaultInputModes": ["application/json", "text/plain"],
            "defaultOutputModes": ["application/json", "text/plain"]
        }
        
    async def handle_message_send(self, message: A2AMessage) -> A2ATask:
        """Handle incoming message and create/update task."""
        # Create new task for this message
        task = A2ATask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            context_id=message.context_id or f"ctx_{uuid.uuid4().hex[:8]}",
            status={
                "state": "submitted",
                "timestamp": None
            }
        )
        
        # Add the incoming message to task history
        task.history.append(message)
        
        # Process the message parts to determine which skill to invoke
        response_parts = []
        
        for part in message.parts:
            if part.get("kind") == "text":
                text_content = part.get("text", "")
                # Simple routing based on text content
                skill_result = await self._route_to_skill(text_content, message)
                
                # Convert skill result to response part
                if skill_result:
                    response_parts.append({
                        "kind": "data",
                        "data": skill_result,
                        "metadata": {"skill_invoked": True}
                    })
                else:
                    response_parts.append({
                        "kind": "text", 
                        "text": f"Processed: {text_content}",
                        "metadata": {"echo": True}
                    })
            elif part.get("kind") == "data":
                # Handle structured data requests
                data_content = part.get("data", {})
                skill_result = await self._route_data_to_skill(data_content, message)
                response_parts.append({
                    "kind": "data",
                    "data": skill_result,
                    "metadata": {"data_processed": True}
                })
        
        # Create response message from agent
        response_message = A2AMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            role="agent",
            parts=response_parts,
            context_id=task.context_id,
            task_id=task.id
        )
        
        task.history.append(response_message)
        task.status = {
            "state": "completed",
            "timestamp": None
        }
        
        self.tasks[task.id] = task
        return task
        
    async def _route_to_skill(self, text: str, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Route text input to appropriate skill."""
        # Simple keyword-based routing for demo
        text_lower = text.lower()
        
        for skill in self.skills.values():
            if any(tag.lower() in text_lower for tag in skill.tags):
                if skill.handler:
                    try:
                        # Convert text to structured input for skill
                        skill_input = {"query": text, "context": message.metadata}
                        result = await skill.handler(skill_input) if asyncio.iscoroutinefunction(skill.handler) else skill.handler(skill_input)
                        return result
                    except Exception as e:
                        logger.error(f"Error invoking skill {skill.id}: {e}")
                        return {"error": str(e), "skill": skill.id}
        
        return None
        
    async def _route_data_to_skill(self, data: Dict[str, Any], message: A2AMessage) -> Dict[str, Any]:
        """Route structured data to appropriate skill."""
        # Look for skill_id in data
        skill_id = data.get("skill_id")
        if skill_id and skill_id in self.skills:
            skill = self.skills[skill_id]
            if skill.handler:
                try:
                    result = await skill.handler(data) if asyncio.iscoroutinefunction(skill.handler) else skill.handler(data)
                    return result
                except Exception as e:
                    logger.error(f"Error invoking skill {skill_id}: {e}")
                    return {"error": str(e), "skill": skill_id}
        
        # Default: echo the data
        return {"processed_data": data, "echo": True}


class DawnA2ASkillAdapter:
    """
    Adapter that converts DAWN agent capabilities into A2A skills.
    
    This bridges between:
    - OASF Capabilities (schema definitions) 
    - A2A Skills (concrete functions)
    """
    
    def __init__(self, dawn_agent, port: int = 8080, host: str = "localhost"):
        """
        Initialize the A2A skill adapter.
        
        Args:
            dawn_agent: DAWN agent with OASF capabilities
            port: Port for A2A server
            host: Host for A2A server
        """
        self.dawn_agent = dawn_agent
        self.a2a_server = MockA2ASkillServer(port=port, host=host)
        self.skills: Dict[str, A2ASkill] = {}
        self._create_skills_from_capabilities()
        
    def _create_skills_from_capabilities(self) -> None:
        """Convert DAWN capabilities into A2A skills."""
        capabilities = self.dawn_agent.get_capabilities()
        agent_info = self.dawn_agent.get_info()
        
        logger.info(f"Creating A2A skills from {len(capabilities)} DAWN capabilities for agent {agent_info.get('name', 'unknown')}")
        
        for capability in capabilities:
            skill = self._capability_to_skill(capability)
            self.skills[skill.id] = skill
            self.a2a_server.register_skill(skill)
            
    def _capability_to_skill(self, capability: Dict[str, Any]) -> A2ASkill:
        """Convert a DAWN capability into an A2A skill."""
        capability_id = capability.get('id', '')
        capability_type = capability.get('type', '')
        capability_name = capability.get('name', capability_id)
        capability_description = capability.get('description', f"Skill for {capability_name}")
        
        # Create skill ID by combining type and ID
        skill_id = f"{capability_type}.{capability_id}" if capability_type else capability_id
        
        # Extract semantic tags from capability type
        tags = [capability_type] if capability_type else ["general"]
        if '.' in capability_type:
            # Add hierarchical tags: "research.github" -> ["research", "github", "research.github"]
            parts = capability_type.split('.')
            for i in range(len(parts)):
                tags.append('.'.join(parts[:i+1]))
        
        # Create examples from capability metadata
        examples = capability.get('metadata', {}).get('examples', [])
        if not examples and capability_type:
            examples = [f"Invoke {capability_name}"]
        
        # Create skill handler that delegates to DAWN capability
        async def skill_handler(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Handle A2A skill invocation by delegating to DAWN capability."""
            try:
                logger.info(f"A2A skill {skill_id} invoking DAWN capability {capability_id}")
                
                # Transform A2A inputs to DAWN capability inputs
                dawn_inputs = self._transform_a2a_to_dawn_inputs(inputs, capability)
                
                # Invoke DAWN capability
                if hasattr(self.dawn_agent, 'invoke_async'):
                    result = await self.dawn_agent.invoke_async(capability_id, dawn_inputs)
                else:
                    result = self.dawn_agent.invoke(capability_id, dawn_inputs)
                
                # Transform DAWN result to A2A format
                a2a_result = self._transform_dawn_to_a2a_result(result, capability)
                
                logger.info(f"A2A skill {skill_id} completed successfully")
                return a2a_result
                
            except Exception as e:
                logger.error(f"Error in A2A skill {skill_id}: {e}")
                return {
                    "error": str(e),
                    "skill_id": skill_id,
                    "capability_id": capability_id
                }
        
        # Get agent info here to avoid undefined variable error
        agent_info = self.dawn_agent.get_info()
        
        skill = A2ASkill(
            id=skill_id,
            name=capability_name,
            description=capability_description,
            tags=tags,
            examples=examples,
            implementing_capabilities=[capability_id],
            handler=skill_handler,
            metadata={
                "source_capability": capability,
                "dawn_agent": agent_info.get('name', 'unknown')
            }
        )
        
        return skill
        
    def _transform_a2a_to_dawn_inputs(self, a2a_inputs: Dict[str, Any], capability: Dict[str, Any]) -> Dict[str, Any]:
        """Transform A2A message inputs to DAWN capability inputs."""
        # For now, simple pass-through
        # TODO: Add proper schema validation and transformation based on capability parameters
        return a2a_inputs
        
    def _transform_dawn_to_a2a_result(self, dawn_result: Dict[str, Any], capability: Dict[str, Any]) -> Dict[str, Any]:
        """Transform DAWN capability result to A2A format."""
        # Add A2A-specific metadata
        dawn_result["a2a_metadata"] = {
            "capability_id": capability.get('id'),
            "capability_type": capability.get('type'),
            "timestamp": None  # TODO: Add timestamp
        }
        return dawn_result
        
    async def start_a2a_server(self) -> MockA2ASkillServer:
        """Start the A2A server exposing skills."""
        await self.a2a_server.start()
        return self.a2a_server
        
    async def stop_a2a_server(self) -> None:
        """Stop the A2A server."""
        await self.a2a_server.stop()
        
    def get_skills(self) -> List[A2ASkill]:
        """Get all registered A2A skills."""
        return list(self.skills.values())
        
    def get_agent_card(self) -> Dict[str, Any]:
        """Get A2A AgentCard for this agent."""
        return self.a2a_server.get_agent_card() 