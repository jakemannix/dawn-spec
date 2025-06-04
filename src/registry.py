"""
Implementation of agent registry for AGNTCY framework.

This module implements a registry service that aligns with the agent discovery
and management concepts from the AGNTCY framework.
"""
from typing import Dict, List, Optional, Any, Callable
import uuid
from .agent import Agent, Capability


class Registry:
    """
    Registry service for AGNTCY-based agents.
    
    The registry provides agent discovery and management functions aligned with
    the AGNTCY framework for agent interoperability.
    """
    
    def __init__(self, name: str = "Default Registry", description: Optional[str] = None):
        """
        Initialize a new Registry.
        
        Args:
            name: Human-readable name for the registry
            description: Optional description of this registry
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.agents: Dict[str, Agent] = {}
        self.hooks: Dict[str, List[Callable]] = {
            "register": [],
            "unregister": [],
            "update": []
        }
        
    def register(self, agent: Agent) -> str:
        """
        Register an agent with this registry.
        
        Args:
            agent: The agent to register
            
        Returns:
            The ID of the registered agent
        """
        self.agents[agent.id] = agent
        
        # Execute any registered hooks
        for hook in self.hooks["register"]:
            hook(agent)
            
        return agent.id
        
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent from this registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            
            # Execute any registered hooks
            for hook in self.hooks["unregister"]:
                hook(agent)
                
            return True
        return False
            
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by its ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The agent if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.
        
        Returns:
            List of dictionaries representing registered agents
        """
        return [agent.to_dict() for agent in self.agents.values()]
    
    def find_agents_by_capability(self, capability_type: str) -> List[Agent]:
        """
        Find agents that have a specific capability type.
        
        Args:
            capability_type: The type of capability to search for
            
        Returns:
            List of agents with the specified capability type
        """
        matching_agents = []
        for agent in self.agents.values():
            for capability in agent.capabilities:
                if capability.type == capability_type:
                    matching_agents.append(agent)
                    break
        return matching_agents
    
    def find_agents_by_criteria(self, criteria: Dict[str, Any]) -> List[Agent]:
        """
        Find agents matching specific criteria.
        
        Args:
            criteria: Dictionary of criteria to match against agent properties
            
        Returns:
            List of agents matching all criteria
        """
        matching_agents = []
        
        for agent in self.agents.values():
            agent_dict = agent.to_dict()
            matches = True
            
            for key, value in criteria.items():
                # Handle nested criteria with dot notation (e.g., "metadata.domain")
                if "." in key:
                    parts = key.split(".")
                    current = agent_dict
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            matches = False
                            break
                    if matches and current != value:
                        matches = False
                # Handle simple criteria
                elif key in agent_dict and agent_dict[key] != value:
                    matches = False
                    
            if matches:
                matching_agents.append(agent)
                
        return matching_agents
    
    def add_hook(self, event: str, hook: Callable) -> None:
        """
        Add a hook to be called on specific registry events.
        
        Args:
            event: The event to hook into ("register", "unregister", "update")
            hook: The function to call when the event occurs
        """
        if event in self.hooks:
            self.hooks[event].append(hook)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the registry to a dictionary representation.
        
        Returns:
            Dictionary representing the registry
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_count": len(self.agents)
        }