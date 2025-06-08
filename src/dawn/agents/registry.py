"""
Agent Registry

This module provides a registry system for discovering and creating agents.
"""

from typing import Dict, Type, Optional, List, Any
from dawn.core.agent import Agent
from dawn.utils.logging import get_logger


logger = get_logger(__name__)


class AgentRegistry:
    """
    Registry for agent implementations.
    
    This allows agents to be registered and created dynamically,
    supporting plugin-style agent discovery.
    """
    
    _agents: Dict[str, Type[Agent]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **metadata
    ):
        """
        Decorator to register an agent class.
        
        Args:
            name: Name to register the agent under
            description: Optional description of the agent
            tags: Optional list of tags for categorization
            **metadata: Additional metadata to store
            
        Returns:
            Decorator function
        """
        def decorator(agent_class: Type[Agent]):
            if name in cls._agents:
                logger.warning(f"Overwriting existing agent registration: {name}")
            
            cls._agents[name] = agent_class
            cls._metadata[name] = {
                "description": description or agent_class.__doc__,
                "tags": tags or [],
                "class": agent_class.__name__,
                "module": agent_class.__module__,
                **metadata
            }
            
            logger.info(f"Registered agent: {name} ({agent_class.__name__})")
            return agent_class
        
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Agent:
        """
        Create an agent instance by name.
        
        Args:
            name: Name of the agent to create
            **kwargs: Arguments to pass to the agent constructor
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent name is not registered
        """
        if name not in cls._agents:
            available = ", ".join(cls._agents.keys())
            raise ValueError(
                f"Unknown agent: {name}. Available agents: {available}"
            )
        
        agent_class = cls._agents[name]
        
        # Use the registry name as the agent name unless explicitly overridden
        if "name" not in kwargs:
            kwargs["name"] = name
        
        logger.info(f"Creating agent: {name} with args: {kwargs}")
        
        return agent_class(**kwargs)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """
        List all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(cls._agents.keys())
    
    @classmethod
    def get_agent_info(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata about a registered agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent metadata
            
        Raises:
            ValueError: If agent name is not registered
        """
        if name not in cls._agents:
            raise ValueError(f"Unknown agent: {name}")
        
        return cls._metadata.get(name, {})
    
    @classmethod
    def find_by_tag(cls, tag: str) -> List[str]:
        """
        Find agents by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of agent names with the given tag
        """
        results = []
        for name, metadata in cls._metadata.items():
            if tag in metadata.get("tags", []):
                results.append(name)
        return results
    
    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)."""
        cls._agents.clear()
        cls._metadata.clear()
        logger.info("Cleared agent registry")


# Convenience instance
registry = AgentRegistry()


# Auto-register built-in agents when imported
def _auto_register():
    """Auto-register built-in Dawn agents."""
    try:
        from dawn.agents.github import GitHubAgent
        registry.register(
            "github",
            description="GitHub repository research agent",
            tags=["research", "github", "code"]
        )(GitHubAgent)
    except ImportError:
        logger.debug("GitHub agent not available")
    
    try:
        from dawn.agents.arxiv import ArxivAgent
        registry.register(
            "arxiv",
            description="arXiv paper research agent",
            tags=["research", "papers", "academic"]
        )(ArxivAgent)
    except ImportError:
        logger.debug("arXiv agent not available")
    
    try:
        from dawn.agents.synthesis import SynthesisAgent
        registry.register(
            "synthesis",
            description="Multi-source synthesis agent",
            tags=["synthesis", "analysis", "comparison"]
        )(SynthesisAgent)
    except ImportError:
        logger.debug("Synthesis agent not available")


# Run auto-registration
_auto_register() 