"""
Dawn Agents

This module provides pre-built agent implementations and a registry
system for agent discovery.
"""

from dawn.agents.registry import registry, AgentRegistry

# Import agents if available (they auto-register)
try:
    from dawn.agents.github import GitHubAgent
except ImportError:
    pass

try:
    from dawn.agents.arxiv import ArxivAgent
except ImportError:
    pass

try:
    from dawn.agents.synthesis import SynthesisAgent
except ImportError:
    pass

__all__ = [
    "registry",
    "AgentRegistry",
]
