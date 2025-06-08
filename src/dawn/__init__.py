"""
Dawn - Multi-Protocol Agent Framework

A comprehensive framework for building agents that support multiple protocols
including Google A2A, AGNTCY ACP, and MCP with LangGraph reasoning.
"""

from dawn.core import Agent, LangGraphAgent
from dawn.protocols import ProtocolAdapter, ProtocolEndpoint
from dawn.utils.logging import get_logger
from dawn.runners.base import DemoRunner, BaseDemoRunner

# Version info
__version__ = "0.2.0"
__author__ = "Dawn Spec Contributors"

# Core exports
__all__ = [
    # Core classes
    "Agent",
    "LangGraphAgent",
    # Protocol interfaces
    "ProtocolAdapter", 
    "ProtocolEndpoint",
    # Demo runners
    "DemoRunner",
    "BaseDemoRunner",
    # Utilities
    "get_logger",
    # Version
    "__version__",
]

# Set up package-level logger
logger = get_logger(__name__)
logger.info(f"Dawn framework v{__version__} initialized")
