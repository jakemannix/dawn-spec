# src/agent_factory.py

import yaml
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from .agent_core import AgentOrchestrator, AgentImplementationType
from .implementations.text_matching_agent import TextMatchingAgent

# Import LangGraph conditionally
try:
    from .implementations.langgraph_agent import LangGraphAgent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class AgentFactory:
    """Factory for creating and configuring agent implementations"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml", env_file: str = ".env"):
        # Load environment variables from .env file
        load_dotenv(env_file)
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration if file is missing"""
        return {
            "default_implementation": "text_matching",
            "implementations": {
                "text_matching": {"enabled": True},
                "langgraph": {"enabled": False}
            },
            "global": {
                "logging": {"level": "INFO"}
            }
        }
    
    async def create_orchestrator(self) -> AgentOrchestrator:
        """Create and configure an agent orchestrator with all enabled implementations"""
        orchestrator = AgentOrchestrator()
        
        # Configure logging
        logging_config = self.config.get("global", {}).get("logging", {})
        log_level = logging_config.get("level", "INFO")
        logging.basicConfig(level=getattr(logging, log_level))
        
        # Register all enabled implementations
        for impl_name, impl_config in self.config.get("implementations", {}).items():
            if impl_config.get("enabled", False):
                try:
                    implementation = await self._create_implementation(impl_name, impl_config)
                    impl_type = AgentImplementationType(impl_name)
                    await orchestrator.register_implementation(impl_type, implementation)
                    self.logger.info(f"Registered {impl_name} implementation")
                except Exception as e:
                    self.logger.error(f"Failed to register {impl_name}: {e}")
        
        # Set default active implementation
        default_impl = self.config.get("default_implementation", "text_matching")
        try:
            default_type = AgentImplementationType(default_impl)
            await orchestrator.set_active_implementation(default_type)
            self.logger.info(f"Set active implementation to {default_impl}")
        except Exception as e:
            self.logger.error(f"Failed to set default implementation {default_impl}: {e}")
        
        return orchestrator
    
    async def _create_implementation(self, impl_name: str, config: Dict[str, Any]):
        """Create a specific agent implementation"""
        # Apply environment variable overrides to config
        enhanced_config = self._apply_env_overrides(impl_name, config)
        
        if impl_name == "text_matching":
            return TextMatchingAgent(enhanced_config)
        elif impl_name == "langgraph":
            if not LANGGRAPH_AVAILABLE:
                raise ImportError("LangGraph is not available. Install with: uv pip install -e .[langgraph]")
            return LangGraphAgent(enhanced_config)
        else:
            raise ValueError(f"Unknown implementation: {impl_name}")
    
    def _apply_env_overrides(self, impl_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        enhanced_config = config.copy()
        
        if impl_name == "langgraph" and "llm" in enhanced_config:
            llm_config = enhanced_config["llm"].copy()
            
            # Override with environment variables if present
            if os.getenv("PRINCIPAL_AGENT_TYPE"):
                provider = self._clean_env_value(os.getenv("PRINCIPAL_AGENT_TYPE"))
                if provider:
                    self.logger.debug(f"Overriding provider with env var: {provider}")
                    llm_config["provider"] = provider
                    
            if os.getenv("DEFAULT_MODEL"):
                model = self._clean_env_value(os.getenv("DEFAULT_MODEL"))
                if model:
                    self.logger.debug(f"Overriding model with env var: {model}")
                    llm_config["model"] = model
                    
            if os.getenv("TEMPERATURE"):
                temp_str = self._clean_env_value(os.getenv("TEMPERATURE"))
                if temp_str:
                    self.logger.debug(f"Overriding temperature with env var: {temp_str}")
                    llm_config["temperature"] = float(temp_str)
                    
            if os.getenv("MAX_TOKENS"):
                tokens_str = self._clean_env_value(os.getenv("MAX_TOKENS"))
                if tokens_str:
                    self.logger.debug(f"Overriding max_tokens with env var: {tokens_str}")
                    llm_config["max_tokens"] = int(tokens_str)
                
            enhanced_config["llm"] = llm_config
        
        return enhanced_config
    
    def _clean_env_value(self, value: str) -> str:
        """Clean environment variable value by removing comments and whitespace"""
        if not value:
            return ""
        # Remove inline comments and strip whitespace
        return value.split('#')[0].strip()
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration section"""
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration (runtime only, doesn't save to file)"""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Convenience function for quick setup
async def create_default_orchestrator() -> AgentOrchestrator:
    """Create orchestrator with default configuration"""
    factory = AgentFactory()
    return await factory.create_orchestrator()


# Example usage function
async def switch_implementation_example():
    """Example of how to switch between implementations at runtime"""
    orchestrator = await create_default_orchestrator()
    
    # Use text matching initially
    response1 = await orchestrator.process_request("search github for langchain")
    print(f"Text matching response: {response1.response_text}")
    
    # Switch to LangGraph (if available)
    try:
        await orchestrator.set_active_implementation(AgentImplementationType.LANGGRAPH)
        response2 = await orchestrator.process_request("search github for langchain")
        print(f"LangGraph response: {response2.response_text}")
    except Exception as e:
        print(f"LangGraph not available: {e}")
    
    # Health check
    health = await orchestrator.health_check()
    print(f"System health: {health}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(switch_implementation_example()) 