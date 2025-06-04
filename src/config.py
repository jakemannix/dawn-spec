"""
Configuration module for DAWN/AGNTCY implementation.

This module loads environment variables and provides configuration settings
for the application. It uses python-dotenv to load variables from a .env file.
"""
import os
from typing import Dict, Optional, Any, Union

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class APIConfig:
    """Configuration for API credentials and settings."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")
    
    # Anthropic API Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # GitHub API Configuration
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    
    @classmethod
    def is_openai_configured(cls) -> bool:
        """Check if OpenAI API is configured."""
        return bool(cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith("sk-xxxxxxx"))
    
    @classmethod
    def is_anthropic_configured(cls) -> bool:
        """Check if Anthropic API is configured."""
        return bool(cls.ANTHROPIC_API_KEY and not cls.ANTHROPIC_API_KEY.startswith("sk-ant-xxxxxxx"))
    
    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Check if Gemini API is configured."""
        return bool(cls.GEMINI_API_KEY and not cls.GEMINI_API_KEY.startswith("AIza-xxxxxxx"))
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get a dictionary of available providers."""
        return {
            "openai": cls.is_openai_configured(),
            "anthropic": cls.is_anthropic_configured(),
            "gemini": cls.is_gemini_configured(),
        }


class AgentConfig:
    """Configuration for agent behavior and defaults."""
    
    # Default model for agents
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o")
    
    # Principal agent type
    PRINCIPAL_AGENT_TYPE: str = os.getenv("PRINCIPAL_AGENT_TYPE", "openai")
    
    # Generation parameters
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    
    @classmethod
    def get_principal_agent_config(cls) -> Dict[str, Any]:
        """Get principal agent configuration."""
        return {
            "type": cls.PRINCIPAL_AGENT_TYPE,
            "model": cls.DEFAULT_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
        }


class ServiceConfig:
    """Configuration for service endpoints and authentication."""
    
    # ACP server configuration
    ACP_SERVER_HOST: str = os.getenv("ACP_SERVER_HOST", "localhost")
    ACP_SERVER_PORT: int = int(os.getenv("ACP_SERVER_PORT", "8000"))
    
    # Authentication settings
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    
    @classmethod
    def get_acp_server_url(cls) -> str:
        """Get ACP server URL."""
        return f"http://{cls.ACP_SERVER_HOST}:{cls.ACP_SERVER_PORT}"


def check_configuration() -> Dict[str, Union[bool, str]]:
    """
    Check the configuration and return the status.
    
    Returns:
        Dictionary with configuration status
    """
    providers = APIConfig.get_available_providers()
    principal_agent_type = AgentConfig.PRINCIPAL_AGENT_TYPE
    
    # Check if requested principal agent provider is configured
    if principal_agent_type in providers and not providers[principal_agent_type]:
        principal_status = f"Error: {principal_agent_type} is not properly configured"
    else:
        principal_status = f"Using {principal_agent_type} as principal agent"
    
    return {
        "providers": providers,
        "principal_agent": principal_status,
        "acp_server": ServiceConfig.get_acp_server_url(),
        "auth_enabled": ServiceConfig.ENABLE_AUTH,
    }


if __name__ == "__main__":
    """Print configuration status when run directly."""
    import json
    
    config_status = check_configuration()
    print(json.dumps(config_status, indent=2))
    
    print("\nAvailable providers:")
    for provider, available in APIConfig.get_available_providers().items():
        status = "✅ Configured" if available else "❌ Not configured"
        print(f"- {provider}: {status}")