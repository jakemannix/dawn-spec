"""
Implementation of agent using the Anthropic Claude API.

This module provides an AnthropicAgent implementation that works with
Claude models via the Anthropic API.
"""
from typing import Dict, List, Optional, Any, Union
import uuid
import time
import json
import logging
import sys

import anthropic
from anthropic.types import MessageParam

from .interfaces import IAgent
from .config import APIConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('anthropic_agent')


class AnthropicAgent(IAgent):
    """
    Agent implementation that uses Anthropic Claude models.
    
    This agent can be used to perform various cognitive tasks using
    Claude models via the Anthropic API.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[Dict[str, Any]],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new AnthropicAgent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            capabilities: List of capabilities the agent provides
            model: Anthropic model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt for all interactions
            metadata: Additional metadata about the agent
        """
        self._id = f"anthropic-agent-{str(uuid.uuid4())[:8]}"
        self._name = name
        self._description = description
        self._capabilities = capabilities
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._metadata = metadata or {}
        
        # Initialize Anthropic client
        if not APIConfig.is_anthropic_configured():
            logger.warning("Anthropic API is not configured. Agent may not function properly.")
            self._client = None
        else:
            self._client = anthropic.Anthropic(api_key=APIConfig.ANTHROPIC_API_KEY)
            logger.info(f"AnthropicAgent initialized with model: {model}")
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities,
            "model": self._model,
            "provider": "anthropic",
            "metadata": self._metadata
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return self._capabilities
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke a specific capability with given inputs and configuration.
        
        Args:
            capability_id: ID of the capability to invoke
            inputs: Input data for the capability
            config: Optional configuration parameters
            
        Returns:
            Dictionary containing the result of the capability invocation
        """
        # Check if client is initialized
        if self._client is None:
            return {"error": "Anthropic API is not configured"}
        
        # Find the requested capability
        capability = self._find_capability(capability_id)
        if capability is None:
            return {"error": f"Unknown capability: {capability_id}"}
        
        # Get configuration parameters (with defaults from agent config)
        merged_config = self._merge_configs(config or {})
        
        # Prepare the prompt based on capability type
        capability_type = capability.get("type", "")
        
        if "prompt" in inputs:
            prompt = inputs["prompt"]
        elif "text" in inputs:
            prompt = inputs["text"]
        else:
            # Default handling based on capability type
            if capability_type == "text_generation":
                prompt = self._format_text_generation_prompt(inputs, merged_config)
            elif capability_type == "chat":
                return self._handle_chat(inputs, merged_config)
            elif capability_type == "summarization":
                prompt = self._format_summarization_prompt(inputs, merged_config)
            elif capability_type == "classification":
                prompt = self._format_classification_prompt(inputs, merged_config)
            elif capability_type == "extraction":
                prompt = self._format_extraction_prompt(inputs, merged_config)
            else:
                # Generic case - just convert inputs to a prompt
                prompt = f"Task: {capability['name']}\n\nInputs:\n{json.dumps(inputs, indent=2)}"
        
        # Invoke the Claude API
        try:
            logger.info(f"Invoking Claude API for capability: {capability_id}")
            
            # Construct the messages array
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Invoke the model
            # Handle system prompt correctly
            system_prompt = merged_config.get("system_prompt")
            if system_prompt:
                response = self._client.messages.create(
                    model=merged_config["model"],
                    max_tokens=merged_config["max_tokens"],
                    temperature=merged_config["temperature"],
                    system=system_prompt,
                    messages=messages
                )
            else:
                response = self._client.messages.create(
                    model=merged_config["model"],
                    max_tokens=merged_config["max_tokens"],
                    temperature=merged_config["temperature"],
                    messages=messages
                )
            
            # Extract the response content
            result = {
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason,
                "finish_reason": response.stop_reason
            }
            
            logger.info(f"Claude API response received, tokens used: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")
            return result
            
        except Exception as e:
            # Handle all exceptions generically for testing simplicity
            error_type = e.__class__.__name__
            error_message = str(e)
            logger.error(f"Anthropic API error ({error_type}): {error_message}")
            
            # Check if it's a rate limit exception
            if "rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
                return {"error": f"Rate limit exceeded: {error_message}", "retry_after": 60}
            
            # Generic error handling
            return {"error": f"API error: {error_message}"}
    
    def _handle_chat(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat interactions with message history."""
        if "messages" not in inputs:
            return {"error": "Chat capability requires 'messages' in inputs"}
        
        messages = inputs["messages"]
        
        # Convert message format if needed
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map roles to Anthropic format
            if role == "assistant":
                role = "assistant"
            elif role == "system":
                # System messages are handled separately in Anthropic
                continue
            else:
                role = "user"
            
            anthropic_messages.append({"role": role, "content": content})
        
        try:
            # Extract system prompt from messages or use default
            system_prompt = config["system_prompt"]
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", system_prompt)
                    break
            
            # Call the Anthropic API
            if system_prompt:
                response = self._client.messages.create(
                    model=config["model"],
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    system=system_prompt,
                    messages=anthropic_messages
                )
            else:
                response = self._client.messages.create(
                    model=config["model"],
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    messages=anthropic_messages
                )
            
            # Extract the response content
            result = {
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason,
                "finish_reason": response.stop_reason
            }
            
            logger.info(f"Claude chat response received, tokens used: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")
            return result
            
        except Exception as e:
            logger.error(f"Error in chat handling: {str(e)}")
            return {"error": f"Chat processing error: {str(e)}"}
    
    def _format_text_generation_prompt(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Format a prompt for text generation capability."""
        prompt = inputs.get("prompt", "")
        if not prompt:
            if "topic" in inputs:
                prompt = f"Write about the following topic:\n\n{inputs['topic']}"
            elif "question" in inputs:
                prompt = f"Answer the following question:\n\n{inputs['question']}"
        return prompt
    
    def _format_summarization_prompt(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Format a prompt for summarization capability."""
        text = inputs.get("text", "")
        if not text:
            return "Please provide text to summarize in the 'text' field."
        
        prompt = f"Please summarize the following text:\n\n{text}\n\n"
        
        if "length" in inputs:
            prompt += f"The summary should be approximately {inputs['length']} words."
        elif "max_length" in inputs:
            prompt += f"The summary should be no more than {inputs['max_length']} words."
        
        return prompt
    
    def _format_classification_prompt(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Format a prompt for classification capability."""
        text = inputs.get("text", "")
        categories = inputs.get("categories", [])
        
        if not text:
            return "Please provide text to classify in the 'text' field."
        if not categories:
            return "Please provide categories for classification in the 'categories' field."
        
        categories_str = "\n".join([f"- {cat}" for cat in categories])
        prompt = f"""Please classify the following text into one of these categories:

{categories_str}

Text to classify:
{text}

Classification:"""
        return prompt
    
    def _format_extraction_prompt(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Format a prompt for extraction capability."""
        text = inputs.get("text", "")
        schema = inputs.get("schema", {})
        
        if not text:
            return "Please provide text to extract from in the 'text' field."
        if not schema:
            return "Please provide an extraction schema in the 'schema' field."
        
        schema_str = json.dumps(schema, indent=2)
        prompt = f"""From the following text, extract information according to this schema:

{schema_str}

Text:
{text}

Extracted information (in valid JSON format):"""
        return prompt
    
    def _find_capability(self, capability_id: str) -> Optional[Dict[str, Any]]:
        """Find a capability by ID."""
        for capability in self._capabilities:
            if capability.get("id") == capability_id:
                return capability
        return None
    
    def _merge_configs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided config with agent defaults."""
        return {
            "model": config.get("model", self._model),
            "temperature": config.get("temperature", self._temperature),
            "max_tokens": config.get("max_tokens", self._max_tokens),
            "system_prompt": config.get("system_prompt", self._system_prompt)
        }
    
    def health_check(self) -> bool:
        """
        Check if the agent is functioning properly.
        
        Returns:
            Boolean indicating whether the agent is healthy
        """
        if self._client is None:
            return False
        
        try:
            # Simple health check message
            response = self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply with 'OK' if you're working."}]
            )
            
            content = response.content[0].text.strip().lower()
            return "ok" in content
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# Factory function to create an AnthropicAgent with standard capabilities
def create_anthropic_agent(
    name: str = "Claude Assistant",
    description: str = "AI assistant powered by Anthropic Claude",
    model: str = "claude-3-sonnet-20240229",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None
) -> AnthropicAgent:
    """
    Create an AnthropicAgent with standard capabilities.
    
    Args:
        name: Human-readable name for the agent
        description: Description of the agent's purpose
        model: Anthropic model to use
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt for all interactions
        
    Returns:
        Configured AnthropicAgent instance
    """
    capabilities = [
        {
            "id": "text-generation",
            "type": "text_generation",
            "name": "Text Generation",
            "description": "Generate text based on a prompt"
        },
        {
            "id": "chat",
            "type": "chat",
            "name": "Chat Conversation",
            "description": "Engage in a conversational interaction"
        },
        {
            "id": "summarization",
            "type": "summarization",
            "name": "Text Summarization",
            "description": "Summarize longer text into a concise version"
        },
        {
            "id": "classification",
            "type": "classification",
            "name": "Text Classification",
            "description": "Classify text into predefined categories"
        },
        {
            "id": "extraction",
            "type": "extraction",
            "name": "Information Extraction",
            "description": "Extract structured information from text"
        }
    ]
    
    return AnthropicAgent(
        name=name,
        description=description,
        capabilities=capabilities,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt
    )