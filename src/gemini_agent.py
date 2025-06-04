"""
Implementation of agent using the Google Gemini API.

This module provides a GeminiAgent implementation that works with
Gemini models via the Google Generative AI API.
"""
from typing import Dict, List, Optional, Any, Union
import uuid
import json
import logging
import sys
import time

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .interfaces import IAgent
from .config import APIConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('gemini_agent')


class GeminiAgent(IAgent):
    """
    Agent implementation that uses Google Gemini models.
    
    This agent can be used to perform various cognitive tasks using
    Gemini models via the Google Generative AI API.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[Dict[str, Any]],
        model: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new GeminiAgent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            capabilities: List of capabilities the agent provides
            model: Gemini model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt for all interactions
            metadata: Additional metadata about the agent
        """
        self._id = f"gemini-agent-{str(uuid.uuid4())[:8]}"
        self._name = name
        self._description = description
        self._capabilities = capabilities
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._metadata = metadata or {}
        
        # Initialize Gemini API
        if not APIConfig.is_gemini_configured():
            logger.warning("Gemini API is not configured. Agent may not function properly.")
            self._model_instance = None
        else:
            # Configure API
            genai.configure(api_key=APIConfig.GEMINI_API_KEY)
            
            # Set up safety settings
            self._safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Create model instance
            try:
                self._model_instance = genai.GenerativeModel(
                    model_name=self._model,
                    generation_config={
                        "temperature": self._temperature,
                        "max_output_tokens": self._max_tokens
                    },
                    safety_settings=self._safety_settings
                )
                logger.info(f"GeminiAgent initialized with model: {model}")
            except Exception as e:
                logger.error(f"Error initializing Gemini model: {str(e)}")
                self._model_instance = None
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities,
            "model": self._model,
            "provider": "google",
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
        # Check if model is initialized
        if self._model_instance is None:
            return {"error": "Gemini API is not configured"}
        
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
        
        # Invoke the Gemini API
        try:
            logger.info(f"Invoking Gemini API for capability: {capability_id}")
            
            # Add system prompt if provided
            if self._system_prompt:
                full_prompt = f"{self._system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            start_time = time.time()
            
            # Generate content
            response = self._model_instance.generate_content(full_prompt)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract the response content
            result = {
                "content": response.text,
                "model": self._model,
                "duration_seconds": duration
            }
            
            # Add usage information if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                result["usage"] = {
                    "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                }
            
            logger.info(f"Gemini API response received in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            error_type = e.__class__.__name__
            error_message = str(e)
            logger.error(f"Gemini API error ({error_type}): {error_message}")
            
            # Handle rate limiting
            if "quota" in error_message.lower() or "rate" in error_message.lower():
                return {"error": f"Rate limit exceeded: {error_message}", "retry_after": 60}
            
            return {"error": f"API error: {error_message}"}
    
    def _handle_chat(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat interactions with message history."""
        if "messages" not in inputs:
            return {"error": "Chat capability requires 'messages' in inputs"}
        
        messages = inputs["messages"]
        
        # Initialize chat session
        chat = self._model_instance.start_chat(
            history=[]
        )
        
        # Process messages
        system_prompt = None
        history = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
                continue
            
            history.append({
                "role": "user" if role == "user" else "model",
                "parts": [content]
            })
        
        # Add system prompt to the first user message if available
        if system_prompt and history:
            first_user_idx = None
            for i, msg in enumerate(history):
                if msg["role"] == "user":
                    first_user_idx = i
                    break
            
            if first_user_idx is not None:
                user_content = history[first_user_idx]["parts"][0]
                history[first_user_idx]["parts"][0] = f"{system_prompt}\n\n{user_content}"
        
        try:
            # Update chat history
            chat.history = history
            
            # Process the most recent user message to get a response
            last_user_message = None
            for msg in reversed(history):
                if msg["role"] == "user":
                    last_user_message = msg["parts"][0]
                    break
            
            if last_user_message is None:
                return {"error": "No user message found in the conversation"}
            
            # Send the last user message to get a response
            start_time = time.time()
            response = chat.send_message(last_user_message)
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract the response content
            result = {
                "content": response.text,
                "model": self._model,
                "duration_seconds": duration
            }
            
            # Add usage information if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                result["usage"] = {
                    "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                }
            
            logger.info(f"Gemini chat response received in {duration:.2f} seconds")
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
        if self._model_instance is None:
            return False
        
        try:
            # Simple health check message
            response = self._model_instance.generate_content("Reply with 'OK' if you're working.")
            content = response.text.strip().lower()
            return "ok" in content
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# Factory function to create a GeminiAgent with standard capabilities
def create_gemini_agent(
    name: str = "Gemini Assistant",
    description: str = "AI assistant powered by Google Gemini",
    model: str = "gemini-1.5-pro",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None
) -> GeminiAgent:
    """
    Create a GeminiAgent with standard capabilities.
    
    Args:
        name: Human-readable name for the agent
        description: Description of the agent's purpose
        model: Gemini model to use
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt for all interactions
        
    Returns:
        Configured GeminiAgent instance
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
    
    return GeminiAgent(
        name=name,
        description=description,
        capabilities=capabilities,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt
    )