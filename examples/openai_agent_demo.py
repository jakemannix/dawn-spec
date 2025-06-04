"""
Demonstration of implementing a DAWN agent using OpenAI.

This example shows how to implement a DAWN agent using the OpenAI API
and ReAct (Reasoning+Acting) pattern for task handling.
"""
import sys
import os
import json
import uuid
from typing import Dict, List, Optional, Any, Union

# Add the parent directory to the Python path to allow importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interfaces import IAgent, IPrincipalAgent
from src.config import APIConfig, AgentConfig, check_configuration

# Import OpenAI
import openai

# Check if OpenAI is configured
if not APIConfig.is_openai_configured():
    print("Error: OpenAI API is not configured.")
    print("Please set OPENAI_API_KEY in .env file.")
    print("You can copy template.env to .env and update with your API key.")
    sys.exit(1)

# Configure OpenAI client
client = openai.OpenAI(
    api_key=APIConfig.OPENAI_API_KEY,
    organization=APIConfig.OPENAI_ORG_ID
)


class OpenAIPrincipalAgent(IPrincipalAgent):
    """Implementation of Principal Agent using OpenAI's API with ReAct pattern."""
    
    def __init__(self, name: str = "OpenAI Principal Agent", model: Optional[str] = None):
        self._id = str(uuid.uuid4())
        self._name = name
        self._description = "Principal agent using OpenAI's API with ReAct pattern"
        self._capabilities = [
            {
                "id": "task-decomposition",
                "type": "task_decomposition",
                "name": "Task Decomposition",
                "description": "Break down complex tasks into simpler subtasks"
            },
            {
                "id": "agent-selection",
                "type": "agent_selection",
                "name": "Agent Selection",
                "description": "Select appropriate agents for subtasks"
            },
            {
                "id": "task-execution",
                "type": "task_execution",
                "name": "Task Execution",
                "description": "Execute and coordinate task execution across agents"
            }
        ]
        
        # Configuration
        self._model = model or AgentConfig.DEFAULT_MODEL
        self._temperature = AgentConfig.TEMPERATURE
        self._max_tokens = AgentConfig.MAX_TOKENS
        
        # Log the selected model
        print(f"Using model: {self._model}")
        
    def get_info(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "capabilities": self._capabilities
        }
        
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return self._capabilities
        
    def health_check(self) -> bool:
        try:
            # Simple test completion to check if the API is working
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hello, are you working? Respond with 'yes' or 'no'."}],
                max_tokens=10
            )
            content = response.choices[0].message.content.strip().lower()
            return "yes" in content
        except Exception as e:
            error_message = str(e)
            print(f"OpenAI API health check failed: {error_message}")
            
            # Provide more helpful information based on common error types
            if "401" in error_message:
                print("\nYour API key may be invalid or has expired. Please check your API key in the .env file.")
            elif "429" in error_message:
                print("\nYou've hit a rate limit or quota. This could be due to:")
                print("- Exceeding your OpenAI API usage tier")
                print("- Too many requests in a short time period")
                print("- Insufficient quota for the specific model")
                print("\nCheck your OpenAI usage dashboard at: https://platform.openai.com/usage")
            elif "404" in error_message:
                print(f"\nThe model '{self._model}' was not found or is not available to you.")
                print("Try using a different model or check your OpenAI account access.")
            
            return False
            
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Simple dispatch based on capability
        if capability_id == "task-decomposition":
            return self._invoke_task_decomposition(inputs, config)
        elif capability_id == "agent-selection":
            return self._invoke_agent_selection(inputs, config)
        elif capability_id == "task-execution":
            return self._invoke_task_execution(inputs, config)
        else:
            raise ValueError(f"Unknown capability: {capability_id}")
    
    def _invoke_task_decomposition(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke task decomposition capability."""
        task_description = inputs.get("description", "")
        if not task_description:
            return {"error": "Task description is required"}
            
        # Construct the prompt for task decomposition
        prompt = f"""
        Your task is to break down the following complex task into smaller, manageable subtasks:
        
        TASK: {task_description}
        
        For each subtask, provide:
        1. A clear description
        2. The input requirements
        3. The expected output format
        4. Required capabilities (what kind of agent would handle this)
        5. Dependencies (which other subtasks must be completed first)
        
        Format your response as a JSON array of subtask objects with the following structure:
        {{
            "subtasks": [
                {{
                    "id": "subtask-1",
                    "description": "Description of subtask 1",
                    "inputs": {{"key1": "value1", "key2": "value2"}},
                    "output_schema": {{"expected_key1": "type", "expected_key2": "type"}},
                    "required_capabilities": ["capability_type1", "capability_type2"],
                    "dependencies": []
                }},
                {{
                    "id": "subtask-2",
                    "description": "Description of subtask 2",
                    "inputs": {{"key1": "value1", "key2": "value2"}},
                    "output_schema": {{"expected_key1": "type", "expected_key2": "type"}},
                    "required_capabilities": ["capability_type3"],
                    "dependencies": ["subtask-1"]
                }}
            ]
        }}
        
        ENSURE:
        - Each subtask has a clear purpose and well-defined inputs/outputs
        - Dependencies are correctly specified (no circular dependencies)
        - The complete set of subtasks fully addresses the original task
        - Each required capability is specific (e.g., "text_generation", "data_retrieval", "calculation")
        
        RESPONSE:
        """
        
        # Call OpenAI API for decomposition
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens
            )
            
            content = response.choices[0].message.content.strip()
            
            # Find and extract the JSON part
            try:
                # Parse the JSON from the response
                # Look for the start of a JSON object or array
                json_start = content.find('{')
                if json_start == -1:
                    json_start = content.find('[')
                
                if json_start != -1:
                    # Extract the JSON part
                    json_content = content[json_start:]
                    result = json.loads(json_content)
                    return result
                else:
                    return {"error": "Invalid response format - no JSON found", "content": content}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON in response", "content": content}
                
        except Exception as e:
            return {"error": f"API error: {str(e)}"}
    
    def _invoke_agent_selection(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke agent selection capability."""
        # Implementation would follow similar pattern to task decomposition
        return {"message": "Agent selection not implemented yet"}
    
    def _invoke_task_execution(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke task execution capability."""
        # Implementation would follow similar pattern to task decomposition
        return {"message": "Task execution not implemented yet"}
    
    # IPrincipalAgent interface methods
    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = self._invoke_task_decomposition({"description": task.get("description", "")})
        return result.get("subtasks", [])
        
    def discover_agents(self, capability_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        # This would use the Gateway Agent in a real implementation
        # For now, return a dummy response
        return {"message": "Agent discovery not implemented yet"}
        
    def create_execution_plan(self, subtasks: List[Dict[str, Any]], available_agents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        # This would create an execution plan in a real implementation
        # For now, return a dummy response
        return [{"message": "Execution plan creation not implemented yet"}]
        
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This would execute the plan in a real implementation
        # For now, return a dummy response
        return {"message": "Plan execution not implemented yet"}
        
    def handle_error(self, error: Dict[str, Any], execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This would handle errors in a real implementation
        # For now, return a dummy response
        return {"message": "Error handling not implemented yet"}


def check_openai_tier_info():
    """Check OpenAI tier and billing information if possible."""
    print("Checking OpenAI account information...\n")
    
    try:
        # Check billing info if available in the API
        # Note: Current OpenAI API doesn't have direct billing/quota endpoints exposed
        # This is a placeholder for when that functionality becomes available
        
        # For now, we'll use a simple trial completion to diagnose issues
        test_models = ["gpt-4o", "gpt-4"]
        results = {}
        
        for model in test_models:
            print(f"Testing access to {model}...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                results[model] = "✅ Accessible"
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg:
                    results[model] = "❌ Authentication error"
                elif "429" in error_msg:
                    results[model] = "❌ Rate limit or quota exceeded"
                elif "404" in error_msg:
                    results[model] = "❌ Model not found or not available"
                else:
                    results[model] = f"❌ Error: {error_msg[:100]}"
        
        print("\nModel Access Status:")
        for model, status in results.items():
            print(f"- {model}: {status}")
            
        # Check if any models are accessible
        if any("✅" in status for status in results.values()):
            print("\n✅ Your OpenAI account has access to at least one model.")
        else:
            print("\n❌ Your account doesn't have access to any of the tested models.")
            print("This might be due to:")
            print("- Invalid API key")
            print("- Account not set up with billing")
            print("- Exceeded usage limits")
            print("\nCheck your account at: https://platform.openai.com/account")
            
    except Exception as e:
        print(f"Error checking tier info: {str(e)}")


def check_available_models():
    """Check available OpenAI models and quotas."""
    print("Checking available OpenAI models...\n")
    
    try:
        # List available models
        models = client.models.list()
        
        # Parse the response
        available_models = []
        for model in models.data:
            model_info = {
                "id": model.id,
                "owned_by": getattr(model, "owned_by", "Unknown"),
                "created": getattr(model, "created", None),
            }
            available_models.append(model_info)
        
        # Sort models by owner and ID
        available_models.sort(key=lambda x: (x["owned_by"], x["id"]))
        
        # Group by owner
        models_by_owner = {}
        for model in available_models:
            owner = model["owned_by"]
            if owner not in models_by_owner:
                models_by_owner[owner] = []
            models_by_owner[owner].append(model["id"])
        
        print("Available models by owner:")
        for owner, models in models_by_owner.items():
            print(f"\n{owner}:")
            for model in models:
                print(f"  - {model}")
        
        # Check specifically for gpt-4 models
        gpt4_models = [m["id"] for m in available_models if "gpt-4" in m["id"]]
        if gpt4_models:
            print("\nGPT-4 models available:")
            for model in gpt4_models:
                print(f"  - {model}")
        else:
            print("\nNo GPT-4 models available - you may need to request access.")
        
        return available_models
    except Exception as e:
        print(f"Error checking models: {str(e)}")
        return []


def run_demo():
    """Run a simple demo of the OpenAI principal agent."""
    print("=== OpenAI Agent Demo ===\n")
    
    # Check configuration
    config_status = check_configuration()
    print("Configuration status:")
    print(json.dumps(config_status, indent=2))
    print()
    
    # Check OpenAI account tier and model access
    check_openai_tier_info()
    print()
    
    # Check available models
    available_models = check_available_models()
    print()
    
    # Select a model based on availability
    model_to_use = AgentConfig.DEFAULT_MODEL  # Default from config
    
    # Extract all available model IDs
    model_ids = [model["id"] for model in available_models]
    
    # Check if the configured model is available
    if model_to_use not in model_ids:
        print(f"Error: The configured model '{model_to_use}' is not available with your current OpenAI account.")
        print("Available models:")
        for model in sorted(model_ids):
            if "gpt-4" in model:
                print(f"  - {model}")
        print("\nPlease update your .env file with one of these models.")
        return
    
    # Create the principal agent with the selected model
    principal_agent = OpenAIPrincipalAgent(model=model_to_use)
    
    # Check agent health
    health_status = principal_agent.health_check()
    print(f"Agent health status: {'✅ Healthy' if health_status else '❌ Unhealthy'}")
    
    if not health_status:
        print("Agent health check failed. Exiting.")
        return
    
    # Get agent info
    agent_info = principal_agent.get_info()
    print("\nAgent information:")
    print(json.dumps(agent_info, indent=2))
    
    # Demo task decomposition
    print("\n=== Task Decomposition Demo ===\n")
    
    # Define a task
    task = {
        "description": "Create a weekly report on competitor pricing for our main products and email it to the marketing team."
    }
    
    print(f"Task: {task['description']}\n")
    
    # Decompose the task
    print("Decomposing task...")
    subtasks = principal_agent.decompose_task(task)
    
    # Print decomposed subtasks
    print("\nTask decomposed into the following subtasks:\n")
    for i, subtask in enumerate(subtasks, 1):
        print(f"Subtask {i}: {subtask['description']}")
        print(f"  Required capabilities: {subtask['required_capabilities']}")
        print(f"  Dependencies: {subtask['dependencies']}")
        print()
    
    print("=== Demo Complete ===")


def run_diagnostics():
    """Run only the OpenAI diagnostics without the full demo."""
    print("=== OpenAI Diagnostics ===\n")
    
    # Check configuration
    config_status = check_configuration()
    print("Configuration status:")
    print(json.dumps(config_status, indent=2))
    print()
    
    # Check OpenAI account tier and model access
    check_openai_tier_info()
    print()
    
    # Check available models
    available_models = check_available_models()
    print()
    
    print("=== Diagnostics Complete ===")
    print(f"\nYour current DEFAULT_MODEL is set to: {AgentConfig.DEFAULT_MODEL}")
    print("Important: The demo will only use the model configured in your .env file.")
    print("If that model is not available to your account, the demo will exit with an error.")
    print("Please update your .env file to use one of the available models listed above.")


def run_demo_with_error_handling():
    """Run the demo with error handling and fallback to diagnostics if needed."""
    try:
        # First try to run the demo directly
        print("=== OpenAI Agent Demo ===\n")
        
        # Check configuration
        config_status = check_configuration()
        print("Configuration status:")
        print(json.dumps(config_status, indent=2))
        print()
        
        # Get available models without extensive diagnostics
        try:
            models = client.models.list()
            model_ids = [model.id for model in models.data]
            
            # Use the model from the config (.env)
            model_to_use = AgentConfig.DEFAULT_MODEL
            
            # Check if the configured model is available
            if model_to_use not in model_ids:
                print(f"Error: The configured model '{model_to_use}' is not available with your current OpenAI account.")
                print("Available models:")
                for model in sorted(model_ids):
                    if "gpt-4" in model:
                        print(f"  - {model}")
                print("\nPlease update your .env file with one of these models.")
                run_diagnostics()
                return
            
            # Create and run the agent
            principal_agent = OpenAIPrincipalAgent(model=model_to_use)
            
            # Check agent health
            health_status = principal_agent.health_check()
            if not health_status:
                print("\nAgent health check failed. Running diagnostics...\n")
                run_diagnostics()
                return
            
            # Continue with the demo if health check passed
            agent_info = principal_agent.get_info()
            print("\nAgent information:")
            print(json.dumps(agent_info, indent=2))
            
            # Demo task decomposition
            print("\n=== Task Decomposition Demo ===\n")
            
            # Define a task
            task = {
                "description": "Create a weekly report on competitor pricing for our main products and email it to the marketing team."
            }
            
            print(f"Task: {task['description']}\n")
            
            # Decompose the task
            print("Decomposing task...")
            subtasks = principal_agent.decompose_task(task)
            
            # Print decomposed subtasks
            print("\nTask decomposed into the following subtasks:\n")
            for i, subtask in enumerate(subtasks, 1):
                print(f"Subtask {i}: {subtask['description']}")
                print(f"  Required capabilities: {subtask['required_capabilities']}")
                print(f"  Dependencies: {subtask['dependencies']}")
                print()
            
            print("=== Demo Complete ===")
            
        except Exception as e:
            print(f"Error during model selection: {str(e)}")
            print("\nRunning diagnostics to troubleshoot...\n")
            run_diagnostics()
            return
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print("\nRunning diagnostics to troubleshoot...\n")
        run_diagnostics()
        return


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run diagnostics only
    if len(sys.argv) > 1 and sys.argv[1].lower() == "diagnostics":
        run_diagnostics()
    # Otherwise run the demo with error handling
    else:
        run_demo_with_error_handling()