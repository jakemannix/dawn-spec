"""
A2A Interoperability Demo

This demo shows how DAWN agents can communicate using Google's A2A protocol,
demonstrating the peer-to-peer communication paradigm where both agents
maintain their own planning and reasoning capabilities.
"""
import asyncio
import logging
import sys
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import DAWN components
from src.agent import A2ACapableAgent, Capability


class MockTextAgent(A2ACapableAgent):
    """
    A mock agent that provides text processing capabilities.
    
    This agent demonstrates how DAWN agents can be exposed via A2A protocol.
    """
    
    def __init__(self, agent_name: str):
        super().__init__(
            name=agent_name,
            description=f"Mock text processing agent - {agent_name}",
            provider="DAWN Demo",
            version="1.0.0"
        )
        
        # Add text processing capabilities
        self._add_text_capabilities()
        
    def _add_text_capabilities(self):
        """Add text processing capabilities to the agent."""
        
        # Text transformation capability
        transform_cap = Capability(
            capability_type="text_transformation",
            name="Text Transform",
            description="Transform text by applying various operations",
            parameters={
                "text": {
                    "type": "string",
                    "description": "The text to transform",
                    "required": True
                },
                "operation": {
                    "type": "string",
                    "description": "Operation to apply: uppercase, lowercase, reverse, or title",
                    "required": True
                }
            }
        )
        self.add_capability(transform_cap)
        
        # Text analysis capability
        analyze_cap = Capability(
            capability_type="text_analysis",
            name="Text Analysis",
            description="Analyze text and provide statistics",
            parameters={
                "text": {
                    "type": "string",
                    "description": "The text to analyze",
                    "required": True
                }
            }
        )
        self.add_capability(analyze_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> list[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"MockTextAgent invoking capability {capability_id} with inputs: {inputs}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "text_transformation":
                return self._handle_text_transformation(inputs)
            elif capability.type == "text_analysis":
                return self._handle_text_analysis(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in capability {capability_id}: {e}")
            return {"error": str(e)}
            
    def _handle_text_transformation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text transformation requests."""
        text = inputs.get("text", "")
        operation = inputs.get("operation", "").lower()
        
        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "title":
            result = text.title()
        else:
            return {"error": f"Unknown operation: {operation}"}
            
        return {
            "original_text": text,
            "operation": operation,
            "result": result,
            "success": True
        }
        
    def _handle_text_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text analysis requests."""
        text = inputs.get("text", "")
        
        # Simple text analysis
        words = text.split()
        chars = len(text)
        chars_no_spaces = len(text.replace(" ", ""))
        
        return {
            "text": text,
            "statistics": {
                "character_count": chars,
                "character_count_no_spaces": chars_no_spaces,
                "word_count": len(words),
                "sentence_count": text.count('.') + text.count('!') + text.count('?'),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
            },
            "success": True
        }
        
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


class MockReasoningAgent(A2ACapableAgent):
    """
    A mock agent that provides reasoning and planning capabilities.
    
    This agent demonstrates how one agent can delegate tasks to another via A2A.
    """
    
    def __init__(self, agent_name: str):
        super().__init__(
            name=agent_name,
            description=f"Mock reasoning agent - {agent_name}",
            provider="DAWN Demo",
            version="1.0.0"
        )
        
        # Add reasoning capabilities
        self._add_reasoning_capabilities()
        
    def _add_reasoning_capabilities(self):
        """Add reasoning capabilities to the agent."""
        
        # Text processing workflow capability
        workflow_cap = Capability(
            capability_type="text_workflow",
            name="Text Processing Workflow",
            description="Execute a complete text processing workflow using remote agents",
            parameters={
                "text": {
                    "type": "string",
                    "description": "The text to process",
                    "required": True
                },
                "workflow_steps": {
                    "type": "array",
                    "description": "List of processing steps to execute",
                    "required": True
                },
                "remote_agent_url": {
                    "type": "string", 
                    "description": "URL of the remote text processing agent",
                    "required": True
                }
            }
        )
        self.add_capability(workflow_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> list[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"MockReasoningAgent invoking capability {capability_id} with inputs: {inputs}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "text_workflow":
                # This is an async operation, but we'll simulate it for the demo
                return {"error": "This capability requires async execution. Use invoke_async instead."}
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in capability {capability_id}: {e}")
            return {"error": str(e)}
            
    async def invoke_async(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of invoke for capabilities that require remote calls."""
        logger.info(f"MockReasoningAgent async invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "text_workflow":
                return await self._handle_text_workflow(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in async capability {capability_id}: {e}")
            return {"error": str(e)}
            
    async def _handle_text_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text workflow requests by delegating to remote agent."""
        text = inputs.get("text", "")
        workflow_steps = inputs.get("workflow_steps", [])
        remote_agent_url = inputs.get("remote_agent_url", "")
        
        if not remote_agent_url:
            return {"error": "remote_agent_url is required"}
            
        try:
            # Connect to remote agent
            logger.info(f"Connecting to remote agent at {remote_agent_url}")
            remote_agent_id = await self.connect_to_a2a_agent(remote_agent_url, "text_processor")
            
            # Discover capabilities on remote agent
            logger.info("Discovering remote capabilities")
            remote_capabilities = await self.discover_remote_a2a_capabilities(remote_agent_id)
            logger.info(f"Found {len(remote_capabilities)} remote capabilities")
            
            # Execute workflow steps
            results = []
            current_text = text
            
            for step in workflow_steps:
                logger.info(f"Executing workflow step: {step}")
                
                if step.get("type") == "transform":
                    # Find text transformation capability
                    transform_cap = None
                    for cap in remote_capabilities:
                        if "text_transformation" in cap.get("name", "").lower() or "transform" in cap.get("name", "").lower():
                            transform_cap = cap
                            break
                            
                    if transform_cap:
                        result = await self.invoke_remote_a2a_capability(
                            remote_agent_id,
                            transform_cap["name"], 
                            {
                                "text": current_text,
                                "operation": step.get("operation", "uppercase")
                            }
                        )
                        if result.get("success"):
                            current_text = result.get("result", current_text)
                            results.append(result)
                        else:
                            results.append({"step": step, "error": result.get("error", "Unknown error")})
                    else:
                        results.append({"step": step, "error": "Text transformation capability not found"})
                        
                elif step.get("type") == "analyze":
                    # Find text analysis capability
                    analyze_cap = None
                    for cap in remote_capabilities:
                        if "text_analysis" in cap.get("name", "").lower() or "analysis" in cap.get("name", "").lower():
                            analyze_cap = cap
                            break
                            
                    if analyze_cap:
                        result = await self.invoke_remote_a2a_capability(
                            remote_agent_id,
                            analyze_cap["name"],
                            {"text": current_text}
                        )
                        results.append(result)
                    else:
                        results.append({"step": step, "error": "Text analysis capability not found"})
                else:
                    results.append({"step": step, "error": f"Unknown step type: {step.get('type')}"})
            
            return {
                "original_text": text,
                "final_text": current_text,
                "workflow_steps": workflow_steps,
                "step_results": results,
                "remote_agent_capabilities": remote_capabilities,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in text workflow: {e}")
            return {"error": str(e)}
            
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


async def run_a2a_server_demo():
    """Run the A2A server demo."""
    logger.info("=== A2A Server Demo ===")
    
    # Create a text processing agent
    text_agent = MockTextAgent("TextProcessor")
    
    logger.info(f"Created agent: {text_agent.name}")
    logger.info(f"Agent capabilities: {[cap.name for cap in text_agent.capabilities]}")
    
    try:
        # Start the agent as an A2A server
        logger.info("Starting A2A server on port 8080...")
        server = await text_agent.start_a2a_server(port=8080)
        
        server_info = text_agent.get_a2a_server_info()
        logger.info(f"A2A server started: {server_info}")
        
        logger.info("A2A server is running. You can connect to it from another terminal.")
        logger.info("Press Ctrl+C to stop the server.")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping server...")
    except Exception as e:
        logger.error(f"Error running A2A server: {e}")
    finally:
        await text_agent.stop_a2a_server()
        logger.info("A2A server stopped")


async def run_a2a_client_demo():
    """Run the A2A client demo."""
    logger.info("=== A2A Client Demo ===")
    
    # Create a reasoning agent that will act as a client
    reasoning_agent = MockReasoningAgent("WorkflowOrchestrator")
    
    logger.info(f"Created reasoning agent: {reasoning_agent.name}")
    
    # Define a workflow to execute on the remote agent
    workflow_inputs = {
        "text": "Hello World! This is a test of the A2A protocol integration.",
        "workflow_steps": [
            {"type": "analyze"},  # First analyze the original text
            {"type": "transform", "operation": "uppercase"},  # Transform to uppercase
            {"type": "analyze"},  # Analyze the transformed text
            {"type": "transform", "operation": "reverse"},  # Reverse the text
            {"type": "analyze"}  # Final analysis
        ],
        "remote_agent_url": "http://localhost:8080"  # Connect to the server
    }
    
    try:
        # Get the workflow capability
        workflow_capability = None
        for cap in reasoning_agent.capabilities:
            if cap.type == "text_workflow":
                workflow_capability = cap
                break
                
        if workflow_capability:
            logger.info("Executing text processing workflow...")
            result = await reasoning_agent.invoke_async(workflow_capability.id, workflow_inputs)
            
            if result.get("success"):
                logger.info("Workflow completed successfully!")
                logger.info(f"Original text: {result['original_text']}")
                logger.info(f"Final text: {result['final_text']}")
                logger.info(f"Steps executed: {len(result['step_results'])}")
                
                # Show step results
                for i, step_result in enumerate(result['step_results']):
                    logger.info(f"Step {i+1}: {step_result}")
            else:
                logger.error(f"Workflow failed: {result.get('error')}")
        else:
            logger.error("Workflow capability not found")
            
    except Exception as e:
        logger.error(f"Error running A2A client: {e}")


async def main():
    """Main function to run the demo."""
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        await run_a2a_client_demo()
    else:
        await run_a2a_server_demo()


if __name__ == "__main__":
    asyncio.run(main())