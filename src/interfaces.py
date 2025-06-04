"""
Core interfaces for the DAWN/AGNTCY specification implementation.

This module defines the interfaces that all components in the system must implement
to ensure interoperability and standardization.
"""
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod


class IAgent(ABC):
    """
    Interface that all agents must implement.
    
    This is the base interface for any agent in the system, defining
    the minimal set of methods required for agent interoperability.
    """
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Return agent metadata including capabilities.
        
        Returns:
            Dictionary containing agent metadata and capabilities.
        """
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Return list of agent capabilities.
        
        Returns:
            List of dictionaries describing agent capabilities.
        """
        pass
        
    @abstractmethod
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke a specific capability with given inputs and configuration.
        
        Args:
            capability_id: ID of the capability to invoke
            inputs: Input data for the capability
            config: Optional configuration parameters
            
        Returns:
            Dictionary containing the result of the capability invocation.
        """
        pass
        
    @abstractmethod
    def health_check(self) -> bool:
        """
        Return the health status of the agent.
        
        Returns:
            Boolean indicating whether the agent is healthy.
        """
        pass


class IPrincipalAgent(IAgent):
    """
    Interface for the orchestration agent in DAWN architecture.
    
    The Principal Agent is responsible for planning and orchestrating
    complex tasks across multiple specialized agents.
    """
    
    @abstractmethod
    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down a complex task into subtasks.
        
        Args:
            task: Dictionary describing the complex task
            
        Returns:
            List of dictionaries describing subtasks.
        """
        pass
        
    @abstractmethod
    def discover_agents(self, capability_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Find agents with capabilities matching requirements.
        
        Args:
            capability_requirements: List of dictionaries describing required capabilities
            
        Returns:
            Dictionary mapping capability IDs to lists of agent IDs.
        """
        pass
        
    @abstractmethod
    def create_execution_plan(self, subtasks: List[Dict[str, Any]], available_agents: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Create a plan for executing subtasks with selected agents.
        
        Args:
            subtasks: List of dictionaries describing subtasks
            available_agents: Dictionary mapping capability IDs to lists of agent IDs
            
        Returns:
            List of dictionaries describing the execution plan.
        """
        pass
        
    @abstractmethod
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the plan and return aggregated results.
        
        Args:
            execution_plan: List of dictionaries describing the execution plan
            
        Returns:
            Dictionary containing the aggregated results of plan execution.
        """
        pass
        
    @abstractmethod
    def handle_error(self, error: Dict[str, Any], execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle errors during plan execution.
        
        Args:
            error: Dictionary describing the error
            execution_plan: List of dictionaries describing the execution plan
            
        Returns:
            Dictionary containing error handling result.
        """
        pass


class IGatewayAgent(IAgent):
    """
    Interface for registry/gateway agent in DAWN architecture.
    
    The Gateway Agent is responsible for maintaining a registry of available
    agents and their capabilities, and facilitating agent discovery.
    """
    
    @abstractmethod
    def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """
        Register an agent in the registry.
        
        Args:
            agent_info: Dictionary containing agent information
            
        Returns:
            String ID of the registered agent.
        """
        pass
        
    @abstractmethod
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            Boolean indicating success or failure.
        """
        pass
        
    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Dictionary containing agent information, or None if not found.
        """
        pass
        
    @abstractmethod
    def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List agents matching optional filters.
        
        Args:
            filters: Optional dictionary of filter criteria
            
        Returns:
            List of dictionaries containing agent information.
        """
        pass
        
    @abstractmethod
    def find_agents_by_capability(self, capability_type: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find agents with a specific capability type and parameters.
        
        Args:
            capability_type: Type of capability to search for
            parameters: Optional parameters the capability should support
            
        Returns:
            List of dictionaries containing agent information.
        """
        pass
        
    @abstractmethod
    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Validate an agent's capabilities and accessibility.
        
        Args:
            agent_id: ID of the agent to validate
            
        Returns:
            Dictionary containing validation results.
        """
        pass


class IMessage(ABC):
    """
    Interface for standardized message format.
    
    Messages are the standardized format for communication between
    agents in the system.
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """
        Get unique message identifier.
        
        Returns:
            String ID of the message.
        """
        pass
        
    @abstractmethod
    def get_sender_id(self) -> str:
        """
        Get sender agent identifier.
        
        Returns:
            String ID of the sender agent.
        """
        pass
        
    @abstractmethod
    def get_recipient_id(self) -> str:
        """
        Get recipient agent identifier.
        
        Returns:
            String ID of the recipient agent.
        """
        pass
        
    @abstractmethod
    def get_content(self) -> Any:
        """
        Get message content.
        
        Returns:
            Content of the message.
        """
        pass
        
    @abstractmethod
    def get_conversation_id(self) -> str:
        """
        Get conversation identifier for related messages.
        
        Returns:
            String ID of the conversation.
        """
        pass
        
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get message metadata.
        
        Returns:
            Dictionary containing message metadata.
        """
        pass
        
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary representation.
        
        Returns:
            Dictionary representation of the message.
        """
        pass
        
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IMessage':
        """
        Create message from dictionary representation.
        
        Args:
            data: Dictionary representation of a message
            
        Returns:
            New message instance.
        """
        pass


class ITask(ABC):
    """
    Interface for defining units of work.
    
    Tasks represent units of work that can be executed by agents.
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """
        Get unique task identifier.
        
        Returns:
            String ID of the task.
        """
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """
        Get task description.
        
        Returns:
            String description of the task.
        """
        pass
        
    @abstractmethod
    def get_inputs(self) -> Dict[str, Any]:
        """
        Get task inputs.
        
        Returns:
            Dictionary containing task inputs.
        """
        pass
        
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get schema for expected task output.
        
        Returns:
            Dictionary containing the output schema.
        """
        pass
        
    @abstractmethod
    def get_required_capabilities(self) -> List[Dict[str, Any]]:
        """
        Get capabilities required for task execution.
        
        Returns:
            List of dictionaries describing required capabilities.
        """
        pass
        
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get IDs of dependent tasks.
        
        Returns:
            List of string IDs of dependent tasks.
        """
        pass
        
    @abstractmethod
    def get_status(self) -> str:
        """
        Get task status.
        
        Returns:
            String representing task status.
        """
        pass
        
    @abstractmethod
    def set_status(self, status: str) -> None:
        """
        Update task status.
        
        Args:
            status: New status string
        """
        pass
        
    @abstractmethod
    def set_result(self, result: Dict[str, Any]) -> None:
        """
        Set task result.
        
        Args:
            result: Dictionary containing task result
        """
        pass
        
    @abstractmethod
    def get_result(self) -> Optional[Dict[str, Any]]:
        """
        Get task result.
        
        Returns:
            Dictionary containing task result, or None if not complete.
        """
        pass