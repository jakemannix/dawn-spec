"""
MCP Interoperability Demo

This demo shows how DAWN agents can communicate using the Model Context Protocol (MCP),
demonstrating the centralized intelligence paradigm where one agent uses another's
capabilities as discoverable tools.
"""
import asyncio
import logging
import sys
from typing import Dict, Any
from mcp import StdioServerParameters

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import DAWN components
from src.agent import MCPCapableAgent, Capability


class MockAnalyticsAgent(MCPCapableAgent):
    """
    A mock agent that provides data analytics capabilities via MCP.
    
    This agent demonstrates how DAWN capabilities can be exposed as MCP tools
    for consumption by other agents or LLMs.
    """
    
    def __init__(self, agent_name: str):
        super().__init__(
            name=agent_name,
            description=f"Mock analytics agent - {agent_name}",
            provider="DAWN Demo",
            version="1.0.0"
        )
        
        # Add analytics capabilities with schema validation
        self._add_analytics_capabilities()
        
    def _add_analytics_capabilities(self):
        """Add analytics capabilities to the agent."""
        
        # Data analysis capability with schema
        analysis_cap = Capability(
            capability_type="data_analysis",
            name="Data Analysis",
            description="Perform comprehensive data analysis on datasets",
            parameters={
                "analysis_id": {
                    "type": "string",
                    "description": "Unique identifier for the analysis",
                    "required": True
                },
                "dataset_info": {
                    "type": "object",
                    "description": "Information about the dataset",
                    "required": True
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["descriptive", "exploratory", "statistical", "predictive"],
                    "required": True
                },
                "output_format": {
                    "type": "string",
                    "description": "Desired output format",
                    "enum": ["json", "csv", "html"],
                    "default": "json"
                }
            },
            business_logic_schema="data_analysis"
        )
        self.add_capability(analysis_cap)
        
        # Statistics calculation capability
        stats_cap = Capability(
            capability_type="statistical_calculation",
            name="Statistical Calculation",
            description="Calculate statistical metrics for numerical data",
            parameters={
                "data": {
                    "type": "array",
                    "description": "Array of numerical values",
                    "required": True
                },
                "metrics": {
                    "type": "array",
                    "description": "Statistical metrics to calculate",
                    "items": {
                        "type": "string",
                        "enum": ["mean", "median", "mode", "std_dev", "variance", "min", "max", "range"]
                    },
                    "required": True
                }
            }
        )
        self.add_capability(stats_cap)
        
        # Visualization capability
        viz_cap = Capability(
            capability_type="data_visualization",
            name="Data Visualization",
            description="Generate visualizations for data analysis",
            parameters={
                "data": {
                    "type": "object",
                    "description": "Data to visualize",
                    "required": True
                },
                "chart_type": {
                    "type": "string",
                    "description": "Type of chart to generate",
                    "enum": ["bar", "line", "scatter", "histogram", "pie"],
                    "required": True
                },
                "title": {
                    "type": "string",
                    "description": "Chart title",
                    "default": "Data Visualization"
                }
            }
        )
        self.add_capability(viz_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> list[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"MockAnalyticsAgent invoking capability {capability_id} with inputs: {inputs}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        # Validate input payload if schema is configured
        if capability.business_logic_schema:
            validation_result = capability.validate_input_payload(inputs)
            if not validation_result.get("valid", False):
                return {
                    "error": "Input validation failed",
                    "validation_error": validation_result.get("error"),
                    "capability_id": capability_id
                }
                
        try:
            if capability.type == "data_analysis":
                return self._handle_data_analysis(inputs)
            elif capability.type == "statistical_calculation":
                return self._handle_statistical_calculation(inputs)
            elif capability.type == "data_visualization":
                return self._handle_data_visualization(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in capability {capability_id}: {e}")
            return {"error": str(e)}
            
    def _handle_data_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data analysis requests."""
        analysis_id = inputs.get("analysis_id", "")
        dataset_info = inputs.get("dataset_info", {})
        analysis_type = inputs.get("analysis_type", "descriptive")
        output_format = inputs.get("output_format", "json")
        
        # Mock data analysis results
        dataset_name = dataset_info.get("dataset_name", "Unknown Dataset")
        
        if analysis_type == "descriptive":
            results = {
                "summary": {
                    "dataset_name": dataset_name,
                    "total_records": 10000,
                    "columns": ["id", "value", "category", "timestamp"],
                    "data_types": {
                        "id": "integer",
                        "value": "float",
                        "category": "string",
                        "timestamp": "datetime"
                    }
                },
                "statistics": {
                    "value_column": {
                        "mean": 42.7,
                        "median": 41.2,
                        "std_dev": 8.3,
                        "min": 12.1,
                        "max": 89.4
                    }
                }
            }
        elif analysis_type == "exploratory":
            results = {
                "patterns": [
                    "Strong correlation between value and category",
                    "Seasonal trend in timestamp data",
                    "Outliers detected in 2.3% of records"
                ],
                "recommendations": [
                    "Consider feature engineering for seasonal patterns",
                    "Review outlier detection methodology",
                    "Investigate category distribution"
                ]
            }
        else:
            results = {
                "analysis_type": analysis_type,
                "status": "completed",
                "note": f"Mock {analysis_type} analysis results"
            }
            
        return {
            "analysis_id": analysis_id,
            "dataset_info": dataset_info,
            "analysis_type": analysis_type,
            "output_format": output_format,
            "results": results,
            "success": True
        }
        
    def _handle_statistical_calculation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistical calculation requests."""
        data = inputs.get("data", [])
        metrics = inputs.get("metrics", [])
        
        if not data:
            return {"error": "No data provided for statistical calculation"}
            
        if not metrics:
            return {"error": "No metrics specified for calculation"}
            
        # Calculate requested metrics
        results = {}
        
        try:
            import statistics
            
            for metric in metrics:
                if metric == "mean":
                    results[metric] = statistics.mean(data)
                elif metric == "median":
                    results[metric] = statistics.median(data)
                elif metric == "mode":
                    try:
                        results[metric] = statistics.mode(data)
                    except statistics.StatisticsError:
                        results[metric] = "No unique mode found"
                elif metric == "std_dev":
                    results[metric] = statistics.stdev(data) if len(data) > 1 else 0
                elif metric == "variance":
                    results[metric] = statistics.variance(data) if len(data) > 1 else 0
                elif metric == "min":
                    results[metric] = min(data)
                elif metric == "max":
                    results[metric] = max(data)
                elif metric == "range":
                    results[metric] = max(data) - min(data)
                else:
                    results[metric] = f"Unknown metric: {metric}"
                    
        except Exception as e:
            return {"error": f"Statistical calculation failed: {str(e)}"}
            
        return {
            "data_points": len(data),
            "requested_metrics": metrics,
            "results": results,
            "success": True
        }
        
    def _handle_data_visualization(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data visualization requests."""
        data = inputs.get("data", {})
        chart_type = inputs.get("chart_type", "bar")
        title = inputs.get("title", "Data Visualization")
        
        # Mock visualization generation
        chart_info = {
            "chart_type": chart_type,
            "title": title,
            "data_summary": {
                "data_points": len(data) if isinstance(data, list) else len(data.keys()) if isinstance(data, dict) else 0,
                "data_type": type(data).__name__
            },
            "chart_url": f"https://example.com/charts/{chart_type}/{hash(str(data))}.png",
            "chart_config": {
                "width": 800,
                "height": 600,
                "theme": "default"
            }
        }
        
        return {
            "chart_info": chart_info,
            "title": title,
            "chart_type": chart_type,
            "success": True,
            "note": "This is a mock visualization. In a real implementation, this would generate actual charts."
        }
        
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


class MockOrchestratorAgent(MCPCapableAgent):
    """
    A mock orchestrator agent that uses MCP tools from other agents.
    
    This agent demonstrates the centralized intelligence pattern where one agent
    maintains control and uses other agents' capabilities as tools.
    """
    
    def __init__(self, agent_name: str):
        super().__init__(
            name=agent_name,
            description=f"Mock orchestrator agent - {agent_name}",
            provider="DAWN Demo",
            version="1.0.0"
        )
        
        # Add orchestration capabilities
        self._add_orchestration_capabilities()
        
    def _add_orchestration_capabilities(self):
        """Add orchestration capabilities to the agent."""
        
        # Research workflow capability
        research_cap = Capability(
            capability_type="research_workflow",
            name="Research Workflow",
            description="Execute a research workflow using remote analytics tools",
            parameters={
                "query_id": {
                    "type": "string",
                    "description": "Unique identifier for the research query",
                    "required": True
                },
                "query_text": {
                    "type": "string",
                    "description": "The research question or query",
                    "required": True
                },
                "analytics_server_id": {
                    "type": "string",
                    "description": "ID of the analytics MCP server to use",
                    "required": True
                }
            },
            business_logic_schema="research_query"
        )
        self.add_capability(research_cap)
        
    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata including capabilities."""
        return self.to_dict()
        
    def get_capabilities(self) -> list[Dict[str, Any]]:
        """Return list of agent capabilities."""
        return [cap.to_dict() for cap in self.capabilities]
        
    def invoke(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke a specific capability with given inputs."""
        logger.info(f"MockOrchestratorAgent invoking capability {capability_id}")
        
        # For MCP workflows, we need async execution
        return {"error": "This capability requires async execution. Use invoke_async instead."}
        
    async def invoke_async(self, capability_id: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of invoke for capabilities that require MCP calls."""
        logger.info(f"MockOrchestratorAgent async invoking capability {capability_id}")
        
        # Find the capability
        capability = None
        for cap in self.capabilities:
            if cap.id == capability_id:
                capability = cap
                break
                
        if capability is None:
            return {"error": f"Capability {capability_id} not found"}
            
        try:
            if capability.type == "research_workflow":
                return await self._handle_research_workflow(inputs)
            else:
                return {"error": f"Unknown capability type: {capability.type}"}
                
        except Exception as e:
            logger.error(f"Error in async capability {capability_id}: {e}")
            return {"error": str(e)}
            
    async def _handle_research_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle research workflow requests by using MCP tools."""
        query_id = inputs.get("query_id", "")
        query_text = inputs.get("query_text", "")
        analytics_server_id = inputs.get("analytics_server_id", "")
        
        if not analytics_server_id:
            return {"error": "analytics_server_id is required"}
            
        try:
            # Discover available tools on the analytics server
            logger.info(f"Discovering MCP tools on server {analytics_server_id}")
            tools = await self.discover_mcp_tools(analytics_server_id)
            logger.info(f"Found {len(tools)} MCP tools")
            
            # Execute a research workflow using the discovered tools
            workflow_results = []
            
            # Step 1: Generate mock dataset for analysis
            mock_dataset = {
                "dataset_id": f"DS-{query_id}",
                "dataset_name": f"Research Data for: {query_text}",
                "format": "json",
                "row_count": 1000,
                "column_count": 5
            }
            
            # Step 2: Perform data analysis
            analysis_tool = None
            for tool in tools:
                if "data_analysis" in tool.get("name", "").lower():
                    analysis_tool = tool
                    break
                    
            if analysis_tool:
                analysis_inputs = {
                    "analysis_id": f"ANAL-{query_id}",
                    "dataset_info": mock_dataset,
                    "analysis_type": "exploratory",
                    "output_format": "json"
                }
                
                analysis_result = await self.call_mcp_tool(
                    analytics_server_id,
                    analysis_tool["name"],
                    analysis_inputs
                )
                workflow_results.append({
                    "step": "data_analysis",
                    "tool": analysis_tool["name"],
                    "result": analysis_result
                })
            else:
                workflow_results.append({
                    "step": "data_analysis",
                    "error": "Data analysis tool not found"
                })
                
            # Step 3: Calculate statistics on mock data
            stats_tool = None
            for tool in tools:
                if "statistical" in tool.get("name", "").lower():
                    stats_tool = tool
                    break
                    
            if stats_tool:
                mock_data = [42.1, 38.7, 45.2, 41.8, 39.9, 44.3, 40.5, 43.1, 37.8, 46.2]
                stats_inputs = {
                    "data": mock_data,
                    "metrics": ["mean", "median", "std_dev", "min", "max"]
                }
                
                stats_result = await self.call_mcp_tool(
                    analytics_server_id,
                    stats_tool["name"],
                    stats_inputs
                )
                workflow_results.append({
                    "step": "statistical_analysis",
                    "tool": stats_tool["name"],
                    "result": stats_result
                })
            else:
                workflow_results.append({
                    "step": "statistical_analysis",
                    "error": "Statistical calculation tool not found"
                })
                
            # Step 4: Generate visualization
            viz_tool = None
            for tool in tools:
                if "visualization" in tool.get("name", "").lower():
                    viz_tool = tool
                    break
                    
            if viz_tool:
                viz_inputs = {
                    "data": {"values": [42.1, 38.7, 45.2, 41.8, 39.9]},
                    "chart_type": "line",
                    "title": f"Research Results: {query_text}"
                }
                
                viz_result = await self.call_mcp_tool(
                    analytics_server_id,
                    viz_tool["name"],
                    viz_inputs
                )
                workflow_results.append({
                    "step": "visualization",
                    "tool": viz_tool["name"],
                    "result": viz_result
                })
            else:
                workflow_results.append({
                    "step": "visualization",
                    "error": "Visualization tool not found"
                })
                
            return {
                "query_id": query_id,
                "query_text": query_text,
                "analytics_server_id": analytics_server_id,
                "available_tools": [tool["name"] for tool in tools],
                "workflow_steps": len(workflow_results),
                "workflow_results": workflow_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in research workflow: {e}")
            return {"error": str(e)}
            
    def health_check(self) -> bool:
        """Return the health status of the agent."""
        return True


async def run_mcp_server_demo():
    """Run the MCP server demo."""
    logger.info("=== MCP Server Demo ===")
    
    # Create an analytics agent
    analytics_agent = MockAnalyticsAgent("AnalyticsService")
    
    logger.info(f"Created agent: {analytics_agent.name}")
    logger.info(f"Agent capabilities: {[cap.name for cap in analytics_agent.capabilities]}")
    
    try:
        # Start the agent as an MCP server
        logger.info("Starting MCP server...")
        mcp_server = analytics_agent.start_mcp_server("analytics-service")
        
        server_info = analytics_agent.get_mcp_server_info()
        logger.info(f"MCP server started: {server_info}")
        
        logger.info("MCP server is running. You can connect to it using the MCP client.")
        logger.info("Press Ctrl+C to stop the server.")
        
        # Run the MCP server
        await mcp_server.run()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping server...")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
    finally:
        analytics_agent.stop_mcp_server()
        logger.info("MCP server stopped")


async def run_mcp_client_demo():
    """Run the MCP client demo."""
    logger.info("=== MCP Client Demo ===")
    
    # Create an orchestrator agent that will act as an MCP client
    orchestrator_agent = MockOrchestratorAgent("ResearchOrchestrator")
    
    logger.info(f"Created orchestrator agent: {orchestrator_agent.name}")
    
    try:
        # Connect to the analytics MCP server
        server_params = StdioServerParameters(
            command="python",
            args=["examples/mcp_interop_demo.py", "server"]
        )
        
        server_id = "analytics-service"
        logger.info(f"Connecting to MCP server: {server_id}")
        
        connected = await orchestrator_agent.connect_to_mcp_server(server_id, server_params)
        
        if connected:
            logger.info("Successfully connected to MCP server")
            
            # Execute a research workflow
            workflow_inputs = {
                "query_id": "RQ-DEMO123",
                "query_text": "Analyze customer satisfaction trends in Q4 2023",
                "analytics_server_id": server_id
            }
            
            # Get the research workflow capability
            workflow_capability = None
            for cap in orchestrator_agent.capabilities:
                if cap.type == "research_workflow":
                    workflow_capability = cap
                    break
                    
            if workflow_capability:
                logger.info("Executing research workflow using MCP tools...")
                result = await orchestrator_agent.invoke_async(workflow_capability.id, workflow_inputs)
                
                if result.get("success"):
                    logger.info("Research workflow completed successfully!")
                    logger.info(f"Query: {result['query_text']}")
                    logger.info(f"Available tools: {result['available_tools']}")
                    logger.info(f"Workflow steps executed: {result['workflow_steps']}")
                    
                    # Show workflow step results
                    for i, step_result in enumerate(result['workflow_results']):
                        if 'error' in step_result:
                            logger.info(f"Step {i+1} ({step_result['step']}): ERROR - {step_result['error']}")
                        else:
                            logger.info(f"Step {i+1} ({step_result['step']}): SUCCESS - Tool: {step_result['tool']}")
                else:
                    logger.error(f"Research workflow failed: {result.get('error')}")
            else:
                logger.error("Research workflow capability not found")
        else:
            logger.error("Failed to connect to MCP server")
            
    except Exception as e:
        logger.error(f"Error running MCP client: {e}")


async def main():
    """Main function to run the demo."""
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        await run_mcp_client_demo()
    else:
        await run_mcp_server_demo()


if __name__ == "__main__":
    asyncio.run(main())