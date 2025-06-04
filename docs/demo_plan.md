# DAWN/AGNTCY Demo Implementation Plan

This document outlines how the core interfaces will interact in a minimal viable demo, and specifies the unit tests that would demonstrate this functionality.

## Demo Scenario: Weather Summary Email Task

In this demo, we'll implement a scenario where a user asks the system to "Get the current weather for San Francisco and send me an email summary." This task requires multiple capabilities (weather data retrieval and email sending) that will be handled by specialized agents orchestrated by a Principal Agent.

## Component Interactions

### 1. Initial Task Submission

```
User → Principal Agent: Submit task "Get the current weather for San Francisco and send me an email summary"
```

### 2. Task Decomposition (Principal Agent)

```
Principal Agent: Decomposes task into subtasks:
- Subtask 1: Get weather data for San Francisco
- Subtask 2: Generate weather summary text
- Subtask 3: Send email with summary
```

### 3. Agent Discovery via Gateway Agent

```
Principal Agent → Gateway Agent: Find agents with "weather_data" capability
Gateway Agent → Principal Agent: Returns list of weather agent IDs

Principal Agent → Gateway Agent: Find agents with "text_generation" capability
Gateway Agent → Principal Agent: Returns list of text generation agent IDs

Principal Agent → Gateway Agent: Find agents with "email_sending" capability
Gateway Agent → Principal Agent: Returns list of email agent IDs
```

### 4. Execution Planning (Principal Agent)

```
Principal Agent: Creates execution plan:
- Step 1: Invoke WeatherAgent to get data (agent_id = "weather-agent-123")
- Step 2: Invoke TextAgent to generate summary (agent_id = "text-agent-456")
- Step 3: Invoke EmailAgent to send email (agent_id = "email-agent-789")
```

### 5. Task Execution via Agent Invocation

```
Principal Agent → WeatherAgent: Invoke "get_weather" capability with {location: "San Francisco"}
WeatherAgent → Principal Agent: Returns weather data {temperature: 72, conditions: "Sunny", humidity: 65}

Principal Agent → TextAgent: Invoke "generate_text" with {prompt: "Write a short weather summary", data: {temperature: 72, conditions: "Sunny", humidity: 65}}
TextAgent → Principal Agent: Returns summary text "Today in San Francisco: It's a sunny day with a temperature of 72°F and 65% humidity."

Principal Agent → EmailAgent: Invoke "send_email" with {to: "user@example.com", subject: "Weather Summary", body: "Today in San Francisco: It's a sunny day with a temperature of 72°F and 65% humidity."}
EmailAgent → Principal Agent: Returns confirmation {success: true, message_id: "abc123"}
```

### 6. Result Aggregation and Response

```
Principal Agent → User: Returns task completion status and summary
{
  status: "completed",
  summary: "Weather information retrieved and email sent successfully",
  details: {
    weather: {temperature: 72, conditions: "Sunny", humidity: 65},
    email_sent: true,
    email_id: "abc123"
  }
}
```

## Unit Test Plan

The following unit tests would demonstrate the functionality of our minimal viable demo:

### 1. Agent Interface Tests

```python
def test_agent_info_retrieval():
    """Test that agent properly returns its information and capabilities."""
    
def test_agent_capability_invocation():
    """Test that agent can be invoked with a specified capability."""
    
def test_agent_health_check():
    """Test that agent health check returns appropriate status."""
```

### 2. Principal Agent Tests

```python
def test_task_decomposition():
    """Test that Principal Agent correctly decomposes a complex task into subtasks."""
    
def test_agent_discovery():
    """Test that Principal Agent can discover agents with required capabilities."""
    
def test_execution_plan_creation():
    """Test that Principal Agent creates a valid execution plan for subtasks."""
    
def test_plan_execution():
    """Test that Principal Agent correctly executes a plan and aggregates results."""
    
def test_error_handling():
    """Test that Principal Agent properly handles errors during plan execution."""
```

### 3. Gateway Agent Tests

```python
def test_agent_registration():
    """Test that Gateway Agent can register a new agent."""
    
def test_agent_unregistration():
    """Test that Gateway Agent can unregister an existing agent."""
    
def test_agent_retrieval():
    """Test that Gateway Agent can retrieve information about a specific agent."""
    
def test_agent_listing():
    """Test that Gateway Agent can list all registered agents with filtering."""
    
def test_capability_based_discovery():
    """Test that Gateway Agent can find agents with specific capabilities."""
    
def test_agent_validation():
    """Test that Gateway Agent can validate an agent's capabilities and accessibility."""
```

### 4. Message Tests

```python
def test_message_creation():
    """Test that messages can be created with proper structure."""
    
def test_message_serialization():
    """Test that messages can be correctly serialized to and from dictionaries."""
    
def test_message_metadata():
    """Test that message metadata is properly handled."""
```

### 5. Task Tests

```python
def test_task_creation():
    """Test that tasks can be created with proper structure."""
    
def test_task_status_updates():
    """Test that task status can be properly updated."""
    
def test_task_result_handling():
    """Test that task results can be set and retrieved."""
    
def test_task_dependency_management():
    """Test that task dependencies are properly managed."""
```

### 6. Integration Tests

```python
def test_weather_email_workflow():
    """Test the complete workflow for the weather summary email scenario."""
    
def test_gateway_principal_interaction():
    """Test the interaction between Gateway Agent and Principal Agent."""
    
def test_principal_specialized_agent_interaction():
    """Test the interaction between Principal Agent and specialized agents."""
```

## Mock Implementations

For initial testing, we'll use mock implementations of the specialized agents:

### Mock Weather Agent

```python
class MockWeatherAgent(IAgent):
    """Mock implementation of a weather data agent."""
    
    def invoke(self, capability_id, inputs, config=None):
        """Return mock weather data regardless of input."""
        return {
            "temperature": 72,
            "conditions": "Sunny",
            "humidity": 65
        }
```

### Mock Text Generation Agent

```python
class MockTextAgent(IAgent):
    """Mock implementation of a text generation agent."""
    
    def invoke(self, capability_id, inputs, config=None):
        """Return mock generated text regardless of input."""
        weather = inputs.get("data", {})
        return {
            "text": f"Today in {inputs.get('location', 'San Francisco')}: It's a {weather.get('conditions', 'sunny')} day with a temperature of {weather.get('temperature', 72)}°F and {weather.get('humidity', 65)}% humidity."
        }
```

### Mock Email Agent

```python
class MockEmailAgent(IAgent):
    """Mock implementation of an email sending agent."""
    
    def invoke(self, capability_id, inputs, config=None):
        """Return mock email sending confirmation."""
        return {
            "success": True,
            "message_id": "abc123"
        }
```

## Next Steps

Once the interfaces and mock implementations are in place, we will:

1. Implement the concrete classes that fulfill these interfaces
2. Set up the HTTP endpoints for agent communication
3. Create the demo script that runs the weather email workflow
4. Add proper error handling and logging
5. Replace mock implementations with actual functionality when ready