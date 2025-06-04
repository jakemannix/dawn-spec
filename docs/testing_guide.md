# Testing Guide for DAWN Implementation

This guide provides instructions for testing the DAWN architecture implementation with various levels of API access.

## Prerequisites

Before testing, ensure you have:

1. Set up the environment as described in the README
2. Copied `template.env` to `.env` and configured your API keys
3. Installed the project: `pip install -e .`

## Testing Levels

### Level 1: No API Keys Required

These tests don't require any API keys and use mock implementations:

```bash
# Test the weather email demo
python examples/weather_email_demo.py

# Test via CLI
python dawn_cli.py run-demo weather-email
```

### Level 2: Minimal API Usage

These tests use minimal API tokens to verify functionality without excessive quota usage:

```bash
# Test OpenAI with diagnostics (minimal tokens)
python examples/openai_agent_demo.py diagnostics

# Test Anthropic with minimal token usage
python dawn_cli.py invoke anthropic text-generation --prompt "Hello" --config '{"max_tokens": 10}'

# Test Gemini with minimal token usage
python test_gemini.py
```

### Level 3: Full Functionality Testing

These tests use the full functionality but may consume more API quota:

```bash
# Test OpenAI agent
python dawn_cli.py run-demo openai

# Test Anthropic agent
python dawn_cli.py run-demo anthropic

# Test Gemini agent
python dawn_cli.py run-demo gemini

# Test research agent (uses all available APIs)
python dawn_cli.py run-demo research
```

## Component Testing

### Testing Core Interfaces

The implementation includes tests for the core interfaces:

```bash
# Run interface tests
python -m pytest tests/test_interfaces.py
```

### Testing Anthropic Agent

Tests for the Anthropic agent implementation:

```bash
# Run Anthropic agent tests
python -m pytest tests/test_anthropic_agent.py
```

### Testing Gateway Agent

Test the AGP gateway implementation:

```bash
# Run the AGP gateway demo
python examples/agp_gateway_demo.py
```

## API Health Checks

You can verify API access with the health check functionality:

```bash
# Check OpenAI API access
python dawn_cli.py invoke openai health-check

# Check Anthropic API access
python dawn_cli.py invoke anthropic health-check

# Check Gemini API access (note: may fail due to quota limits)
python dawn_cli.py invoke gemini health-check
```

## Common Testing Issues

### Gemini API Quota

Gemini has strict quota limits for free tier users. To work around this:

1. Set a low `max_tokens` value (e.g., 50) when testing
2. Use a different provider as your principal agent
3. Wait several hours between testing sessions
4. Run the minimal test with `test_gemini.py`

### OpenAI API Issues

If you encounter issues with the OpenAI API:

1. Verify your API key is correct
2. Check that your API key has sufficient quota
3. Run the diagnostics mode: `python examples/openai_agent_demo.py diagnostics`

### Anthropic API Issues

For issues with the Anthropic API:

1. Ensure your API key is in the correct format
2. Check if your API key has expired
3. Use a lower max_tokens value for testing

## Protocol Buffer Testing

If you modify the protocol buffer definitions, test them by:

1. Regenerating the code: `python generate_protos.py`
2. Running the ACP agent example: `python examples/acp_agent_example.py`
3. Running the AGP gateway demo: `python examples/agp_gateway_demo.py`

## Test Coverage

The project includes test coverage reporting:

```bash
# Run tests with coverage report
python -m pytest --cov=src tests/
```

## Continuous Testing During Development

During development, you might want to:

1. Use mock implementations where possible to avoid API costs
2. Test with the CLI for quick feedback
3. Use minimal token limits for API calls
4. Focus on testing one component at a time