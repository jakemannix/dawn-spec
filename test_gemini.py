"""
Simple test script for the Gemini agent.

This script is specifically designed to test the Gemini agent with minimal token usage
to avoid quota limits. It only requests a very short response (max_tokens=50) and uses
a simple prompt ("Say hi!") to verify that the agent and API are working correctly.

Usage:
    python test_gemini.py

Note: Even with these limits, you may still encounter quota errors if you've used up
your daily Gemini API quota. If that happens, try again later or use a different
agent implementation.
"""
import sys
import os
from src.gemini_agent import create_gemini_agent

# Create a simple Gemini agent with minimal token usage to avoid quota issues
agent = create_gemini_agent(
    name="Test Gemini Agent",
    description="Simple test agent for Gemini",
    model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=50  # Use very small limit to avoid quota issues
)

# Get agent info
info = agent.get_info()
print("Agent Info:")
print(f"- Name: {info['name']}")
print(f"- ID: {info['id']}")
print(f"- Model: {info['model']}")
print(f"- Provider: {info['provider']}")
print(f"- Capabilities: {len(info['capabilities'])}")

# Simple capability test with very short text
result = agent.invoke("text-generation", {"prompt": "Say hi!"})

# Check result
if "error" in result:
    print(f"\nError: {result['error']}")
else:
    print("\nGenerated Text:")
    print(result["content"])
    
    # Print metadata if available
    if "duration_seconds" in result:
        print(f"\nDuration: {result['duration_seconds']:.2f} seconds")
    if "usage" in result:
        print(f"Input Tokens: {result['usage'].get('input_tokens', 'N/A')}")
        print(f"Output Tokens: {result['usage'].get('output_tokens', 'N/A')}")