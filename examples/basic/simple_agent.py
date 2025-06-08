"""
Simple Agent Example

This example shows how to use the Dawn framework to run agents
with different protocol configurations.
"""

import asyncio
import os
from dawn.runners.base import BaseDemoRunner


async def main():
    """Run a simple agent demo."""
    
    # Configure logging level via environment
    os.environ['DAWN_LOG_LEVEL'] = 'INFO'
    
    # Create runner with GitHub agent and ACP protocol
    runner = BaseDemoRunner(
        agents=['github'],
        protocols=['acp'],
        log_level='INFO'
    )
    
    try:
        # Start agents
        await runner.start()
        
        # Run a demo sequence
        demo_commands = [
            ('github', 'search for popular Python web frameworks'),
            ('github', 'analyze the django/django repository'),
        ]
        
        await runner.run_demo_sequence(demo_commands)
        
        # Optional: Run interactive chat
        # await runner.chat_loop()
        
    finally:
        # Clean up
        await runner.stop()


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the demo
    asyncio.run(main()) 