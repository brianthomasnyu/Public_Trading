"""
SEC Filings Agent - LangChain Enhanced Main Entry Point

This module starts the LangChain-enhanced SEC Filings Agent.
The agent is designed for data aggregation and analysis only.
NO TRADING DECISIONS are made by this agent.

ENHANCEMENT: Phase 1 - LangChain Integration
- LangChain Tool format for orchestrator integration
- Memory management for context persistence
- Tracing for comprehensive monitoring
- Preserved existing functionality
"""

import asyncio
import logging
from agent import SecFilingsAgent, SecFilingsAgentTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the LangChain-enhanced SEC Filings Agent"""
    logger.info("Starting LangChain Enhanced SEC Filings Agent")
    
    try:
        # Initialize the legacy agent for backward compatibility
        agent = SecFilingsAgent()
        
        # PSEUDOCODE: Initialize LangChain components
        # 1. Set up LangChain tracing
        # 2. Initialize memory management
        # 3. Configure tool registry
        # 4. Start monitoring
        
        logger.info("LangChain Enhanced SEC Filings Agent started successfully")
        
        # Run the agent
        await agent.run()
        
    except Exception as e:
        logger.error(f"Error starting SEC Filings Agent: {e}")
        raise

if __name__ == "__main__":
    # PSEUDOCODE: Start the LangChain-enhanced agent
    # 1. Initialize LangChain components
    # 2. Start the agent
    # 3. Handle graceful shutdown
    # 4. NO TRADING DECISIONS - only data aggregation
    
    asyncio.run(main()) 