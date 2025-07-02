import asyncio
from agent import EquityResearchAgent

if __name__ == "__main__":
    agent = EquityResearchAgent()
    asyncio.run(agent.run())

# Next step: Implement EquityResearchAgent in agent.py for API integration and recursive data parsing. 