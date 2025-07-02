import asyncio
from agent import MarketNewsAgent

if __name__ == "__main__":
    agent = MarketNewsAgent()
    asyncio.run(agent.run())

# Next step: Implement MarketNewsAgent in agent.py for news API integration and recursive data parsing. 