import asyncio
from agent import SecFilingsAgent

if __name__ == "__main__":
    agent = SecFilingsAgent()
    asyncio.run(agent.run())

# Next step: Implement SecFilingsAgent in agent.py for SEC EDGAR API integration and recursive data parsing. 