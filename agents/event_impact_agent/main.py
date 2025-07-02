import asyncio
from agent import EventImpactAgent

if __name__ == "__main__":
    agent = EventImpactAgent()
    asyncio.run(agent.run())

# Next step: Implement EventImpactAgent in agent.py for event impact analysis and recursive data parsing. 