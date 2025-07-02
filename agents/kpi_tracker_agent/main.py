import asyncio
from agent import KPITrackerAgent

if __name__ == "__main__":
    agent = KPITrackerAgent()
    asyncio.run(agent.run())

# Next step: Implement KPITrackerAgent in agent.py for KPI extraction and recursive data parsing. 