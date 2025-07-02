import asyncio
from agent import RevenueGeographyAgent

if __name__ == "__main__":
    agent = RevenueGeographyAgent()
    asyncio.run(agent.run())

# Next step: Implement RevenueGeographyAgent in agent.py for FactSet GeoRev API integration and recursive data parsing. 