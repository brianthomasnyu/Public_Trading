import asyncio
from agent import MacroCalendarAgent

if __name__ == "__main__":
    agent = MacroCalendarAgent()
    asyncio.run(agent.run())

# Next step: Implement MacroCalendarAgent in agent.py for FRED/Trading Economics API integration and recursive data parsing. 