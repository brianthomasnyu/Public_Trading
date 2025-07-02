import asyncio
from agent import OptionsFlowAgent

if __name__ == "__main__":
    agent = OptionsFlowAgent()
    asyncio.run(agent.run())

# Next step: Implement OptionsFlowAgent in agent.py for options data API integration and recursive data parsing. 