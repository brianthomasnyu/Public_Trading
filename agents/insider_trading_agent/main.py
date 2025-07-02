import asyncio
from agent import InsiderTradingAgent

if __name__ == "__main__":
    agent = InsiderTradingAgent()
    asyncio.run(agent.run())

# Next step: Implement InsiderTradingAgent in agent.py for insider trading API integration and recursive data parsing. 