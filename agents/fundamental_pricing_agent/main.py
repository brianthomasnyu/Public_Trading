import asyncio
from agent import FundamentalPricingAgent

if __name__ == "__main__":
    agent = FundamentalPricingAgent()
    asyncio.run(agent.run())

# Next step: Implement FundamentalPricingAgent in agent.py for pricing model integration and recursive data parsing. 