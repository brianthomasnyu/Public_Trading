import asyncio
from agent import DataTaggingAgent

if __name__ == "__main__":
    agent = DataTaggingAgent()
    asyncio.run(agent.run())

# Next step: Implement DataTaggingAgent in agent.py for data tagging and timeline indexing. 