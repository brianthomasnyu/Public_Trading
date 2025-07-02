import asyncio
from agent import MLModelTestingAgent

if __name__ == "__main__":
    agent = MLModelTestingAgent()
    asyncio.run(agent.run())

# Next step: Implement MLModelTestingAgent in agent.py for ML model integration and recursive data parsing. 