import asyncio
from agent import SocialMediaNLPAagent

if __name__ == "__main__":
    agent = SocialMediaNLPAagent()
    asyncio.run(agent.run())

# Next step: Implement SocialMediaNLPAagent in agent.py for social media API integration and recursive data parsing. 