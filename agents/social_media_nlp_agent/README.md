# Social Media NLP Agent

## Purpose
Fetches and parses social media posts (Twitter, Reddit, StockTwits), extracts claims and sentiment, and stores events in the knowledge base.

## How it works
- Fetches data from social media APIs
- Extracts claims and sentiment using NLP
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_posts in agent.py
- Add real social media API integration and NLP parsing
- Expand MCP communication 