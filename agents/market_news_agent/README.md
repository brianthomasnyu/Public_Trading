# Market News Agent

## Purpose
Fetches and parses market news from APIs (NewsAPI, Benzinga, Finnhub), performs sentiment analysis, and stores news events in the knowledge base.

## How it works
- Fetches data from news APIs
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_news in agent.py
- Add real news API integration and parsing
- Expand MCP communication 