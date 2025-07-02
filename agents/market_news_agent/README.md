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

## **Market News Agent**
- **AI Reasoning for Data Existence**: Use GPT-4 to check if news is reporting the same event as existing knowledge base entries
- **Event Classification**: AI categorizes news by event type (earnings, regulatory, market movement, etc.)
- **Sentiment Analysis**: AI determines news sentiment and confidence level
- **Tool Selection**: AI chooses between NewsAPI, Benzinga, or Finnhub based on news type and urgency
- **Next Action Decision**: AI decides which agents to trigger based on news content (e.g., earnings news → event impact agent)
- **Impact Prediction**: AI predicts potential market impact of news events
- **Source Credibility**: AI evaluates news source reliability and adjusts processing accordingly 