# Equity Research Agent

## Purpose
Fetches and parses equity research reports from APIs (TipRanks, Zacks, Seeking Alpha), extracts analyst ratings, price targets, and key insights, and stores them in the knowledge base.

## How it works
- Fetches data from research APIs
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_reports in agent.py
- Add real API integration and parsing
- Expand MCP communication

## **Equity Research Agent**
- **AI Reasoning for Data Existence**: Use GPT-4 to analyze if a research report's key insights already exist in the knowledge base, even if from different sources
- **Content Analysis**: Use AI to extract analyst ratings, price targets, and key insights from research reports
- **Relevance Assessment**: AI determines if a research report is relevant to current market conditions or specific tickers
- **Tool Selection**: AI chooses between TipRanks, Zacks, or Seeking Alpha APIs based on the type of research needed
- **Next Action Decision**: AI decides whether to trigger other agents (e.g., if research mentions debt concerns, trigger SEC filings agent)
- **Sentiment Analysis**: AI analyzes the tone and confidence level of analyst recommendations
- **Cross-Reference**: AI checks if research findings contradict or support existing knowledge base data 