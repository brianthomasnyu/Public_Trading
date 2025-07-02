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