# Insider Trading Agent

## Purpose
Fetches and parses insider trading data from APIs (OpenInsider, Finviz), flags unusual activity, and stores events in the knowledge base.

## How it works
- Fetches data from insider trading APIs
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_trades in agent.py
- Add real insider trading API integration and parsing
- Expand MCP communication 