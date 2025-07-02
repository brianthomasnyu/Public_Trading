# Fundamental Pricing Agent

## Purpose
Calculates intrinsic value, DCF, and relative valuation from financials, and stores pricing events in the knowledge base.

## How it works
- Calculates pricing models from financial data
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_pricing in agent.py
- Add real pricing model integration and parsing
- Expand MCP communication 