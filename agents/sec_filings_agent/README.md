# SEC Filings Agent

## Purpose
Fetches and parses SEC filings (10-K, 10-Q, 8-K) from the SEC EDGAR API, extracts financial metrics, and stores them in the knowledge base.

## How it works
- Fetches data from SEC EDGAR API
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_filings in agent.py
- Add real SEC EDGAR API integration and parsing
- Expand MCP communication 