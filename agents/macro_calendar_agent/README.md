# Macro-Calendar Impact Agent

## Purpose
Tags macro surprises (CPI, NFP, FOMC) that move equities using FRED/Trading Economics APIs, and stores macro events in the knowledge base.

## How it works
- Tags macro calendar events
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_macro in agent.py
- Add real FRED/Trading Economics API integration and parsing
- Expand MCP communication 