# Event Impact Analysis Agent

## Purpose
Tracks stock price movement after events, compares to historical similar events, and stores impact events in the knowledge base.

## How it works
- Tracks event-driven price impacts
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_impacts in agent.py
- Add real event impact analysis and parsing
- Expand MCP communication 