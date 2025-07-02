# Data Tagging & Temporal Agent

## Purpose
Ensures all data is tagged (purpose, source, event type) and indexed by event time, and stores tag events in the knowledge base.

## How it works
- Tags and indexes all data/events
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_tags in agent.py
- Add real data tagging and timeline indexing
- Expand MCP communication 