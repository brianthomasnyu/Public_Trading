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

## **Insider Trading Agent**
- **AI Reasoning for Data Existence**: Use GPT-4 to check if insider trading patterns are similar to existing knowledge base data
- **Pattern Recognition**: AI identifies unusual insider trading patterns that might indicate significant events
- **Risk Assessment**: AI evaluates the significance of insider transactions based on position size and timing
- **Tool Selection**: AI chooses between OpenInsider, Finviz, or SEC Form 4 based on data freshness needs
- **Next Action Decision**: AI decides if unusual insider activity should trigger news or event impact agents
- **Context Analysis**: AI relates insider trading to recent company events or market conditions
- **Signal Strength**: AI determines the strength of insider trading signals 