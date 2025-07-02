# KPI Tracker Agent

## Purpose
Extracts and tracks KPIs (P/E, Debt/Equity, FCF, ROIC) by ticker/industry, and stores KPI events in the knowledge base.

## How it works
- Extracts and tracks KPIs from financial data
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_kpis in agent.py
- Add real KPI extraction and parsing
- Expand MCP communication 