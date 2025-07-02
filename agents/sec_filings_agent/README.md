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

## **SEC Filings Agent**
- **AI Reasoning for Data Existence**: Use GPT-4 to check if financial metrics from new filings are materially different from existing data
- **Financial Metric Extraction**: AI extracts and normalizes financial data (debt, FCF, IC) from filing text
- **Anomaly Detection**: AI identifies unusual changes in financial metrics that warrant deeper analysis
- **Tool Selection**: AI chooses between SEC EDGAR, Financial Modeling Prep, or other APIs based on filing type
- **Next Action Decision**: AI decides if unusual metrics should trigger KPI tracker, fundamental pricing, or event impact agents
- **Trend Analysis**: AI analyzes patterns in financial metrics over time
- **Risk Assessment**: AI evaluates if financial changes indicate potential risks or opportunities 