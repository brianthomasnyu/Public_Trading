# Public Trading AI Agentic System

## Overview
This project is a modular, agentic AI system for autonomous financial data aggregation, analysis, and decision support for public equities. It leverages parallel, reasoning-capable agents that communicate via the Model Context Protocol (MCP) and update a central PostgreSQL knowledge base. The system is fully containerized with Docker for scalable deployment.

---

## Core Features

### 1. **Parallel Agent Execution**
- Each agent runs as a separate Docker service, enabling true parallelism.
- Orchestrator coordinates tasks and aggregates results.

### 2. **Model Context Protocol (MCP)**
- Standardized protocol for agent-to-agent (A2A) and orchestrator-to-agent communication.
- Agents can request data, trigger other agents, and update the knowledge base autonomously.

### 3. **Recursive Reasoning & Autonomy**
- Agents can reason about next steps, verify claims, and recursively call other agents as needed.
- Example: Social Media NLP Agent detects a claim, triggers SEC Filings Agent to verify, and updates the knowledge base.

### 4. **Event Timeline & Data Tagging**
- All data/events are indexed by time and tagged for context, source, and relevance.
- Enables advanced search, querying, and event-driven analysis.

### 5. **PostgreSQL Knowledge Base**
- Central store for all ingested, tagged, and indexed data.
- Supports event timeline, tagging, and agent queries/updates.

### 6. **Dockerized Microservices**
- Each agent, orchestrator, and the database run in isolated containers.
- Easy to scale, update, and maintain.

---

## Agent List & Intended Functionality

### 1. **Orchestrator Agent**
- Receives user queries, coordinates agent tasks, manages workflow, aggregates results.
- Handles agent-to-agent communication (MCP).

### 2. **Equity Research Agent**
- Ingests and parses equity research reports.
- Tags and indexes insights/events.

### 3. **SEC Filings & Financial Statements Agent**
- Parses 10-K, 10-Q, 8-K, and extracts financials.
- Tags data (e.g., "debt", "FCF", "IC") and indexes by time/event.

### 4. **Market News & Sentiment Agent**
- Scrapes/ingests news, performs sentiment analysis, tags news by event type.

### 5. **Insider Trading Alert Agent**
- Monitors insider transactions, flags unusual activity, tags by event.

### 6. **Social Media NLP Agent**
- Monitors Reddit, Twitter, StockTwits for relevant discussions.
- Extracts claims, sentiment, and triggers verification by other agents.

### 7. **Fundamental Pricing Agent**
- Calculates intrinsic value, DCF, and relative valuation from financials.
- Tags pricing events and valuation changes.

### 8. **KPI Tracker Agent**
- Extracts and tracks KPIs (e.g., P/E, Debt/Equity, FCF, ROIC) by ticker/industry.
- Monitors changes and tags KPI events.

### 9. **Event Impact Analysis Agent**
- Tracks stock price movement after events, compares to historical similar events, tags impact.

### 10. **Data Tagging & Temporal Agent**
- Ensures all data is tagged (purpose, source, event type) and indexed by event time.
- Supports advanced search and querying.

### 11. **Revenue-Geography Exposure Agent**
- Maps company sales by region, links geopolitical events to equity impact.

### 12. **Macro-Calendar Impact Agent**
- Tags macro surprises (CPI, NFP, FOMC) that move equities, indexes macro events and their impact.

### 13. **Options Flow & Volatility Agent**
- Detects unusual options flow, volatility events, tags early signals for price moves.

### 14. **ML Model Testing Agent**
- Downloads/tests ML models for prediction/classification, suggests/implements new models.

---

## Folder Structure
- `/agents/` - All agent microservices (one subfolder per agent)
- `/orchestrator/` - Orchestrator service
- `/db/` - PostgreSQL schema, migrations, and scripts
- `/shared/` - Common code, tool registry, utils
- `/frontend/` - (Optional) Web UI
- `.env` - Environment variables
- `docker-compose.yml` - Service orchestration
- `requirements.txt` - Python dependencies

---

## Next Steps
1. Scaffold all agent and service folders with placeholder files and comments.
2. Set up the PostgreSQL schema for event/timeline/tagged data.
3. Implement the MCP-based orchestrator and agent communication skeleton (no business logic yet).

---

## How to Use
- Build and run with Docker Compose.
- Configure API keys and DB credentials in `.env`.
- Extend each agent with business logic as needed. 