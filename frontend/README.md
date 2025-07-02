# Frontend (Planned)

This folder will contain the web UI for the agentic trading/investment system.

## Intended Features
- Query input for users (natural language, ticker, or event)
- Visualization of event timeline and tagged data (interactive timeline, filters)
- Agent status dashboard (see which agents are running, last update, errors)
- Upload and search for documents (PDFs, research reports)
- KPI and event impact visualizations (charts, tables)
- User authentication (optional, for admin features)

## User Flow
1. User logs in (optional)
2. User submits a query or uploads a document
3. Orchestrator receives the query and dispatches to relevant agents
4. Agents process data, update the knowledge base, and tag events
5. Frontend displays results, timeline, and agent status in real time

## Main Components
- QueryBar: Input for user queries
- TimelineView: Interactive event timeline
- AgentStatusPanel: Shows agent health and activity
- DocumentUploader: Upload PDFs/reports
- KPIView: Visualizes KPIs and event impacts
- SearchBar: Search tagged data/events

## API Interaction
- All frontend/backend communication via REST API (FastAPI orchestrator)
- WebSocket (optional) for real-time updates

## Environment Variables
- `REACT_APP_API_URL`: Backend API endpoint

## Tech Stack
- React (TypeScript)
- REST API integration with orchestrator
- Dockerized deployment 