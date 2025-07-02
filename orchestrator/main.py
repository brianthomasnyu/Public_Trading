import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env
load_dotenv()

# Database connection setup
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('POSTGRES_PORT')
DB_NAME = os.getenv('POSTGRES_DB')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# FastAPI app
app = FastAPI(title="Orchestrator Service")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Model for user queries
class QueryRequest(BaseModel):
    query: str
    ticker: str = None
    event: str = None

# Endpoint to receive user queries
@app.post("/query")
def receive_query(request: QueryRequest):
    # Future: Dispatch to relevant agents, aggregate results
    return {"message": "Query received", "query": request.query, "ticker": request.ticker, "event": request.event}

# Model for MCP messages
class MCPMessage(BaseModel):
    sender: str
    recipient: str
    content: dict
    context: dict = None

# Endpoint for agent-to-agent (A2A) and agent-to-orchestrator communication
@app.post("/mcp")
def mcp_endpoint(message: MCPMessage):
    # Future: Route message to appropriate agent/service
    return {"message": "MCP message received", "sender": message.sender, "recipient": message.recipient}

# --- New Endpoints ---

# GET /timeline: Returns a list of recent events (mock data for now)
@app.get("/timeline")
def get_timeline():
    # Future: Query the events table in the database
    mock_events = [
        {"id": 1, "event_time": "2024-07-01T12:00:00Z", "source_agent": "sec_filings_agent", "event_type": "sec_filing", "ticker": "AAPL", "tags": ["10-K", "debt"], "summary": "Apple 10-K filed, debt updated."},
        {"id": 2, "event_time": "2024-07-01T13:00:00Z", "source_agent": "market_news_agent", "event_type": "news", "ticker": "TSLA", "tags": ["earnings", "sentiment:positive"], "summary": "Tesla beats earnings expectations."}
    ]
    return {"events": mock_events}

# GET /agents/status: Returns the status of all agents (mock data for now)
@app.get("/agents/status")
def get_agents_status():
    # Future: Query the agent_status table in the database
    mock_status = [
        {"agent_name": "sec_filings_agent", "status": "online", "last_run": "2024-07-01T12:05:00Z"},
        {"agent_name": "market_news_agent", "status": "online", "last_run": "2024-07-01T13:05:00Z"},
        {"agent_name": "kpi_tracker_agent", "status": "offline", "last_run": "2024-06-30T23:00:00Z"}
    ]
    return {"agents": mock_status}

# Future: Add endpoints for event timeline, agent status, and knowledge base queries 