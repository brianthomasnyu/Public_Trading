"""
Macro Calendar Agent - Main Entry Point
=====================================

FastAPI server for macro calendar analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import MacroCalendarAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Macro Calendar Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "calendar_update_interval": int(os.getenv("CALENDAR_UPDATE_INTERVAL", "3600")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_events_per_cycle": int(os.getenv("MAX_EVENTS_PER_CYCLE", "20"))
}

agent = MacroCalendarAgent(config)

# Data models
class CalendarRequest(BaseModel):
    event_type: Optional[str] = None
    date_range: Optional[str] = None

class CalendarResponse(BaseModel):
    events: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    timestamp: str
    confidence: float

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    await agent.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": agent.name,
        "version": agent.version,
        "health_score": agent.health_score
    }

@app.get("/calendar/events")
async def get_calendar_events(request: CalendarRequest):
    """Get macro calendar events"""
    try:
        events = await agent.get_calendar_events(
            request.event_type,
            request.date_range
        )
        
        return {
            "status": "success",
            "events": events
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar/upcoming")
async def get_upcoming_events(limit: int = 10):
    """Get upcoming macro events"""
    try:
        upcoming_events = await agent.get_upcoming_events(limit)
        
        return {
            "status": "success",
            "upcoming_events": upcoming_events,
            "count": len(upcoming_events)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar/impact")
async def get_event_impact(event_id: str):
    """Get impact analysis for a specific event"""
    try:
        impact = await agent.analyze_event_impact(event_id)
        
        return {
            "status": "success",
            "event_id": event_id,
            "impact": impact
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "events_tracked": agent.events_tracked,
        "impacts_analyzed": agent.impacts_analyzed,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
