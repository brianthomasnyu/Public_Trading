"""
Event Impact Agent - Main Entry Point
====================================

FastAPI server for event impact analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import EventImpactAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Event Impact Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "impact_update_interval": int(os.getenv("IMPACT_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_events_per_cycle": int(os.getenv("MAX_EVENTS_PER_CYCLE", "50"))
}

agent = EventImpactAgent(config)

# Data models
class EventImpactRequest(BaseModel):
    event_type: str
    ticker_symbol: Optional[str] = None
    event_data: Dict[str, Any]

class EventImpactResponse(BaseModel):
    event_id: str
    impact_score: float
    affected_sectors: List[str]
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

@app.post("/impact/analyze")
async def analyze_event_impact(request: EventImpactRequest):
    """Analyze impact of an event"""
    try:
        impact_analysis = await agent.analyze_event_impact(
            request.event_type,
            request.ticker_symbol,
            request.event_data
        )
        
        return {
            "status": "success",
            "impact_analysis": impact_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/impact/recent")
async def get_recent_impacts(limit: int = 10):
    """Get recent impact analyses"""
    try:
        recent_impacts = await agent.get_recent_impacts(limit)
        
        return {
            "status": "success",
            "recent_impacts": recent_impacts,
            "count": len(recent_impacts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/impact/alerts")
async def get_impact_alerts():
    """Get impact alerts"""
    try:
        alerts = await agent.generate_impact_alerts()
        
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "events_analyzed": agent.events_analyzed,
        "impacts_calculated": agent.impacts_calculated,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement EventImpactAgent in agent.py for event impact analysis and recursive data parsing. 