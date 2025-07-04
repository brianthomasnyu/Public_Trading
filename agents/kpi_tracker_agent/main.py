"""
KPI Tracker Agent - Main Entry Point
==================================

FastAPI server for KPI tracking and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import KPITrackerAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="KPI Tracker Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "kpi_update_interval": int(os.getenv("KPI_UPDATE_INTERVAL", "3600")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_kpis_per_cycle": int(os.getenv("MAX_KPIS_PER_CYCLE", "50"))
}

agent = KPITrackerAgent(config)

# Data models
class KPIRequest(BaseModel):
    ticker_symbol: str
    kpi_type: Optional[str] = "financial"

class KPIResponse(BaseModel):
    ticker: str
    kpi_type: str
    kpis: List[Dict[str, Any]]
    trends: Dict[str, Any]
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

@app.post("/kpi/track")
async def track_kpis(request: KPIRequest):
    """Track KPIs for a ticker"""
    try:
        kpi_analysis = await agent.track_kpis(
            request.ticker_symbol,
            request.kpi_type
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "kpi_analysis": kpi_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kpi/recent")
async def get_recent_kpis(limit: int = 10):
    """Get recent KPI tracking"""
    try:
        recent_kpis = await agent.get_recent_kpis(limit)
        
        return {
            "status": "success",
            "recent_kpis": recent_kpis,
            "count": len(recent_kpis)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kpi/alerts")
async def get_kpi_alerts():
    """Get KPI alerts"""
    try:
        alerts = await agent.generate_kpi_alerts()
        
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
        "kpis_tracked": agent.kpis_tracked,
        "alerts_generated": agent.alerts_generated,
        "trends_analyzed": agent.trends_analyzed,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
