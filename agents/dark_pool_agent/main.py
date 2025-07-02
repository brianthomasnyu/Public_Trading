"""
Dark Pool Agent - Main Entry Point
=================================

FastAPI server for dark pool activity analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import DarkPoolAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Dark Pool Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "dark_pool_update_interval": int(os.getenv("DARK_POOL_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_activities_per_cycle": int(os.getenv("MAX_ACTIVITIES_PER_CYCLE", "50"))
}

agent = DarkPoolAgent(config)

# Data models
class DarkPoolActivityRequest(BaseModel):
    ticker_symbol: str
    date_range: Optional[str] = "7d"

class DarkPoolActivityResponse(BaseModel):
    activity_id: str
    ticker: str
    activity_type: str
    volume: int
    price: float
    timestamp: str
    source: str
    confidence: float

class UnusualActivityRequest(BaseModel):
    ticker_symbol: Optional[str] = None
    activity_threshold: Optional[float] = 0.5

class UnusualActivityResponse(BaseModel):
    ticker: str
    activity_type: str
    volume_ratio: float
    price_impact: float
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

@app.post("/dark-pool/activity")
async def get_dark_pool_activity(request: DarkPoolActivityRequest):
    """Get dark pool activity for a ticker"""
    try:
        activities = await agent.fetch_dark_pool_activity(request.ticker_symbol, request.date_range)
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "activities": activities,
            "count": len(activities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dark-pool/unusual-activity")
async def get_unusual_activity(request: UnusualActivityRequest):
    """Get unusual dark pool activity"""
    try:
        unusual_activities = await agent.detect_unusual_activity(
            request.ticker_symbol, 
            request.activity_threshold
        )
        
        return {
            "status": "success",
            "unusual_activities": unusual_activities,
            "count": len(unusual_activities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dark-pool/volume-analysis")
async def get_volume_analysis(ticker: Optional[str] = None):
    """Get volume analysis for dark pool activity"""
    try:
        volume_analysis = await agent.analyze_volume_patterns(ticker)
        
        return {
            "status": "success",
            "volume_analysis": volume_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dark-pool/price-impact")
async def get_price_impact_analysis(ticker: Optional[str] = None):
    """Get price impact analysis"""
    try:
        price_impact = await agent.analyze_price_impact(ticker)
        
        return {
            "status": "success",
            "price_impact": price_impact
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dark-pool/alerts")
async def get_dark_pool_alerts(ticker: Optional[str] = None):
    """Get dark pool alerts"""
    try:
        alerts = await agent.generate_dark_pool_alerts(ticker)
        
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dark-pool/statistics")
async def get_dark_pool_statistics():
    """Get dark pool statistics"""
    try:
        stats = await agent.get_dark_pool_statistics()
        
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "activities_tracked": agent.activities_tracked,
        "unusual_activities_detected": agent.unusual_activities_detected,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 