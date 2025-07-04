"""
Revenue Geography Agent - Main Entry Point
========================================

FastAPI server for revenue geography analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import RevenueGeographyAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Revenue Geography Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "geography_update_interval": int(os.getenv("GEOGRAPHY_UPDATE_INTERVAL", "3600")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_companies_per_cycle": int(os.getenv("MAX_COMPANIES_PER_CYCLE", "20"))
}

agent = RevenueGeographyAgent(config)

# Data models
class GeographyRequest(BaseModel):
    ticker_symbol: str
    analysis_type: Optional[str] = "comprehensive"

class GeographyResponse(BaseModel):
    ticker: str
    analysis_type: str
    geographic_data: Dict[str, Any]
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

@app.post("/geography/analyze")
async def analyze_geography(request: GeographyRequest):
    """Analyze revenue geography for a ticker"""
    try:
        geography_analysis = await agent.analyze_revenue_geography(
            request.ticker_symbol,
            request.analysis_type
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "geography_analysis": geography_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/geography/recent")
async def get_recent_geography(limit: int = 10):
    """Get recent geography analysis"""
    try:
        recent_geography = await agent.get_recent_geography(limit)
        
        return {
            "status": "success",
            "recent_geography": recent_geography,
            "count": len(recent_geography)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/geography/regions")
async def get_region_analysis(region: Optional[str] = None):
    """Get region-specific analysis"""
    try:
        region_analysis = await agent.get_region_analysis(region)
        
        return {
            "status": "success",
            "region_analysis": region_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "companies_analyzed": agent.companies_analyzed,
        "regions_tracked": agent.regions_tracked,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement RevenueGeographyAgent in agent.py for FactSet GeoRev API integration and recursive data parsing. 