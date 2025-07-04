"""
Options Flow Agent - Main Entry Point
===================================

FastAPI server for options flow analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import OptionsFlowAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Options Flow Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "options_update_interval": int(os.getenv("OPTIONS_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_options_per_cycle": int(os.getenv("MAX_OPTIONS_PER_CYCLE", "100"))
}

agent = OptionsFlowAgent(config)

# Data models
class OptionsFlowRequest(BaseModel):
    ticker_symbol: str
    flow_type: Optional[str] = "all"

class OptionsFlowResponse(BaseModel):
    ticker: str
    flow_type: str
    unusual_activity: List[Dict[str, Any]]
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

@app.post("/options/flow")
async def get_options_flow(request: OptionsFlowRequest):
    """Get options flow for a ticker"""
    try:
        options_flow = await agent.get_options_flow(
            request.ticker_symbol,
            request.flow_type
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "options_flow": options_flow
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/options/unusual")
async def get_unusual_activity(limit: int = 10):
    """Get unusual options activity"""
    try:
        unusual_activity = await agent.get_unusual_activity(limit)
        
        return {
            "status": "success",
            "unusual_activity": unusual_activity,
            "count": len(unusual_activity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/options/alerts")
async def get_options_alerts():
    """Get options flow alerts"""
    try:
        alerts = await agent.generate_options_alerts()
        
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
        "options_analyzed": agent.options_analyzed,
        "unusual_activity_detected": agent.unusual_activity_detected,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement OptionsFlowAgent in agent.py for options data API integration and recursive data parsing. 