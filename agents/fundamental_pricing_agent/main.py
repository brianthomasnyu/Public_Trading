"""
Fundamental Pricing Agent - Main Entry Point
==========================================

FastAPI server for fundamental pricing analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import FundamentalPricingAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Fundamental Pricing Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "pricing_update_interval": int(os.getenv("PRICING_UPDATE_INTERVAL", "3600")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_analyses_per_cycle": int(os.getenv("MAX_ANALYSES_PER_CYCLE", "20"))
}

agent = FundamentalPricingAgent(config)

# Data models
class PricingRequest(BaseModel):
    ticker_symbol: str
    model_type: Optional[str] = "dcf"

class PricingResponse(BaseModel):
    ticker: str
    model_type: str
    intrinsic_value: float
    confidence: float
    analysis: Dict[str, Any]
    timestamp: str

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

@app.post("/pricing/analyze")
async def analyze_pricing(request: PricingRequest):
    """Analyze fundamental pricing for a ticker"""
    try:
        pricing_analysis = await agent.analyze_fundamental_pricing(
            request.ticker_symbol,
            request.model_type
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "pricing_analysis": pricing_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pricing/recent")
async def get_recent_pricing(limit: int = 10):
    """Get recent pricing analysis"""
    try:
        recent_pricing = await agent.get_recent_pricing(limit)
        
        return {
            "status": "success",
            "recent_pricing": recent_pricing,
            "count": len(recent_pricing)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pricing/models")
async def get_pricing_models(ticker: Optional[str] = None):
    """Get available pricing models"""
    try:
        models = await agent.get_pricing_models(ticker)
        
        return {
            "status": "success",
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "analyses_performed": agent.analyses_performed,
        "models_used": agent.models_used,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
