"""
Short Interest Agent - Main Entry Point
======================================

FastAPI server for short interest analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import ShortInterestAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Short Interest Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "short_interest_update_interval": int(os.getenv("SHORT_INTEREST_UPDATE_INTERVAL", "3600")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_tickers_per_cycle": int(os.getenv("MAX_TICKERS_PER_CYCLE", "100"))
}

agent = ShortInterestAgent(config)

# Data models
class ShortInterestRequest(BaseModel):
    ticker_symbol: str

class ShortInterestResponse(BaseModel):
    ticker: str
    short_interest: int
    short_interest_ratio: float
    days_to_cover: float
    change_from_previous: float
    change_percentage: float
    timestamp: str
    source: str
    confidence: float

class ShortSqueezeRequest(BaseModel):
    ticker_symbol: Optional[str] = None
    threshold: Optional[float] = 0.5

class ShortSqueezeResponse(BaseModel):
    ticker: str
    squeeze_probability: float
    risk_factors: List[str]
    potential_catalysts: List[str]
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

@app.post("/short-interest/data")
async def get_short_interest_data(request: ShortInterestRequest):
    """Get short interest data for a ticker"""
    try:
        short_interest_data = await agent.fetch_short_interest_data(request.ticker_symbol)
        
        if not short_interest_data:
            raise HTTPException(status_code=404, detail="Short interest data not found")
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "short_interest_data": short_interest_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/short-interest/squeeze-analysis")
async def get_squeeze_analysis(request: ShortSqueezeRequest):
    """Get short squeeze analysis"""
    try:
        squeeze_analysis = await agent.analyze_short_squeeze_potential(
            request.ticker_symbol,
            request.threshold
        )
        
        return {
            "status": "success",
            "squeeze_analysis": squeeze_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/short-interest/high-ratio")
async def get_high_short_interest_ratio(threshold: float = 0.3):
    """Get stocks with high short interest ratio"""
    try:
        high_ratio_stocks = await agent.get_high_short_interest_ratio_stocks(threshold)
        
        return {
            "status": "success",
            "high_ratio_stocks": high_ratio_stocks,
            "count": len(high_ratio_stocks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/short-interest/trends")
async def get_short_interest_trends(ticker: Optional[str] = None):
    """Get short interest trends"""
    try:
        trends = await agent.analyze_short_interest_trends(ticker)
        
        return {
            "status": "success",
            "trends": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/short-interest/alerts")
async def get_short_interest_alerts(ticker: Optional[str] = None):
    """Get short interest alerts"""
    try:
        alerts = await agent.generate_short_interest_alerts(ticker)
        
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/short-interest/statistics")
async def get_short_interest_statistics():
    """Get short interest statistics"""
    try:
        stats = await agent.get_short_interest_statistics()
        
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/short-interest/borrow-fee")
async def get_borrow_fee_data(ticker: str):
    """Get borrow fee data for a ticker"""
    try:
        borrow_fee_data = await agent.fetch_borrow_fee_data(ticker)
        
        return {
            "status": "success",
            "ticker": ticker,
            "borrow_fee_data": borrow_fee_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "tickers_tracked": agent.tickers_tracked,
        "squeeze_analyses": agent.squeeze_analyses,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 