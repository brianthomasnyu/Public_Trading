"""
Insider Trading Agent - Main Entry Point
=======================================

FastAPI server for insider trading analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import InsiderTradingAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Insider Trading Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "trade_update_interval": int(os.getenv("TRADE_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_trades_per_cycle": int(os.getenv("MAX_TRADES_PER_CYCLE", "50"))
}

agent = InsiderTradingAgent(config)

# Data models
class TradeAnalysisRequest(BaseModel):
    ticker_symbol: str
    analysis_type: Optional[str] = "comprehensive"

class TradeAnalysisResponse(BaseModel):
    ticker: str
    analysis_type: str
    findings: List[Dict[str, Any]]
    patterns: List[str]
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

@app.post("/trades/analyze")
async def analyze_trades(request: TradeAnalysisRequest):
    """Analyze insider trading for a ticker"""
    try:
        trade_analysis = await agent.analyze_insider_trades(
            request.ticker_symbol,
            request.analysis_type
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "trade_analysis": trade_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/recent")
async def get_recent_trades(limit: int = 10):
    """Get recent insider trading analysis"""
    try:
        recent_trades = await agent.get_recent_trades(limit)
        
        return {
            "status": "success",
            "recent_trades": recent_trades,
            "count": len(recent_trades)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/patterns")
async def get_trading_patterns(ticker: Optional[str] = None):
    """Get insider trading patterns"""
    try:
        patterns = await agent.get_trading_patterns(ticker)
        
        return {
            "status": "success",
            "patterns": patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/sentiment")
async def get_trade_sentiment(ticker: str):
    """Get insider trading sentiment analysis"""
    try:
        sentiment = await agent.analyze_trade_sentiment(ticker)
        
        return {
            "status": "success",
            "ticker": ticker,
            "sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "trades_analyzed": agent.trades_analyzed,
        "patterns_detected": agent.patterns_detected,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
