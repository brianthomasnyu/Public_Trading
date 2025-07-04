"""
Market News Agent - Main Entry Point
==================================

FastAPI server for market news analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import MarketNewsAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Market News Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "news_update_interval": int(os.getenv("NEWS_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_news_per_cycle": int(os.getenv("MAX_NEWS_PER_CYCLE", "50"))
}

agent = MarketNewsAgent(config)

# Data models
class NewsRequest(BaseModel):
    ticker_symbol: Optional[str] = None
    news_type: Optional[str] = "all"

class NewsResponse(BaseModel):
    news_items: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
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

@app.get("/news/latest")
async def get_latest_news(request: NewsRequest):
    """Get latest market news"""
    try:
        news_analysis = await agent.get_latest_news(
            request.ticker_symbol,
            request.news_type
        )
        
        return {
            "status": "success",
            "news_analysis": news_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/sentiment")
async def get_news_sentiment(ticker: str):
    """Get news sentiment analysis"""
    try:
        sentiment = await agent.analyze_news_sentiment(ticker)
        
        return {
            "status": "success",
            "ticker": ticker,
            "sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/alerts")
async def get_news_alerts():
    """Get news alerts"""
    try:
        alerts = await agent.generate_news_alerts()
        
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
        "news_processed": agent.news_processed,
        "sentiment_analyses": agent.sentiment_analyses,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 