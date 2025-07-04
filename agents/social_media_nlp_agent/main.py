"""
Social Media NLP Agent - Main Entry Point
=======================================

FastAPI server for social media NLP analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import SocialMediaNLPAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Social Media NLP Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "nlp_update_interval": int(os.getenv("NLP_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_posts_per_cycle": int(os.getenv("MAX_POSTS_PER_CYCLE", "100"))
}

agent = SocialMediaNLPAgent(config)

# Data models
class NLPRequest(BaseModel):
    ticker_symbol: str
    platform: Optional[str] = "all"

class NLPResponse(BaseModel):
    ticker: str
    platform: str
    sentiment: Dict[str, Any]
    trends: List[Dict[str, Any]]
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

@app.post("/nlp/analyze")
async def analyze_sentiment(request: NLPRequest):
    """Analyze social media sentiment for a ticker"""
    try:
        sentiment_analysis = await agent.analyze_social_sentiment(
            request.ticker_symbol,
            request.platform
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "sentiment_analysis": sentiment_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nlp/trends")
async def get_social_trends(limit: int = 10):
    """Get social media trends"""
    try:
        trends = await agent.get_social_trends(limit)
        
        return {
            "status": "success",
            "trends": trends,
            "count": len(trends)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nlp/alerts")
async def get_nlp_alerts():
    """Get NLP alerts"""
    try:
        alerts = await agent.generate_nlp_alerts()
        
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
        "posts_analyzed": agent.posts_analyzed,
        "sentiment_analyses": agent.sentiment_analyses,
        "alerts_generated": agent.alerts_generated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement SocialMediaNLPAgent in agent.py for social media API integration and recursive data parsing. 