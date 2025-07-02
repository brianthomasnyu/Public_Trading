"""
Equity Research Agent - Main Entry Point
=======================================

FastAPI server for equity research analysis and monitoring.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import EquityResearchAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Equity Research Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "research_update_interval": int(os.getenv("RESEARCH_UPDATE_INTERVAL", "3600")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_research_per_cycle": int(os.getenv("MAX_RESEARCH_PER_CYCLE", "20"))
}

agent = EquityResearchAgent(config)

# Data models
class ResearchRequest(BaseModel):
    ticker_symbol: str
    research_type: Optional[str] = "comprehensive"

class ResearchResponse(BaseModel):
    ticker: str
    research_type: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
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

@app.post("/research/analyze")
async def analyze_research(request: ResearchRequest):
    """Analyze equity research for a ticker"""
    try:
        research_analysis = await agent.analyze_equity_research(
            request.ticker_symbol,
            request.research_type
        )
        
        return {
            "status": "success",
            "ticker": request.ticker_symbol,
            "research_analysis": research_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/recent")
async def get_recent_research(limit: int = 10):
    """Get recent research analysis"""
    try:
        recent_research = await agent.get_recent_research(limit)
        
        return {
            "status": "success",
            "recent_research": recent_research,
            "count": len(recent_research)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/analysts")
async def get_analyst_ratings(ticker: Optional[str] = None):
    """Get analyst ratings and recommendations"""
    try:
        analyst_ratings = await agent.get_analyst_ratings(ticker)
        
        return {
            "status": "success",
            "analyst_ratings": analyst_ratings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/sentiment")
async def get_research_sentiment(ticker: str):
    """Get research sentiment analysis"""
    try:
        sentiment = await agent.analyze_research_sentiment(ticker)
        
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
        "research_analyzed": agent.research_analyzed,
        "analysts_tracked": agent.analysts_tracked,
        "sentiment_analyses": agent.sentiment_analyses,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement EquityResearchAgent in agent.py for API integration and recursive data parsing. 