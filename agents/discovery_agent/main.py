"""
Discovery Agent - Main Entry Point
=================================

FastAPI server for intelligent question generation and market investigation.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import DiscoveryAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Discovery Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "question_generation_interval": int(os.getenv("QUESTION_GENERATION_INTERVAL", "3600")),
    "max_questions_per_cycle": int(os.getenv("MAX_QUESTIONS_PER_CYCLE", "10")),
    "investigation_depth": int(os.getenv("INVESTIGATION_DEPTH", "3"))
}

agent = DiscoveryAgent(config)

# Data models
class QuestionGenerationRequest(BaseModel):
    market_context: Optional[str] = None
    focus_area: Optional[str] = None
    question_count: Optional[int] = 5

class QuestionResponse(BaseModel):
    question_id: str
    question: str
    category: str
    priority: str
    reasoning: str
    timestamp: str

class InvestigationRequest(BaseModel):
    question: str
    depth: Optional[int] = 3
    include_agents: Optional[List[str]] = None

class InvestigationResponse(BaseModel):
    investigation_id: str
    question: str
    findings: List[Dict[str, Any]]
    conclusions: List[str]
    next_questions: List[str]
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

@app.post("/questions/generate")
async def generate_questions(request: QuestionGenerationRequest):
    """Generate context-aware market questions"""
    try:
        questions = await agent.generate_market_questions(
            request.market_context,
            request.focus_area,
            request.question_count
        )
        
        return {
            "status": "success",
            "questions": questions,
            "count": len(questions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/investigation/start")
async def start_investigation(request: InvestigationRequest):
    """Start investigation for a specific question"""
    try:
        investigation = await agent.investigate_question(
            request.question,
            request.depth,
            request.include_agents
        )
        
        return {
            "status": "success",
            "investigation": investigation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/questions/active")
async def get_active_questions():
    """Get currently active questions being investigated"""
    try:
        active_questions = await agent.get_active_questions()
        
        return {
            "status": "success",
            "active_questions": active_questions,
            "count": len(active_questions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/investigations/recent")
async def get_recent_investigations(limit: int = 10):
    """Get recent investigations"""
    try:
        recent_investigations = await agent.get_recent_investigations(limit)
        
        return {
            "status": "success",
            "recent_investigations": recent_investigations,
            "count": len(recent_investigations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/context")
async def get_market_context():
    """Get current market context for question generation"""
    try:
        market_context = await agent.get_market_context()
        
        return {
            "status": "success",
            "market_context": market_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/questions/prioritize")
async def prioritize_questions(questions: List[str]):
    """Prioritize a list of questions"""
    try:
        prioritized_questions = await agent.prioritize_questions(questions)
        
        return {
            "status": "success",
            "prioritized_questions": prioritized_questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/questions/categories")
async def get_question_categories():
    """Get available question categories"""
    try:
        categories = await agent.get_question_categories()
        
        return {
            "status": "success",
            "categories": categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "questions_generated": agent.questions_generated,
        "investigations_completed": agent.investigations_completed,
        "insights_discovered": agent.insights_discovered,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 