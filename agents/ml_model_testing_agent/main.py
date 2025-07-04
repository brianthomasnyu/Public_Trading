"""
ML Model Testing Agent - Main Entry Point
========================================

FastAPI server for ML model testing and validation with multi-tool integration.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

# Multi-Tool Integration Imports
from langchain.tracing import LangChainTracer
from llama_index import VectorStoreIndex
from haystack import Pipeline
import autogen

from agent import MLModelTestingAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="ML Model Testing Agent", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = MLModelTestingAgent()

# Data models
class ModelTestRequest(BaseModel):
    model_id: str
    test_type: str
    test_data: Optional[Dict[str, Any]] = None

class ResearchPaperRequest(BaseModel):
    query: str
    max_papers: int = 10

class ModelTestResponse(BaseModel):
    test_id: str
    model_id: str
    test_type: str
    results: Dict[str, float]
    confidence_score: float
    recommendations: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    await agent.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint with multi-tool integration status"""
    try:
        metrics = agent.get_metrics()
        return {
            "status": "healthy",
            "agent": agent.agent_name,
            "version": "2.0.0",
            "health_score": agent.health_score,
            "multi_tool_integration": metrics.get("multi_tool_integration", {}),
            "last_update": metrics.get("last_update"),
            "error_count": metrics.get("error_count", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "agent": agent.agent_name,
            "version": "2.0.0",
            "health_score": 0.0,
            "error": str(e)
        }

@app.post("/model/test", response_model=ModelTestResponse)
async def test_model(request: ModelTestRequest):
    """Test ML model performance"""
    try:
        if request.test_type == "performance":
            test_result = await agent.test_model_performance(request.model_id, request.test_data or {})
        elif request.test_type == "robustness":
            test_result = await agent.test_model_robustness(request.model_id, request.test_data or {})
        elif request.test_type == "drift":
            test_result = await agent.detect_model_drift(request.model_id, request.test_data or {})
        else:
            raise HTTPException(status_code=400, detail="Invalid test type")
        
        return ModelTestResponse(
            test_id=test_result.test_id,
            model_id=test_result.model_id,
            test_type=test_result.test_type,
            results=test_result.results,
            confidence_score=test_result.confidence_score,
            recommendations=test_result.recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/papers")
async def analyze_research_papers(request: ResearchPaperRequest):
    """Analyze research papers for ML model insights"""
    try:
        papers = await agent.parse_research_papers(request.query, request.max_papers)
        
        return {
            "status": "success",
            "query": request.query,
            "papers": papers,
            "count": len(papers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/list")
async def list_models():
    """List available models for testing"""
    try:
        models = await agent.select_models_for_testing()
        
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent performance metrics"""
    try:
        return agent.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement MLModelTestingAgent in agent.py for ML model integration and recursive data parsing. 