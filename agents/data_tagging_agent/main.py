"""
Data Tagging Agent - Main Entry Point
====================================

FastAPI server for data tagging and timeline indexing.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import DataTaggingAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Data Tagging Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "tagging_update_interval": int(os.getenv("TAGGING_UPDATE_INTERVAL", "300")),
    "max_items_per_cycle": int(os.getenv("MAX_ITEMS_PER_CYCLE", "100"))
}

agent = DataTaggingAgent(config)

# Data models
class TaggingRequest(BaseModel):
    data_type: str
    content: str
    source: str

class TaggingResponse(BaseModel):
    item_id: str
    tags: List[str]
    categories: List[str]
    confidence: float
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

@app.post("/tagging/process")
async def process_tagging(request: TaggingRequest):
    """Process data tagging for content"""
    try:
        tagging_result = await agent.process_data_tagging(
            request.data_type,
            request.content,
            request.source
        )
        
        return {
            "status": "success",
            "tagging_result": tagging_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tagging/statistics")
async def get_tagging_statistics():
    """Get tagging statistics"""
    try:
        stats = await agent.get_tagging_statistics()
        
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tagging/categories")
async def get_tagging_categories():
    """Get available tagging categories"""
    try:
        categories = await agent.get_tagging_categories()
        
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
        "items_tagged": agent.items_tagged,
        "tags_generated": agent.tags_generated,
        "categories_used": agent.categories_used,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Next step: Implement DataTaggingAgent in agent.py for data tagging and timeline indexing. 