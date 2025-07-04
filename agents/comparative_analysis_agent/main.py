"""
Comparative Analysis Agent - Main Entry Point
============================================

FastAPI server for comparative analysis operations with multi-tool integration.
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
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import langchain

# Computer Use Integration
try:
    from computer_use import ComputerUseToolSelector, ComputerUseOptimizer
    COMPUTER_USE_AVAILABLE = True
except ImportError:
    COMPUTER_USE_AVAILABLE = False
    ComputerUseToolSelector = None
    ComputerUseOptimizer = None

# LlamaIndex Integration
try:
    from llama_index import VectorStoreIndex, Document, ServiceContext
    from llama_index.llms import OpenAI as LlamaOpenAI
    from llama_index.embeddings import OpenAIEmbedding
    from llama_index.node_parser import SimpleNodeParser
    from llama_index.storage.storage_context import StorageContext
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    VectorStoreIndex = None
    Document = None
    ServiceContext = None

# Haystack Integration
try:
    from haystack import Pipeline
    from haystack.nodes import PreProcessor, EmbeddingRetriever, PromptNode
    from haystack.schema import Document as HaystackDocument
    from haystack.document_stores import InMemoryDocumentStore
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    Pipeline = None
    PreProcessor = None
    EmbeddingRetriever = None

# AutoGen Integration
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None
    UserProxyAgent = None

from agent import ComparativeAnalysisAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Comparative Analysis Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "analysis_depth": os.getenv("ANALYSIS_DEPTH", "detailed"),
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
    "max_entities_per_analysis": int(os.getenv("MAX_ENTITIES_PER_ANALYSIS", "20"))
}

agent = ComparativeAnalysisAgent(config)

# Data models
class PeerComparisonRequest(BaseModel):
    target_entity: str
    peer_group: Optional[List[str]] = None
    metrics: Optional[List[str]] = None

class SectorAnalysisRequest(BaseModel):
    sector: str
    metrics: Optional[List[str]] = None

class HistoricalComparisonRequest(BaseModel):
    entity: str
    time_period: str = "1y"
    metrics: Optional[List[str]] = None

class ComparisonResponse(BaseModel):
    request_id: str
    entities: List[str]
    metrics: List[str]
    analysis_type: str
    results: Dict[str, Any]
    insights: List[str]
    confidence: float
    data_quality: float

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    await agent.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint with multi-tool validation"""
    try:
        # Check multi-tool integration status
        multi_tool_status = {
            "langchain": agent.llm is not None,
            "computer_use": COMPUTER_USE_AVAILABLE and agent.tool_selector is not None,
            "llama_index": LLAMA_INDEX_AVAILABLE and agent.llama_index is not None,
            "haystack": HAYSTACK_AVAILABLE and agent.haystack_pipeline is not None,
            "autogen": AUTOGEN_AVAILABLE and agent.manager is not None
        }
        
        return {
            "status": "healthy",
            "agent": agent.name,
            "version": agent.version,
            "health_score": agent.health_score,
            "multi_tool_integration": multi_tool_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/analysis/peer", response_model=ComparisonResponse)
async def perform_peer_comparison(request: PeerComparisonRequest):
    """Perform peer comparison analysis"""
    try:
        result = await agent.perform_peer_comparison(
            target_entity=request.target_entity,
            peer_group=request.peer_group,
            metrics=request.metrics
        )
        
        return ComparisonResponse(
            request_id=result.request_id,
            entities=result.entities,
            metrics=result.metrics,
            analysis_type=result.analysis_type,
            results=result.results,
            insights=result.insights,
            confidence=result.confidence,
            data_quality=result.data_quality
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/sector", response_model=ComparisonResponse)
async def perform_sector_analysis(request: SectorAnalysisRequest):
    """Perform sector analysis"""
    try:
        result = await agent.perform_sector_analysis(
            sector=request.sector,
            metrics=request.metrics
        )
        
        return ComparisonResponse(
            request_id=result.request_id,
            entities=result.entities,
            metrics=result.metrics,
            analysis_type=result.analysis_type,
            results=result.results,
            insights=result.insights,
            confidence=result.confidence,
            data_quality=result.data_quality
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/historical", response_model=ComparisonResponse)
async def perform_historical_comparison(request: HistoricalComparisonRequest):
    """Perform historical comparison analysis"""
    try:
        result = await agent.perform_historical_comparison(
            entity=request.entity,
            time_period=request.time_period,
            metrics=request.metrics
        )
        
        return ComparisonResponse(
            request_id=result.request_id,
            entities=result.entities,
            metrics=result.metrics,
            analysis_type=result.analysis_type,
            results=result.results,
            insights=result.insights,
            confidence=result.confidence,
            data_quality=result.data_quality
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmarks")
async def get_benchmarks():
    """Get available benchmarks"""
    try:
        benchmarks = []
        for benchmark in agent.benchmarks.values():
            benchmarks.append({
                "benchmark_id": benchmark.benchmark_id,
                "name": benchmark.name,
                "category": benchmark.category,
                "last_updated": benchmark.last_updated.isoformat(),
                "confidence": benchmark.confidence
            })
        
        return {
            "status": "success",
            "benchmarks": benchmarks,
            "count": len(benchmarks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "comparisons_performed": agent.comparisons_performed,
        "insights_generated": agent.insights_generated,
        "benchmarks_updated": agent.benchmarks_updated,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 