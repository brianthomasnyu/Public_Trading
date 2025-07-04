"""
Repository Management Agent - Main Entry Point
=============================================

FastAPI server for repository management operations with multi-tool integration.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
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

from agent import RepositoryManagementAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Repository Management Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "repo_path": os.getenv("REPO_PATH", "."),
    "auto_commit": os.getenv("AUTO_COMMIT", "true").lower() == "true",
    "auto_push": os.getenv("AUTO_PUSH", "false").lower() == "true",
    "branch_protection": os.getenv("BRANCH_PROTECTION", "true").lower() == "true"
}

agent = RepositoryManagementAgent(config)

# Data models
class RepositoryStatusRequest(BaseModel):
    pass

class RepositoryStatusResponse(BaseModel):
    status: str
    branch: str
    last_commit: str
    uncommitted_changes: int
    health_score: float

class CodeChangeRequest(BaseModel):
    file_path: str
    change_type: str
    description: str
    priority: int = 1

class CodeChangeResponse(BaseModel):
    change_id: str
    status: str
    message: str

class CommitRequest(BaseModel):
    commit_message: str
    files: Optional[list] = None

class CommitResponse(BaseModel):
    success: bool
    commit_hash: Optional[str] = None
    message: str

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

@app.post("/status", response_model=RepositoryStatusResponse)
async def get_repository_status(request: RepositoryStatusRequest):
    """Get repository status"""
    try:
        status = await agent.get_repository_status()
        return RepositoryStatusResponse(
            status="success",
            branch=status.branch,
            last_commit=status.last_commit,
            uncommitted_changes=status.uncommitted_changes,
            health_score=status.health_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/changes/detect")
async def detect_changes():
    """Detect code changes in repository"""
    try:
        changes = await agent.detect_code_changes()
        return {
            "status": "success",
            "changes_detected": len(changes),
            "changes": [
                {
                    "change_id": change.change_id,
                    "file_path": change.file_path,
                    "change_type": change.change_type,
                    "description": change.description,
                    "priority": change.priority
                }
                for change in changes
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/commit", response_model=CommitResponse)
async def commit_changes(request: CommitRequest):
    """Commit changes to repository"""
    try:
        success = await agent.commit_changes(request.commit_message, request.files)
        return CommitResponse(
            success=success,
            message="Changes committed successfully" if success else "Failed to commit changes"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/push")
async def push_changes():
    """Push changes to remote repository"""
    try:
        success = await agent.push_changes()
        return {
            "status": "success" if success else "failed",
            "message": "Changes pushed successfully" if success else "Failed to push changes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/branch/create")
async def create_branch(branch_name: str, base_branch: str = None):
    """Create a new branch"""
    try:
        success = await agent.create_branch(branch_name, base_branch)
        return {
            "status": "success" if success else "failed",
            "message": f"Branch {branch_name} created successfully" if success else "Failed to create branch"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/maintenance")
async def run_maintenance():
    """Run repository maintenance tasks"""
    try:
        results = await agent.run_repository_maintenance()
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "changes_processed": agent.changes_processed,
        "commits_made": agent.commits_made,
        "branches_created": agent.branches_created,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 