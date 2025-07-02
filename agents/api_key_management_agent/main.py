"""
API Key Management Agent - Main Entry Point
==========================================

FastAPI server for secure credential management operations.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

from agent import APIKeyManagementAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="API Key Management Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "encryption_key_file": os.getenv("ENCRYPTION_KEY_FILE", "master.key"),
    "credentials_file": os.getenv("CREDENTIALS_FILE", "credentials.json.enc"),
    "backup_enabled": os.getenv("BACKUP_ENABLED", "true").lower() == "true"
}

agent = APIKeyManagementAgent(config)

# Data models
class StoreCredentialRequest(BaseModel):
    name: str
    credential_type: str
    value: str
    provider: str
    description: str = ""
    expires_at: Optional[str] = None
    tags: Optional[List[str]] = None

class StoreCredentialResponse(BaseModel):
    success: bool
    message: str
    credential_id: Optional[str] = None

class RetrieveCredentialRequest(BaseModel):
    name: str
    access_key: Optional[str] = None

class RetrieveCredentialResponse(BaseModel):
    success: bool
    value: Optional[str] = None
    message: str

class RotateCredentialRequest(BaseModel):
    name: str
    new_value: Optional[str] = None

class RotateCredentialResponse(BaseModel):
    success: bool
    message: str

class RevokeCredentialRequest(BaseModel):
    name: str

class RevokeCredentialResponse(BaseModel):
    success: bool
    message: str

class ListCredentialsRequest(BaseModel):
    filter_type: Optional[str] = None
    include_inactive: bool = False

class SecurityAuditResponse(BaseModel):
    timestamp: str
    total_credentials: int
    active_credentials: int
    expired_credentials: List[str]
    weak_credentials: List[str]
    security_score: float
    recommendations: List[str]

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

@app.post("/credentials/store", response_model=StoreCredentialResponse)
async def store_credential(request: StoreCredentialRequest):
    """Store a new credential securely"""
    try:
        success = await agent.store_credential(
            name=request.name,
            credential_type=request.credential_type,
            value=request.value,
            provider=request.provider,
            description=request.description,
            expires_at=request.expires_at,
            tags=request.tags
        )
        
        return StoreCredentialResponse(
            success=success,
            message="Credential stored successfully" if success else "Failed to store credential"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/credentials/retrieve", response_model=RetrieveCredentialResponse)
async def retrieve_credential(request: RetrieveCredentialRequest):
    """Retrieve a credential value"""
    try:
        value = await agent.retrieve_credential(request.name, request.access_key)
        
        return RetrieveCredentialResponse(
            success=value is not None,
            value=value,
            message="Credential retrieved successfully" if value else "Credential not found or access denied"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/credentials/rotate", response_model=RotateCredentialResponse)
async def rotate_credential(request: RotateCredentialRequest):
    """Rotate a credential"""
    try:
        success = await agent.rotate_credential(request.name, request.new_value)
        
        return RotateCredentialResponse(
            success=success,
            message="Credential rotated successfully" if success else "Failed to rotate credential"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/credentials/revoke", response_model=RevokeCredentialResponse)
async def revoke_credential(request: RevokeCredentialRequest):
    """Revoke a credential"""
    try:
        success = await agent.revoke_credential(request.name)
        
        return RevokeCredentialResponse(
            success=success,
            message="Credential revoked successfully" if success else "Failed to revoke credential"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/credentials/list")
async def list_credentials(request: ListCredentialsRequest):
    """List credentials with optional filtering"""
    try:
        credentials = await agent.list_credentials(
            filter_type=request.filter_type,
            include_inactive=request.include_inactive
        )
        
        return {
            "status": "success",
            "credentials": credentials,
            "count": len(credentials)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/audit", response_model=SecurityAuditResponse)
async def run_security_audit():
    """Run security audit on all credentials"""
    try:
        results = await agent.run_security_audit()
        
        return SecurityAuditResponse(
            timestamp=results["timestamp"],
            total_credentials=results["total_credentials"],
            active_credentials=results["active_credentials"],
            expired_credentials=results["expired_credentials"],
            weak_credentials=results["weak_credentials"],
            security_score=results["security_score"],
            recommendations=results["recommendations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "credentials_managed": agent.credentials_managed,
        "rotations_performed": agent.rotations_performed,
        "security_audits": agent.security_audits,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 