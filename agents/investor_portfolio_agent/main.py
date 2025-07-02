"""
Investor Portfolio Agent - Main Entry Point

AI Reasoning: Main entry point for investor portfolio tracking agent
- Initialize agent and validate configuration
- Set up MCP communication with orchestrator
- Handle portfolio tracking requests
- Coordinate with other agents for comprehensive analysis
- NO TRADING DECISIONS - only data processing and coordination
"""

import asyncio
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import uvicorn

from agent import InvestorPortfolioAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for agent communication
app = FastAPI(title="Investor Portfolio Agent", version="1.0.0")

# Initialize agent
agent = InvestorPortfolioAgent()

# ============================================================================
# CRITICAL SYSTEM POLICY: NO TRADING DECISIONS
# ============================================================================
"""
SYSTEM POLICY: This agent is STRICTLY for data aggregation and analysis.
NO TRADING DECISIONS should be made. All portfolio tracking is for
informational purposes only.

AI REASONING: The agent should:
1. Track portfolio changes and holdings
2. Analyze investment patterns and trends
3. Monitor disclosure requirements and compliance
4. Identify potential conflicts of interest
5. NEVER make buy/sell recommendations
6. NEVER provide trading advice
"""

# Data models for API communication
class PortfolioUpdateRequest(BaseModel):
    investor_id: str
    ticker: Optional[str] = None
    force_update: bool = False
    priority: str = "normal"

class PortfolioUpdateResponse(BaseModel):
    success: bool
    investor_id: str
    holdings_count: int
    analysis_results: Dict[str, Any]
    next_action: Dict[str, Any]
    agents_to_trigger: List[str]
    disclaimer: str = "NO TRADING DECISIONS - Data for informational purposes only"

class MCPMessage(BaseModel):
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health")
def health_check():
    """
    AI Reasoning: Health check with agent status and capabilities
    - Monitor agent health and performance
    - Report tracked investors and data sources
    - Provide system status and recommendations
    - NO TRADING DECISIONS - only system monitoring
    """
    return {
        "status": "healthy",
        "agent": "investor_portfolio_agent",
        "timestamp": datetime.utcnow(),
        "tracked_investors": len(agent.investor_profiles),
        "data_sources": list(agent.data_sources.keys()),
        "system_policy": "NO TRADING DECISIONS - Data aggregation only",
        "capabilities": [
            "Portfolio tracking and analysis",
            "Pattern recognition and significance scoring",
            "Multi-source data validation",
            "Agent coordination via MCP"
        ]
    }

# Portfolio update endpoint
@app.post("/update", response_model=PortfolioUpdateResponse)
async def update_portfolio(request: PortfolioUpdateRequest):
    """
    AI Reasoning: Process portfolio update request with intelligent analysis
    - Validate request and investor profile
    - Check existing data and determine update needs
    - Fetch and analyze portfolio data
    - Determine next actions and agent coordination
    - NO TRADING DECISIONS - only data processing
    """
    logger.info(f"Processing portfolio update request for {request.investor_id}")
    
    # PSEUDOCODE for portfolio update processing:
    # 1. Validate investor ID and request parameters
    # 2. Check if investor is tracked by this agent
    # 3. Determine if update is needed (unless forced)
    # 4. Process portfolio update using agent logic
    # 5. Handle errors and implement recovery strategies
    # 6. Return results with next action recommendations
    # 7. NO TRADING DECISIONS - only data processing
    
    try:
        # AI Reasoning: Validate investor
        if request.investor_id not in agent.investor_profiles:
            raise HTTPException(status_code=404, detail=f"Investor {request.investor_id} not tracked")
        
        # AI Reasoning: Process portfolio update
        result = await agent.process_portfolio_update(request.investor_id, request.ticker)
        
        return PortfolioUpdateResponse(
            success=result['success'],
            investor_id=result['investor_id'],
            holdings_count=result['holdings_count'],
            analysis_results=result['analysis_results'],
            next_action=result['next_action'],
            agents_to_trigger=result['agents_to_trigger'],
            disclaimer="NO TRADING DECISIONS - Data for informational purposes only"
        )
        
    except Exception as e:
        logger.error(f"Error processing portfolio update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MCP communication endpoint
@app.post("/mcp")
async def handle_mcp_message(message: MCPMessage):
    """
    AI Reasoning: Handle MCP messages from other agents and orchestrator
    - Process data requests and coordination messages
    - Handle agent-to-agent communication
    - Route messages to appropriate handlers
    - NO TRADING DECISIONS - only data coordination
    """
    logger.info(f"Received MCP message from {message.sender}: {message.message_type}")
    
    # PSEUDOCODE for MCP message handling:
    # 1. Validate message format and sender
    # 2. Route message based on type:
    #    - data_request: Process portfolio data request
    #    - coordination: Handle agent coordination
    #    - alert: Process urgent notifications
    # 3. Execute appropriate action based on message content
    # 4. Send response or trigger other actions
    # 5. Log message for audit trail
    # 6. NO TRADING DECISIONS - only data coordination
    
    try:
        if message.message_type == "data_request":
            # AI Reasoning: Handle data request from other agents
            if message.content.get('type') == 'portfolio_data':
                investor_id = message.content.get('investor_id')
                if investor_id:
                    result = await agent.process_portfolio_update(investor_id)
                    return {
                        "status": "success",
                        "data": result,
                        "timestamp": datetime.utcnow()
                    }
        
        elif message.message_type == "coordination":
            # AI Reasoning: Handle coordination messages
            action = message.content.get('action')
            if action == 'trigger_analysis':
                # Trigger analysis based on coordination request
                pass
        
        return {
            "status": "processed",
            "message_type": message.message_type,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error handling MCP message: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

# Get tracked investors endpoint
@app.get("/investors")
def get_tracked_investors():
    """
    AI Reasoning: Return list of tracked investors with metadata
    - Provide investor profiles and tracking status
    - Include disclosure requirements and data sources
    - NO TRADING DECISIONS - only information display
    """
    investors = []
    for investor_id, profile in agent.investor_profiles.items():
        investors.append({
            "investor_id": investor_id,
            "name": profile.name,
            "type": profile.type,
            "disclosure_requirements": profile.disclosure_requirements,
            "tracking_frequency": profile.tracking_frequency,
            "data_sources": profile.data_sources,
            "last_updated": profile.last_updated.isoformat(),
            "ai_confidence_score": profile.ai_confidence_score
        })
    
    return {
        "investors": investors,
        "total_count": len(investors),
        "disclaimer": "NO TRADING DECISIONS - Data for informational purposes only"
    }

# Get data sources endpoint
@app.get("/data_sources")
def get_data_sources():
    """
    AI Reasoning: Return data source information and reliability metrics
    - Provide data source reliability scores
    - Include update frequencies and delays
    - NO TRADING DECISIONS - only information display
    """
    return {
        "data_sources": agent.data_sources,
        "disclaimer": "NO TRADING DECISIONS - Data for informational purposes only"
    }

# AI Reasoning: Main execution function
async def run_agent():
    """
    AI Reasoning: Main agent execution with continuous operation
    - Initialize agent and validate configuration
    - Start FastAPI server for communication
    - Handle requests and coordinate with other agents
    - NO TRADING DECISIONS - only data processing
    """
    # PSEUDOCODE:
    # 1. Initialize agent and validate configuration
    # 2. Start FastAPI server for agent communication
    # 3. Handle incoming requests and MCP messages
    # 4. Coordinate with other agents as needed
    # 5. Monitor system health and performance
    # 6. Handle errors and implement recovery strategies
    # 7. NO TRADING DECISIONS - only data processing
    
    logger.info("Starting Investor Portfolio Agent server")
    
    # AI Reasoning: Validate agent configuration
    if not agent.investor_profiles:
        logger.error("No investor profiles configured")
        return
    
    logger.info(f"Agent configured with {len(agent.investor_profiles)} tracked investors")
    
    # AI Reasoning: Start server
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    asyncio.run(run_agent()) 