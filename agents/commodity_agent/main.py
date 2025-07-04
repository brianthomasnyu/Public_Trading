"""
Commodity Agent - Main Entry Point
=================================

FastAPI server for commodity analysis and sector impact assessment with multi-tool integration.
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

from agent import CommodityAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Commodity Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = {
    "commodity_update_interval": int(os.getenv("COMMODITY_UPDATE_INTERVAL", "300")),
    "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.2")),
    "max_commodities_per_cycle": int(os.getenv("MAX_COMMODITIES_PER_CYCLE", "20"))
}

agent = CommodityAgent(config)

# Data models
class CommodityDataRequest(BaseModel):
    commodity_symbol: str

class CommodityDataResponse(BaseModel):
    commodity_id: str
    name: str
    symbol: str
    category: str
    current_price: float
    price_change: float
    price_change_pct: float
    volume: int
    timestamp: str
    source: str
    confidence: float

class SectorImpactRequest(BaseModel):
    commodity_symbol: str

class SectorImpactResponse(BaseModel):
    commodity_id: str
    sector: str
    impact_type: str
    impact_score: float
    impact_factors: List[str]
    affected_companies: List[str]
    analysis_date: str
    confidence: float

class SupplyDemandRequest(BaseModel):
    commodity_symbol: str

class SupplyDemandResponse(BaseModel):
    commodity_id: str
    supply_level: str
    demand_level: str
    inventory_levels: str
    production_trend: str
    consumption_trend: str
    analysis_date: str
    confidence: float

class WeatherImpactRequest(BaseModel):
    commodity_symbol: str

class GeopoliticalImpactRequest(BaseModel):
    commodity_symbol: str

class CommodityAlertResponse(BaseModel):
    type: str
    commodity: str
    severity: str
    message: str
    timestamp: str
    impacted_sectors: Optional[List[str]] = None
    impact_score: Optional[float] = None

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
            "agent": agent.name,
            "version": agent.version,
            "health_score": agent.health_score,
            "multi_tool_integration": metrics.get("multi_tool_integration", {}),
            "last_update": metrics.get("last_update"),
            "error_count": metrics.get("error_count", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "agent": agent.name,
            "version": agent.version,
            "health_score": 0.0,
            "error": str(e)
        }

@app.post("/commodity/data", response_model=CommodityDataResponse)
async def get_commodity_data(request: CommodityDataRequest):
    """Get current commodity data"""
    try:
        commodity_data = await agent.fetch_commodity_data(request.commodity_symbol)
        
        if not commodity_data:
            raise HTTPException(status_code=404, detail="Commodity data not found")
        
        return CommodityDataResponse(
            commodity_id=commodity_data.commodity_id,
            name=commodity_data.name,
            symbol=commodity_data.symbol,
            category=commodity_data.category,
            current_price=commodity_data.current_price,
            price_change=commodity_data.price_change,
            price_change_pct=commodity_data.price_change_pct,
            volume=commodity_data.volume,
            timestamp=commodity_data.timestamp.isoformat(),
            source=commodity_data.source,
            confidence=commodity_data.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/commodity/sector-impacts")
async def get_sector_impacts(request: SectorImpactRequest):
    """Get sector impacts for a commodity"""
    try:
        commodity_data = await agent.fetch_commodity_data(request.commodity_symbol)
        
        if not commodity_data:
            raise HTTPException(status_code=404, detail="Commodity data not found")
        
        sector_impacts = await agent.analyze_sector_impact(commodity_data)
        
        return {
            "status": "success",
            "commodity": request.commodity_symbol,
            "sector_impacts": [
                {
                    "commodity_id": impact.commodity_id,
                    "sector": impact.sector,
                    "impact_type": impact.impact_type,
                    "impact_score": impact.impact_score,
                    "impact_factors": impact.impact_factors,
                    "affected_companies": impact.affected_companies,
                    "analysis_date": impact.analysis_date.isoformat(),
                    "confidence": impact.confidence
                }
                for impact in sector_impacts
            ],
            "count": len(sector_impacts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/commodity/supply-demand", response_model=SupplyDemandResponse)
async def get_supply_demand_analysis(request: SupplyDemandRequest):
    """Get supply/demand analysis for a commodity"""
    try:
        supply_demand = await agent.analyze_supply_demand(request.commodity_symbol)
        
        if not supply_demand:
            raise HTTPException(status_code=404, detail="Supply/demand analysis not found")
        
        return SupplyDemandResponse(
            commodity_id=supply_demand.commodity_id,
            supply_level=supply_demand.supply_level,
            demand_level=supply_demand.demand_level,
            inventory_levels=supply_demand.inventory_levels,
            production_trend=supply_demand.production_trend,
            consumption_trend=supply_demand.consumption_trend,
            analysis_date=supply_demand.analysis_date.isoformat(),
            confidence=supply_demand.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/commodity/weather-impact")
async def get_weather_impact(request: WeatherImpactRequest):
    """Get weather impact analysis for a commodity"""
    try:
        weather_impact = await agent.monitor_weather_impact(request.commodity_symbol)
        
        return {
            "status": "success",
            "commodity": request.commodity_symbol,
            "weather_impact": weather_impact
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/commodity/geopolitical-impact")
async def get_geopolitical_impact(request: GeopoliticalImpactRequest):
    """Get geopolitical impact analysis for a commodity"""
    try:
        geopolitical_impact = await agent.analyze_geopolitical_impact(request.commodity_symbol)
        
        return {
            "status": "success",
            "commodity": request.commodity_symbol,
            "geopolitical_impact": geopolitical_impact
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/commodity/alerts")
async def generate_alerts(request: CommodityDataRequest):
    """Generate alerts for a commodity"""
    try:
        commodity_data = await agent.fetch_commodity_data(request.commodity_symbol)
        
        if not commodity_data:
            raise HTTPException(status_code=404, detail="Commodity data not found")
        
        sector_impacts = await agent.analyze_sector_impact(commodity_data)
        alerts = await agent.generate_commodity_alerts(commodity_data, sector_impacts)
        
        return {
            "status": "success",
            "commodity": request.commodity_symbol,
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/commodities/list")
async def list_commodities():
    """List all tracked commodities"""
    try:
        commodities = []
        for symbol, data in agent.commodities.items():
            commodities.append({
                "symbol": symbol.upper(),
                "category": data['category'],
                "last_price": data['last_price'],
                "trend": data['trend']
            })
        
        return {
            "status": "success",
            "commodities": commodities,
            "count": len(commodities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors/list")
async def list_sectors():
    """List all sectors and their commodity dependencies"""
    try:
        sectors = []
        for sector, commodities in agent.sector_commodity_mapping.items():
            sectors.append({
                "sector": sector,
                "commodities": commodities,
                "commodity_count": len(commodities)
            })
        
        return {
            "status": "success",
            "sectors": sectors,
            "count": len(sectors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    return {
        "commodities_tracked": agent.commodities_tracked,
        "impacts_analyzed": agent.impacts_analyzed,
        "sector_alerts": agent.sector_alerts,
        "health_score": agent.health_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 