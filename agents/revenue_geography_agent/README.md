# Revenue Geography Agent

## Overview
The Revenue Geography Agent is an intelligent system that maps company sales and revenue by geographic region using the FactSet GeoRev API. The agent analyzes regional performance and trends using AI, determines significance, and triggers other agents when appropriate for comprehensive financial data analysis.

## AI Reasoning Capabilities

### Core AI Functions
1. **Geographic Revenue Mapping**
   - Maps company sales and revenue by geographic region
   - Identifies regional performance patterns and trends
   - Extracts key geographic entities and relationships
   - Calculates regional concentration and diversification metrics

2. **Regional Performance Analysis**
   - Analyzes regional performance and growth trends
   - Compares regional performance with company averages
   - Identifies regional opportunities and risks
   - Assesses regional market conditions and competition

3. **Geographic Risk Assessment**
   - Evaluates geographic concentration risks
   - Identifies regional dependencies and vulnerabilities
   - Assesses political and economic risks by region
   - Calculates diversification benefits and costs

4. **Trend Analysis & Forecasting**
   - Analyzes geographic revenue trends over time
   - Identifies regional growth and decline patterns
   - Compares with industry and market trends
   - Assesses regional market penetration and expansion

### Advanced AI Reasoning Functions

#### Data Existence Analysis
- Uses GPT-4 to check if geographic mappings are already in knowledge base
- Compares with existing knowledge base entries for same regions
- Determines if new mappings add value or are redundant
- Calculates similarity scores based on geographic overlap and revenue data

#### Intelligent Geographic Analysis Selection
- Analyzes company characteristics (industry, size, global presence)
- Considers regional focus and strategic priorities
- Selects optimal analysis approach:
  - Global companies: Comprehensive regional analysis
  - Regional companies: Focused local market analysis
  - Emerging markets: Growth and expansion analysis
  - Mature markets: Optimization and efficiency analysis

#### Next Action Decision Making
- Analyzes geography insights for key triggers
- If significant regional changes detected → trigger equity research agent
- If unusual geographic patterns → trigger SEC filings agent
- If regional risks identified → trigger event impact agent
- If geographic opportunities detected → trigger multiple analysis agents

### MCP Communication & Coordination
- Processes incoming MCP messages with intelligent routing
- Handles urgent requests with priority
- Maintains message processing guarantees
- Coordinates with other agents for verification and analysis

### Error Handling & Recovery
- Implements intelligent recovery strategies based on error type
- Data validation error: Skip and log
- API error: Retry with backoff
- Database error: Retry with connection reset
- Mapping error: Retry with different parameters

### Health Monitoring & Optimization
- Calculates health scores based on error rates, API response times, and data quality
- Tracks performance metrics over time
- Identifies potential issues early
- Adjusts processing frequency based on reporting cycles and data availability

## Data Sources
- **FactSet GeoRev API**: Geographic revenue data and regional performance metrics
- **Company Financial Data**: Revenue and sales data by region
- **Market Data**: Regional market conditions and competition
- **Economic Data**: Political and economic risk factors by region

## Configuration
Set the following environment variables:
- `FACTSET_GEOREV_API_KEY`: FactSet GeoRev API key
- `POSTGRES_USER`: Database username
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_HOST`: Database host
- `POSTGRES_PORT`: Database port
- `POSTGRES_DB`: Database name
- `ORCHESTRATOR_URL`: Orchestrator MCP endpoint

## Critical System Policy
**NO TRADING DECISIONS**: This agent is STRICTLY for data aggregation, analysis, and knowledge base management. NO TRADING DECISIONS should be made. All analysis is for informational purposes only.

## Next Steps for Implementation
1. Implement GPT-4 integration for AI reasoning functions
2. Add real FactSet GeoRev API integration and parsing
3. Implement MCP communication with orchestrator
4. Add comprehensive error handling and recovery mechanisms
5. Create integration tests for agent coordination
6. Implement data validation and quality checks
7. Add monitoring and alerting capabilities
8. Optimize performance and resource usage 