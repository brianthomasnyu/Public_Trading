# Investor Portfolio Tracking Agent

## Overview

AI Reasoning: This agent tracks notable investor portfolios including congress people, hedge fund managers, institutional investors, and other significant market participants. The agent provides comprehensive portfolio analysis, pattern recognition, and coordination with other agents for holistic market intelligence.

**CRITICAL: NO TRADING DECISIONS - Data for informational purposes only**

## Features

### AI Reasoning Capabilities

1. **Intelligent Portfolio Tracking**
   - Track portfolio changes and holdings across different investor types
   - Analyze investment patterns and identify trends
   - Monitor disclosure compliance and timing
   - Identify potential conflicts of interest

2. **Multi-Source Data Integration**
   - SEC 13F filings for institutional investors
   - SEC Form 4 for insider transactions
   - House.gov for congressional disclosures
   - OpenSecrets for political finance data
   - WhaleWisdom for hedge fund data

3. **Pattern Recognition**
   - Large position changes and new positions
   - Sector concentration and diversification
   - Timing patterns and potential conflicts
   - Significance scoring for changes

4. **Agent Coordination**
   - Trigger other agents based on findings
   - Coordinate comprehensive analysis workflows
   - Share insights via MCP communication

## Tracked Investor Types

### Congress People
- **Example**: Nancy Pelosi
- **Disclosure Requirements**: STOCK Act, 45-day disclosure
- **Data Sources**: House.gov, OpenSecrets, SEC Form 4
- **Tracking Frequency**: Daily

### Hedge Fund Managers
- **Examples**: Bill Ackman, Warren Buffett, Ray Dalio
- **Disclosure Requirements**: 13F, 13D/G filings
- **Data Sources**: SEC 13F, WhaleWisdom, company filings
- **Tracking Frequency**: Quarterly

### Institutional Investors
- **Examples**: Pension funds, endowments, sovereign wealth funds
- **Disclosure Requirements**: 13F filings
- **Data Sources**: SEC 13F, company disclosures
- **Tracking Frequency**: Quarterly

### Corporate Insiders
- **Examples**: CEOs, CFOs, board members
- **Disclosure Requirements**: SEC Form 4
- **Data Sources**: SEC Form 4, company filings
- **Tracking Frequency**: Daily

## AI Reasoning Workflow

### 1. Data Existence Check
```python
# PSEUDOCODE:
# 1. Query knowledge base for existing portfolio data
# 2. Check data freshness and completeness
# 3. Determine if new data fetch is needed
# 4. Assess data quality and confidence
```

### 2. Optimal Data Source Selection
```python
# PSEUDOCODE:
# 1. Analyze investor type and disclosure requirements
# 2. Evaluate data source reliability and freshness
# 3. Select optimal combination of sources
# 4. Prioritize based on data type and urgency
```

### 3. Portfolio Change Analysis
```python
# PSEUDOCODE:
# 1. Compare old vs. new holdings
# 2. Calculate percentage changes and significance
# 3. Identify patterns and anomalies
# 4. Assess potential conflicts of interest
```

### 4. Next Action Decision
```python
# PSEUDOCODE:
# 1. Assess significance of findings
# 2. Consider investor importance
# 3. Determine optimal agent coordination
# 4. Plan comprehensive analysis workflow
```

## API Endpoints

### Health Check
```
GET /health
```
Returns agent status and capabilities.

### Portfolio Update
```
POST /update
```
Process portfolio update for specific investor.

### MCP Communication
```
POST /mcp
```
Handle agent-to-agent communication.

### Tracked Investors
```
GET /investors
```
List all tracked investors with metadata.

### Data Sources
```
GET /data_sources
```
Information about data sources and reliability.

## Configuration

### Environment Variables
```bash
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=financial_data
```

### Investor Profiles
Configure tracked investors in `agent.py`:
```python
self.investor_profiles = {
    'nancy_pelosi': InvestorProfile(
        name="Nancy Pelosi",
        type="congress",
        entity_id="pelosi_nancy",
        disclosure_requirements=["STOCK Act", "45-day disclosure"],
        tracking_frequency="daily",
        data_sources=["House.gov", "OpenSecrets", "SEC Form 4"]
    )
}
```

## Installation

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**
4. **Run the agent**
   ```bash
   python main.py
   ```

## Docker Deployment

```bash
docker build -t investor-portfolio-agent .
docker run -p 8001:8001 investor-portfolio-agent
```

## Integration

### MCP Communication
The agent communicates with other agents via MCP:
- Sends portfolio analysis results
- Requests additional data from other agents
- Coordinates comprehensive analysis workflows

### Knowledge Base
Stores portfolio data and analysis results in PostgreSQL:
- Portfolio holdings and changes
- Analysis results and significance scores
- Agent coordination metadata

## Error Handling

### Recovery Strategies
- API rate limit handling with exponential backoff
- Data source fallback mechanisms
- Agent restart and recovery procedures
- Error logging and monitoring

### Data Validation
- Cross-reference data across multiple sources
- Validate data format and completeness
- Flag anomalies and inconsistencies
- Calculate confidence scores

## Performance Optimization

### Caching
- Cache frequently accessed portfolio data
- Implement intelligent cache invalidation
- Optimize database queries

### Parallel Processing
- Fetch data from multiple sources concurrently
- Process multiple investors in parallel
- Coordinate agent communication efficiently

## Security Considerations

### Data Privacy
- Handle sensitive investor information securely
- Implement proper access controls
- Log data access for audit trails

### API Security
- Validate all incoming requests
- Implement rate limiting
- Secure communication channels

## Monitoring and Logging

### Health Monitoring
- Track agent performance and availability
- Monitor data source reliability
- Alert on system issues

### Audit Logging
- Log all portfolio updates and analysis
- Track agent coordination activities
- Maintain compliance audit trails

## Future Enhancements

### AI Reasoning Improvements
- Enhanced pattern recognition algorithms
- Predictive analytics for portfolio changes
- Advanced conflict detection
- Sentiment analysis integration

### Data Source Expansion
- Additional regulatory filings
- Alternative data sources
- Real-time data feeds
- International investor tracking

### Agent Coordination
- Advanced workflow orchestration
- Dynamic agent selection
- Intelligent resource allocation
- Automated response optimization

## Disclaimer

**NO TRADING DECISIONS**: This agent is strictly for data aggregation and analysis. It does not make trading decisions or provide investment advice. All analysis is for informational purposes only.

## Contributing

1. Follow the AI reasoning patterns established in the codebase
2. Add comprehensive pseudocode for new features
3. Maintain the NO TRADING DECISIONS policy
4. Test thoroughly before submitting changes
5. Update documentation for new capabilities 