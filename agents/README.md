# AI Financial Data Aggregation Agents

This folder contains all intelligent agent microservices. Each agent runs as a separate Docker service with uniform structure and is responsible for specific data ingestion, analysis, or reasoning tasks.

## CRITICAL SYSTEM POLICY

**NO TRADING DECISIONS OR RECOMMENDATIONS**

All agents are STRICTLY for data aggregation, analysis, and knowledge base management. NO TRADING ADVICE is provided by any agent.

## Uniform Agent Structure

Every agent follows the same professional structure:

```
agents/agent_name/
├── agent.py          # Main agent logic with AI reasoning
├── main.py           # FastAPI server with endpoints
├── Dockerfile        # Containerized deployment
├── requirements.txt  # Python dependencies
└── README.md         # Comprehensive documentation
```

## Agent Categories

### Data Collection Agents

1. **SEC Filings Agent** (`sec_filings_agent/`)
   - Analyzes SEC filings, financial statements, and regulatory documents
   - Extracts key financial metrics and disclosures
   - Tracks filing patterns and compliance

2. **Market News Agent** (`market_news_agent/`)
   - Processes market news, announcements, and media coverage
   - Performs sentiment analysis on news content
   - Tracks news impact on stock prices

3. **Social Media NLP Agent** (`social_media_nlp_agent/`)
   - Analyzes social media sentiment and trends
   - Processes Reddit, Twitter, and other platforms
   - Identifies viral content and sentiment shifts

4. **Insider Trading Agent** (`insider_trading_agent/`)
   - Tracks insider trading activities and Form 4 filings
   - Monitors executive and director transactions
   - Analyzes insider trading patterns

5. **Investor Portfolio Agent** (`investor_portfolio_agent/`)
   - Monitors institutional and congressional trading activities
   - Tracks hedge fund and mutual fund positions
   - Analyzes portfolio changes and trends

### Analysis Agents

6. **Equity Research Agent** (`equity_research_agent/`)
   - Processes analyst reports, ratings, and research coverage
   - Extracts price targets and recommendations
   - Tracks analyst sentiment changes

7. **Fundamental Pricing Agent** (`fundamental_pricing_agent/`)
   - Performs valuation analysis using multiple methodologies
   - Calculates DCF, DDM, and relative valuation models
   - Tracks intrinsic value vs market price

8. **KPI Tracker Agent** (`kpi_tracker_agent/`)
   - Monitors key performance indicators and earnings metrics
   - Tracks revenue, profit, and growth metrics
   - Analyzes KPI trends and forecasts

9. **Event Impact Agent** (`event_impact_agent/`)
   - Analyzes the impact of events and catalysts on performance
   - Tracks earnings, product launches, and regulatory events
   - Calculates event-driven price impacts

10. **Comparative Analysis Agent** (`comparative_analysis_agent/`)
    - Performs peer, sector, and historical comparisons
    - Benchmarks companies against competitors
    - Analyzes relative performance metrics

11. **ML Model Testing Agent** (`ml_model_testing_agent/`)
    - Validates and tests machine learning models and predictions
    - Processes research papers and academic literature
    - Implements and validates predictive models

### Specialized Agents

12. **Options Flow Agent** (`options_flow_agent/`)
    - Analyzes options trading patterns and unusual activity
    - Tracks options volume and open interest
    - Identifies unusual options activity

13. **Macro Calendar Agent** (`macro_calendar_agent/`)
    - Tracks economic events and macro trends
    - Monitors Fed meetings, economic data releases
    - Analyzes macro impact on markets

14. **Revenue Geography Agent** (`revenue_geography_agent/`)
    - Analyzes geographic revenue distribution
    - Maps company sales by region and country
    - Tracks geographic expansion and risks

15. **Data Tagging Agent** (`data_tagging_agent/`)
    - Categorizes and organizes data for better retrieval
    - Implements intelligent tagging and indexing
    - Enables semantic search and correlation

16. **Dark Pool Agent** (`dark_pool_agent/`)
    - Monitors alternative trading venues and OTC activity
    - Tracks dark pool volume and patterns
    - Analyzes institutional trading behavior

17. **Short Interest Agent** (`short_interest_agent/`)
    - Tracks short interest and borrowing patterns
    - Monitors short squeeze potential
    - Analyzes short interest trends

18. **Commodity Agent** (`commodity_agent/`)
    - Monitors commodity prices and sector impacts
    - Tracks energy, metals, agriculture, and softs
    - Analyzes commodity-sector correlations

### System Management Agents

19. **Discovery Agent** (`discovery_agent/`)
    - Generates context-aware questions and coordinates with other agents
    - Implements intelligent question generation
    - Coordinates multi-agent investigations

20. **Repository Management Agent** (`repository_management_agent/`)
    - Manages codebase, version control, and development workflows
    - Handles Git operations and code deployment
    - Manages development and deployment pipelines

21. **API Key Management Agent** (`api_key_management_agent/`)
    - Securely manages credentials and access controls
    - Handles API key rotation and security
    - Manages authentication and authorization

## A2A Protocol & MCP Communication

All agents implement the Agent-to-Agent (A2A) protocol with MCP (Message Communication Protocol):

### Communication Features
- **Inter-Agent Messaging**: Seamless communication between agents
- **Parallel Execution**: Concurrent processing across multiple agents
- **Context Sharing**: Intelligent context propagation
- **Database Integration**: Centralized knowledge base coordination
- **Message Routing**: Intelligent routing based on agent capabilities
- **Error Recovery**: Robust error handling and retry mechanisms

### MCP Endpoints
Each agent exposes:
- `POST /mcp`: Receive messages from other agents
- `GET /health`: Health check and status
- Agent-specific endpoints for specialized functionality

## Agent Capabilities

### AI Reasoning
Every agent implements comprehensive AI reasoning:
- **Query Understanding**: Natural language processing
- **Pattern Recognition**: Automated trend identification
- **Anomaly Detection**: Unusual activity identification
- **Correlation Analysis**: Multi-factor relationship analysis
- **Confidence Scoring**: Quality assessment of results

### Data Processing
- **Multi-Source Integration**: Combines data from multiple sources
- **Real-Time Processing**: Live data analysis and updates
- **Historical Analysis**: Long-term trend analysis
- **Quality Validation**: Data accuracy and completeness checks

### Error Handling
- **Graceful Degradation**: Continues operation with partial failures
- **Retry Mechanisms**: Automatic retry with exponential backoff
- **Health Monitoring**: Continuous health assessment
- **Recovery Strategies**: Intelligent recovery from failures

## Deployment

### Docker Deployment
All agents are containerized with:
- **Security**: Non-root user execution
- **Health Checks**: Automated health monitoring
- **Resource Limits**: CPU and memory constraints
- **Network Isolation**: Secure inter-agent communication

### Configuration
Agents are configured via environment variables:
- Database connections
- API keys and credentials
- Processing intervals and thresholds
- Communication endpoints

## Development

### Adding New Agents
1. Create agent directory with uniform structure
2. Implement agent logic with AI reasoning
3. Add MCP communication capabilities
4. Update orchestrator agent mapping
5. Add to docker-compose.yml

### Testing
- Unit tests for agent logic
- Integration tests for MCP communication
- End-to-end tests for complete workflows
- Performance tests for scalability

## Monitoring

### Health Monitoring
- Agent status and availability
- Performance metrics and response times
- Error rates and recovery success
- Data quality and processing accuracy

### Analytics
- Query processing statistics
- Agent utilization and efficiency
- Communication patterns and bottlenecks
- System-wide performance metrics

---

**Total Agents: 21 | All Standardized | All Containerized | All AI-Powered** 