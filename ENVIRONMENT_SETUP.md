# Environment Configuration Guide

## Overview

This guide provides a comprehensive template for configuring your financial data aggregation system environment variables. The system requires various API keys, database settings, and agent-specific configurations.

## CRITICAL SYSTEM POLICY

**NO TRADING DECISIONS OR RECOMMENDATIONS**

This system is STRICTLY for data aggregation, analysis, and knowledge base management. NO TRADING ADVICE is provided by any agent.

## Quick Setup

1. Create a `.env` file in your project root
2. Copy the template below and replace placeholder values
3. Never commit the `.env` file to version control

## Environment Variables Template

```bash
# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
POSTGRES_USER=financial_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=financial_data

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
ORCHESTRATOR_URL=http://localhost:8000
AGENT_TIMEOUT=30
MAX_RETRIES=3
HEALTH_CHECK_INTERVAL=60
LOG_LEVEL=INFO
ENABLE_DEBUG_MODE=false

# ============================================================================
# API KEYS - MARKET DATA
# ============================================================================
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here
FINNHUB_API_KEY=your_finnhub_key_here
QUANDL_API_KEY=your_quandl_key_here
YAHOO_FINANCE_API_KEY=your_yahoo_key_here

# ============================================================================
# API KEYS - FINANCIAL RESEARCH
# ============================================================================
TIPRANKS_API_KEY=your_tipranks_key_here
ZACKS_API_KEY=your_zacks_key_here
SEEKING_ALPHA_API_KEY=your_seeking_alpha_key_here
BLOOMBERG_API_KEY=your_bloomberg_key_here
REUTERS_API_KEY=your_reuters_key_here

# ============================================================================
# API KEYS - SEC AND REGULATORY
# ============================================================================
SEC_API_KEY=your_sec_key_here
EDGAR_API_KEY=your_edgar_key_here

# ============================================================================
# API KEYS - SOCIAL MEDIA
# ============================================================================
TWITTER_API_KEY=your_twitter_key_here
TWITTER_API_SECRET=your_twitter_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_SECRET=your_twitter_access_secret_here

REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=your_reddit_user_agent_here

STOCKTWITS_API_KEY=your_stocktwits_key_here

# ============================================================================
# API KEYS - OPTIONS AND DERIVATIVES
# ============================================================================
CBOE_API_KEY=your_cboe_key_here
SQUEEZEMETRICS_API_KEY=your_squeezemetrics_key_here
OPTIONMETRICS_API_KEY=your_optionmetrics_key_here

# ============================================================================
# API KEYS - ALTERNATIVE DATA
# ============================================================================
IEX_API_KEY=your_iex_key_here
DARK_POOL_API_KEY=your_dark_pool_key_here
INSIDER_TRADING_API_KEY=your_insider_trading_key_here

# ============================================================================
# API KEYS - ACADEMIC AND RESEARCH
# ============================================================================
ARXIV_API_KEY=your_arxiv_key_here
PAPERS_WITH_CODE_API_KEY=your_papers_with_code_key_here
GOOGLE_SCHOLAR_API_KEY=your_google_scholar_key_here
SSRN_API_KEY=your_ssrn_key_here

# ============================================================================
# AI REASONING CONFIGURATION
# ============================================================================
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1

CONFIDENCE_THRESHOLD=0.7
RELEVANCE_THRESHOLD=0.6
QUALITY_THRESHOLD=0.8
ANOMALY_DETECTION_THRESHOLD=0.75

# ============================================================================
# AGENT UPDATE INTERVALS (in seconds)
# ============================================================================
RESEARCH_UPDATE_INTERVAL=3600
SEC_UPDATE_INTERVAL=1800
NEWS_UPDATE_INTERVAL=300
SOCIAL_MEDIA_UPDATE_INTERVAL=600
INSIDER_UPDATE_INTERVAL=3600
PRICING_UPDATE_INTERVAL=7200
KPI_UPDATE_INTERVAL=3600
IMPACT_UPDATE_INTERVAL=300
OPTIONS_UPDATE_INTERVAL=300
ML_UPDATE_INTERVAL=86400
MACRO_UPDATE_INTERVAL=3600
GEOGRAPHY_UPDATE_INTERVAL=7200
TAGGING_UPDATE_INTERVAL=1800
DARK_POOL_UPDATE_INTERVAL=600
SHORT_INTEREST_UPDATE_INTERVAL=3600
COMMODITY_UPDATE_INTERVAL=1800
PORTFOLIO_UPDATE_INTERVAL=7200

# ============================================================================
# AGENT ALERT THRESHOLDS
# ============================================================================
RESEARCH_ALERT_THRESHOLD=0.2
SEC_ALERT_THRESHOLD=0.15
NEWS_ALERT_THRESHOLD=0.25
SOCIAL_MEDIA_ALERT_THRESHOLD=0.3
INSIDER_ALERT_THRESHOLD=0.4
PRICING_ALERT_THRESHOLD=0.1
KPI_ALERT_THRESHOLD=0.15
IMPACT_ALERT_THRESHOLD=0.2
OPTIONS_ALERT_THRESHOLD=0.35
ML_ALERT_THRESHOLD=0.1
MACRO_ALERT_THRESHOLD=0.25
GEOGRAPHY_ALERT_THRESHOLD=0.2
TAGGING_ALERT_THRESHOLD=0.1
DARK_POOL_ALERT_THRESHOLD=0.4
SHORT_INTEREST_ALERT_THRESHOLD=0.3
COMMODITY_ALERT_THRESHOLD=0.25
PORTFOLIO_ALERT_THRESHOLD=0.3

# ============================================================================
# FRONTEND CONFIGURATION
# ============================================================================
FRONTEND_PORT=3000
FRONTEND_HOST=localhost
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# ============================================================================
# SECURITY SETTINGS
# ============================================================================
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
ENABLE_SSL=false
ENABLE_AUDIT_LOGGING=true

# ============================================================================
# MONITORING AND ANALYTICS
# ============================================================================
ENABLE_HEALTH_MONITORING=true
ENABLE_PERFORMANCE_METRICS=true
ENABLE_ERROR_TRACKING=true
METRICS_PORT=9090
ENABLE_PROMETHEUS_METRICS=true

# ============================================================================
# DEVELOPMENT SETTINGS
# ============================================================================
ENVIRONMENT=development
ENABLE_TEST_MODE=false
ENABLE_MOCK_DATA=false
ENABLE_DEBUG_ENDPOINTS=false
```

## Required API Keys by Agent

### Data Collection Agents

#### SEC Filings Agent
- `SEC_API_KEY` - SEC EDGAR database access
- `EDGAR_API_KEY` - Alternative SEC data provider

#### Market News Agent
- `BLOOMBERG_API_KEY` - Bloomberg news and data
- `REUTERS_API_KEY` - Reuters news feed
- `ALPHA_VANTAGE_API_KEY` - Market news and sentiment

#### Social Media NLP Agent
- `TWITTER_API_KEY` - Twitter API access
- `TWITTER_API_SECRET` - Twitter API secret
- `TWITTER_ACCESS_TOKEN` - Twitter access token
- `TWITTER_ACCESS_SECRET` - Twitter access secret
- `REDDIT_CLIENT_ID` - Reddit API client ID
- `REDDIT_CLIENT_SECRET` - Reddit API client secret
- `REDDIT_USER_AGENT` - Reddit user agent string
- `STOCKTWITS_API_KEY` - StockTwits API access

#### Insider Trading Agent
- `INSIDER_TRADING_API_KEY` - Insider trading data provider

#### Investor Portfolio Agent
- `IEX_API_KEY` - IEX Cloud institutional data
- `BLOOMBERG_API_KEY` - Bloomberg institutional holdings

### Analysis Agents

#### Equity Research Agent
- `TIPRANKS_API_KEY` - TipRanks analyst ratings
- `ZACKS_API_KEY` - Zacks investment research
- `SEEKING_ALPHA_API_KEY` - Seeking Alpha premium content

#### Fundamental Pricing Agent
- `ALPHA_VANTAGE_API_KEY` - Fundamental data
- `POLYGON_API_KEY` - Market data and financials
- `FINNHUB_API_KEY` - Financial statements

#### KPI Tracker Agent
- `ALPHA_VANTAGE_API_KEY` - Earnings and KPI data
- `POLYGON_API_KEY` - Financial metrics
- `FINNHUB_API_KEY` - Company fundamentals

#### Event Impact Agent
- `ALPHA_VANTAGE_API_KEY` - Earnings calendar
- `POLYGON_API_KEY` - Event data
- `FINNHUB_API_KEY` - News and events

#### ML Model Testing Agent
- `ARXIV_API_KEY` - Academic papers
- `PAPERS_WITH_CODE_API_KEY` - Research papers
- `GOOGLE_SCHOLAR_API_KEY` - Academic research
- `SSRN_API_KEY` - Social Science Research Network

### Specialized Agents

#### Options Flow Agent
- `CBOE_API_KEY` - CBOE options data
- `SQUEEZEMETRICS_API_KEY` - Options flow analysis
- `OPTIONMETRICS_API_KEY` - Options analytics

#### Macro Calendar Agent
- `ALPHA_VANTAGE_API_KEY` - Economic indicators
- `POLYGON_API_KEY` - Macro data
- `FINNHUB_API_KEY` - Economic calendar

#### Revenue Geography Agent
- `ALPHA_VANTAGE_API_KEY` - Geographic revenue data
- `POLYGON_API_KEY` - Company segment data

#### Dark Pool Agent
- `DARK_POOL_API_KEY` - Dark pool data provider
- `IEX_API_KEY` - Alternative trading data

#### Short Interest Agent
- `ALPHA_VANTAGE_API_KEY` - Short interest data
- `FINNHUB_API_KEY` - Short selling metrics

#### Commodity Agent
- `ALPHA_VANTAGE_API_KEY` - Commodity prices
- `POLYGON_API_KEY` - Commodity data
- `FINNHUB_API_KEY` - Commodity indices

## AI Reasoning Configuration

### OpenAI Integration
- `OPENAI_API_KEY` - Required for AI reasoning capabilities
- `OPENAI_MODEL` - Model to use (gpt-4 recommended)
- `OPENAI_MAX_TOKENS` - Maximum tokens per request
- `OPENAI_TEMPERATURE` - Response creativity (0.1 for analysis)

### Reasoning Thresholds
- `CONFIDENCE_THRESHOLD` - Minimum confidence for results (0.7)
- `RELEVANCE_THRESHOLD` - Minimum relevance score (0.6)
- `QUALITY_THRESHOLD` - Minimum data quality (0.8)
- `ANOMALY_DETECTION_THRESHOLD` - Anomaly detection sensitivity (0.75)

## Agent Configuration

### Update Intervals
Each agent has configurable update intervals (in seconds):
- Real-time agents: 300-600 seconds
- Daily agents: 3600-7200 seconds
- Weekly agents: 86400 seconds

### Alert Thresholds
Each agent has configurable alert thresholds (0.0-1.0):
- High sensitivity: 0.1-0.2
- Medium sensitivity: 0.2-0.3
- Low sensitivity: 0.3-0.4

## Security Considerations

### Production Deployment
1. **Use strong passwords** for database and API keys
2. **Enable SSL/TLS** encryption
3. **Configure firewall rules** to restrict access
4. **Set up monitoring and alerting**
5. **Implement proper backup strategies**
6. **Use different API keys** for development and production
7. **Regularly rotate credentials**

### Environment Separation
- Use different API keys for development, staging, and production
- Configure separate databases for each environment
- Use environment-specific alert thresholds
- Enable debug mode only in development

## Monitoring and Analytics

### Health Monitoring
- `ENABLE_HEALTH_MONITORING=true` - Agent health checks
- `ENABLE_PERFORMANCE_METRICS=true` - Performance tracking
- `ENABLE_ERROR_TRACKING=true` - Error monitoring
- `ENABLE_USAGE_ANALYTICS=true` - Usage statistics

### Metrics Collection
- `METRICS_PORT=9090` - Prometheus metrics endpoint
- `ENABLE_PROMETHEUS_METRICS=true` - Prometheus integration

## Development Setup

### Local Development
```bash
# Create .env file
cp .env.template .env

# Edit with your values
nano .env

# Start system
docker-compose up -d
```

### Testing
- `ENABLE_TEST_MODE=false` - Enable for testing
- `ENABLE_MOCK_DATA=false` - Use mock data
- `ENABLE_DEBUG_ENDPOINTS=false` - Debug endpoints

## Troubleshooting

### Common Issues
1. **Missing API keys** - Check all required keys are set
2. **Database connection** - Verify PostgreSQL credentials
3. **Agent timeouts** - Increase `AGENT_TIMEOUT` value
4. **Rate limiting** - Check API usage limits
5. **Memory issues** - Adjust `MAX_MEMORY_USAGE_MB`

### Validation
The system validates environment variables on startup:
- Required API keys are present
- Database connection is successful
- Agent configurations are valid
- Security settings are appropriate

## Next Steps

1. **Obtain API keys** from required data providers
2. **Set up PostgreSQL database** with proper credentials
3. **Configure OpenAI API** for AI reasoning
4. **Test agent connections** individually
5. **Monitor system health** and performance
6. **Set up alerts** for critical events
7. **Implement backup** and recovery procedures

---

**Remember**: This system is for data aggregation and analysis only. No trading decisions or recommendations are provided. 