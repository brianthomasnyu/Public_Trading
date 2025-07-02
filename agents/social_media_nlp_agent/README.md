# Social Media NLP Agent

## Overview
The Social Media NLP Agent is an intelligent system that analyzes social media posts from multiple sources (Twitter, Reddit, StockTwits) to extract claims, sentiment, and credibility using advanced AI reasoning. The agent determines relevance and triggers other agents when appropriate for comprehensive financial data analysis.

## AI Reasoning Capabilities

### Core AI Functions
1. **Claim Extraction & Analysis**
   - Uses GPT-4 to extract structured claims from social media posts
   - Identifies factual claims vs opinions vs rumors with confidence scores
   - Extracts key entities (companies, people, events, dates)
   - Categorizes claims by type (financial, operational, regulatory)

2. **Sentiment Analysis & Crowd Behavior**
   - Analyzes post language and tone for sentiment markers
   - Identifies potential meme stock signals and crowd behavior patterns
   - Tracks sentiment trends and shifts over time
   - Considers source credibility in sentiment assessment

3. **Credibility Assessment**
   - Evaluates source reputation and verification status
   - Considers historical accuracy of claims from sources
   - Assesses claim plausibility and supporting evidence
   - Assigns credibility scores and confidence levels

4. **Intelligent Data Source Selection**
   - Chooses optimal data source based on claim type and requirements
   - Twitter: Real-time discussions and breaking news
   - Reddit: Detailed discussions and community sentiment
   - StockTwits: Financial-focused discussions
   - Factors in API rate limits and historical data quality

### Advanced AI Reasoning Functions

#### Data Existence Analysis
- Uses GPT-4 to check if social media claims are already verified or refuted in knowledge base
- Compares with existing knowledge base entries for same claims
- Determines if new data adds value or is redundant
- Calculates similarity scores based on claim overlap and verification status

#### Verification Logic
- Analyzes each claim for verifiability and potential impact
- Prioritizes claims based on credibility, source reputation, and market impact
- Routes claims to appropriate verification agents:
  - Financial claims → SEC filings or news agents
  - Technical claims → options flow or event impact agents
  - Regulatory claims → news or SEC filings agents

#### Next Action Decision Making
- Analyzes post insights for key triggers
- If debt claims detected → trigger SEC filings agent
- If earnings claims detected → trigger news or event impact agents
- If technical analysis claims → trigger options flow agent
- If high-credibility claims → trigger multiple verification agents

### MCP Communication & Coordination
- Processes incoming MCP messages with intelligent routing
- Handles urgent requests with priority
- Maintains message processing guarantees
- Coordinates with other agents for verification and analysis

### Error Handling & Recovery
- Implements intelligent recovery strategies based on error type
- API rate limit: Wait and retry with backoff
- Network error: Retry with exponential backoff
- Data validation error: Skip and log
- Database error: Retry with connection reset

### Health Monitoring & Optimization
- Calculates health scores based on error rates, API response times, and data quality
- Tracks performance metrics over time
- Identifies potential issues early
- Adjusts processing frequency based on system load and activity levels

## Data Sources
- **Twitter API**: Real-time discussions and breaking news
- **Reddit API**: Detailed discussions and community sentiment
- **StockTwits API**: Financial-focused discussions

## Configuration
Set the following environment variables:
- `TWITTER_API_KEY`: Twitter API key
- `REDDIT_CLIENT_ID`: Reddit API client ID
- `REDDIT_CLIENT_SECRET`: Reddit API client secret
- `REDDIT_USER_AGENT`: Reddit API user agent
- `STOCKTWITS_API_KEY`: StockTwits API key

## Critical System Policy
**NO TRADING DECISIONS**: This agent is STRICTLY for data aggregation, analysis, and knowledge base management. NO TRADING DECISIONS should be made. All analysis is for informational purposes only.

## Next Steps for Implementation
1. Implement GPT-4 integration for AI reasoning functions
2. Add real API integrations for Twitter, Reddit, and StockTwits
3. Implement MCP communication with orchestrator
4. Add comprehensive error handling and recovery mechanisms
5. Create integration tests for agent coordination
6. Implement data validation and quality checks
7. Add monitoring and alerting capabilities
8. Optimize performance and resource usage 