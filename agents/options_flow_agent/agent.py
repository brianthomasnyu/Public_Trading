"""
Options Flow Analysis Agent

AI Reasoning: This agent analyzes options flow data to identify:
1. Unusual options activity (unusual volume, open interest changes)
2. Options flow patterns (call/put ratios, money flow)
3. Volatility events and gamma exposure
4. Options chain analysis and liquidity
5. Options-based sentiment indicators
6. Institutional options positioning

NO TRADING DECISIONS - Only data aggregation and analysis for informational purposes.
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('POSTGRES_PORT')
DB_NAME = os.getenv('POSTGRES_DB')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}:{DB_NAME}"
engine = create_engine(DATABASE_URL)

# ============================================================================
# CRITICAL SYSTEM POLICY: NO TRADING DECISIONS
# ============================================================================
"""
SYSTEM POLICY: This agent is STRICTLY for data aggregation and analysis.
NO TRADING DECISIONS should be made. All options flow analysis is for
informational purposes only.

AI REASONING: The agent should:
1. Monitor options flow patterns and unusual activity
2. Analyze options-based sentiment indicators
3. Track volatility events and gamma exposure
4. Identify institutional options positioning
5. NEVER make buy/sell recommendations
6. NEVER provide trading advice
"""

@dataclass
class OptionsFlowEvent:
    """AI Reasoning: Comprehensive options flow event with analysis metadata"""
    ticker: str
    event_type: str  # unusual_volume, gamma_exposure, flow_pattern, volatility_spike
    timestamp: datetime
    strike_price: Optional[float] = None
    expiration_date: Optional[datetime] = None
    option_type: Optional[str] = None  # call, put
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    money_flow: Optional[float] = None
    ai_significance_score: float = 0.0
    ai_analysis_notes: List[str] = None

@dataclass
class OptionsFlowPattern:
    """AI Reasoning: Pattern analysis with confidence scoring"""
    pattern_type: str  # call_heavy, put_heavy, gamma_squeeze, volatility_spread
    confidence_score: float
    supporting_indicators: List[str]
    time_horizon: str  # short_term, medium_term, long_term
    ai_relevance_score: float = 0.0

class OptionsFlowAgent:
    """
    AI Reasoning: Intelligent options flow analysis system
    - Monitor unusual options activity and flow patterns
    - Analyze options-based sentiment and positioning
    - Track volatility events and gamma exposure
    - Identify institutional options activity
    - Coordinate with other agents for comprehensive analysis
    - NO TRADING DECISIONS - only data aggregation and analysis
    """
    
    def __init__(self):
        # AI Reasoning: Options data sources with reliability scoring
        self.data_sources = {
            'cboe': {
                'reliability': 0.95,
                'update_frequency': 'real_time',
                'data_types': ['volume', 'open_interest', 'implied_volatility'],
                'api_key': os.getenv('CBOE_API_KEY')
            },
            'squeezemetrics': {
                'reliability': 0.90,
                'update_frequency': 'daily',
                'data_types': ['gamma_exposure', 'unusual_activity'],
                'api_key': os.getenv('SQUEEZEMETRICS_API_KEY')
            },
            'optionmetrics': {
                'reliability': 0.88,
                'update_frequency': 'daily',
                'data_types': ['flow_analysis', 'sentiment_indicators'],
                'api_key': os.getenv('OPTIONMETRICS_API_KEY')
            }
        }
        
        # AI Reasoning: Options flow analysis thresholds and patterns
        self.analysis_thresholds = {
            'unusual_volume': {'multiplier': 3.0, 'significance': 'high'},
            'gamma_exposure': {'threshold': 0.10, 'significance': 'critical'},
            'call_put_ratio': {'threshold': 2.0, 'significance': 'medium'},
            'money_flow': {'threshold': 1000000, 'significance': 'high'},
            'volatility_spike': {'threshold': 0.50, 'significance': 'high'}
        }
        
        # AI Reasoning: Pattern recognition and classification
        self.flow_patterns = {
            'gamma_squeeze': {
                'indicators': ['high_gamma', 'low_liquidity', 'momentum'],
                'confidence_threshold': 0.75
            },
            'institutional_positioning': {
                'indicators': ['large_blocks', 'strategic_strikes', 'timing'],
                'confidence_threshold': 0.80
            },
            'sentiment_shift': {
                'indicators': ['ratio_changes', 'flow_direction', 'volume_spikes'],
                'confidence_threshold': 0.70
            }
        }
        
        self.agent_name = "options_flow_agent"

    async def check_knowledge_base_for_existing_data(self, ticker: str, event_type: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Check knowledge base for existing options flow data
        - Query existing options events and patterns
        - Assess data freshness and completeness
        - Determine if new data fetch is needed
        - Identify data gaps and inconsistencies
        - NO TRADING DECISIONS - only data validation
        """
        # PSEUDOCODE:
        # 1. Query knowledge base for ticker's recent options flow data
        # 2. Check last update timestamp and data freshness
        # 3. Assess data completeness against expected patterns
        # 4. Identify missing or outdated information
        # 5. Calculate confidence in existing data quality
        # 6. Determine if new data fetch is warranted
        # 7. Return existing data with quality assessment
        # 8. NO TRADING DECISIONS - only data validation
        
        try:
            with engine.connect() as conn:
                # AI Reasoning: Intelligent query based on event type and time range
                if event_type:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'options_flow_agent' 
                        AND data->>'ticker' = :ticker 
                        AND data->>'event_type' = :event_type
                        ORDER BY event_time DESC 
                        LIMIT 20
                    """)
                    result = conn.execute(query, {"ticker": ticker, "event_type": event_type})
                else:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'options_flow_agent' 
                        AND data->>'ticker' = :ticker
                        ORDER BY event_time DESC 
                        LIMIT 50
                    """)
                    result = conn.execute(query, {"ticker": ticker})
                
                existing_data = [dict(row) for row in result]
                
                # AI Reasoning: Assess data quality and freshness
                data_quality = {
                    'total_records': len(existing_data),
                    'latest_update': existing_data[0]['event_time'] if existing_data else None,
                    'data_freshness_hours': None,
                    'completeness_score': 0.0,
                    'confidence_level': 0.0
                }
                
                if existing_data:
                    latest_time = existing_data[0]['event_time']
                    data_quality['data_freshness_hours'] = (datetime.utcnow() - latest_time).total_seconds() / 3600
                    
                    # AI Reasoning: Calculate completeness based on expected data patterns
                    event_types = [event['data'].get('event_type') for event in existing_data]
                    data_quality['completeness_score'] = len(set(event_types)) / len(self.analysis_thresholds)
                    
                    # AI Reasoning: Assess confidence based on data consistency
                    data_quality['confidence_level'] = min(1.0, data_quality['completeness_score'] * 0.9)
                
                return {
                    'existing_data': existing_data,
                    'data_quality': data_quality,
                    'needs_update': data_quality['data_freshness_hours'] is None or data_quality['data_freshness_hours'] > 1.0
                }
                
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            return {'existing_data': [], 'data_quality': {}, 'needs_update': True}
    
    async def select_optimal_data_sources(self, ticker: str, analysis_type: str) -> List[str]:
        """
        AI Reasoning: Select optimal data sources for options flow analysis
        - Evaluate data source reliability and freshness
        - Match data sources to analysis requirements
        - Prioritize sources based on data quality
        - Consider API rate limits and costs
        - NO TRADING DECISIONS - only source optimization
        """
        # PSEUDOCODE:
        # 1. Analyze required data types for the analysis
        # 2. Evaluate available data sources and their capabilities
        # 3. Check data source reliability and update frequency
        # 4. Assess API rate limits and availability
        # 5. Prioritize sources based on data quality and cost
        # 6. Select optimal combination of data sources
        # 7. Return prioritized list of data sources
        # 8. NO TRADING DECISIONS - only source optimization
        
        selected_sources = []
        
        # AI Reasoning: Match analysis type to data source capabilities
        if analysis_type == 'unusual_activity':
            selected_sources = ['cboe', 'squeezemetrics']
        elif analysis_type == 'gamma_exposure':
            selected_sources = ['squeezemetrics', 'cboe']
        elif analysis_type == 'flow_patterns':
            selected_sources = ['optionmetrics', 'cboe']
        else:
            selected_sources = ['cboe', 'squeezemetrics', 'optionmetrics']
        
        # AI Reasoning: Filter by reliability and availability
        reliable_sources = [
            source for source in selected_sources 
            if self.data_sources[source]['reliability'] > 0.85
        ]
        
        return reliable_sources[:2]  # Limit to top 2 sources
    
    async def analyze_options_flow_patterns(self, flow_data: List[OptionsFlowEvent]) -> Dict[str, Any]:
        """
        AI Reasoning: Analyze options flow patterns and identify significant events
        - Detect unusual options activity patterns
        - Identify gamma exposure and volatility events
        - Analyze call/put ratios and money flow
        - Classify patterns by significance and confidence
        - NO TRADING DECISIONS - only pattern analysis
        """
        # PSEUDOCODE:
        # 1. Group options flow data by time periods and patterns
        # 2. Calculate key metrics (volume ratios, money flow, IV changes)
        # 3. Apply pattern recognition algorithms
        # 4. Score patterns by significance and confidence
        # 5. Identify unusual activity and anomalies
        # 6. Classify patterns by type and time horizon
        # 7. Generate analysis summary with confidence scores
        # 8. NO TRADING DECISIONS - only pattern analysis
        
        analysis_results = {
            'patterns_detected': [],
            'unusual_activity': [],
            'significance_score': 0.0,
            'confidence_level': 0.0,
            'analysis_notes': []
        }
        
        if not flow_data:
            return analysis_results
        
        # AI Reasoning: Calculate aggregate metrics
        total_volume = sum(event.volume or 0 for event in flow_data)
        call_volume = sum(event.volume or 0 for event in flow_data if event.option_type == 'call')
        put_volume = sum(event.volume or 0 for event in flow_data if event.option_type == 'put')
        
        call_put_ratio = call_volume / put_volume if put_volume > 0 else float('inf')
        total_money_flow = sum(event.money_flow or 0 for event in flow_data)
        
        # AI Reasoning: Detect unusual activity patterns
        for event in flow_data:
            if event.volume and event.volume > self.analysis_thresholds['unusual_volume']['multiplier'] * 1000:
                analysis_results['unusual_activity'].append({
                    'event': event,
                    'significance': 'high',
                    'reason': 'unusual_volume'
                })
            
            if event.implied_volatility and event.implied_volatility > self.analysis_thresholds['volatility_spike']['threshold']:
                analysis_results['unusual_activity'].append({
                    'event': event,
                    'significance': 'high',
                    'reason': 'volatility_spike'
                })
        
        # AI Reasoning: Classify flow patterns
        if call_put_ratio > self.analysis_thresholds['call_put_ratio']['threshold']:
            analysis_results['patterns_detected'].append({
                'pattern_type': 'call_heavy',
                'confidence_score': 0.8,
                'indicators': ['high_call_put_ratio', 'call_volume_spike']
            })
        
        if total_money_flow > self.analysis_thresholds['money_flow']['threshold']:
            analysis_results['patterns_detected'].append({
                'pattern_type': 'high_money_flow',
                'confidence_score': 0.7,
                'indicators': ['large_money_flow', 'institutional_activity']
            })
        
        # AI Reasoning: Calculate overall significance
        analysis_results['significance_score'] = len(analysis_results['unusual_activity']) * 0.3 + len(analysis_results['patterns_detected']) * 0.2
        analysis_results['confidence_level'] = min(1.0, analysis_results['significance_score'] * 0.8)
        
        return analysis_results
    
    async def determine_next_best_action(self, analysis_results: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        AI Reasoning: Determine next best action based on options flow analysis
        - Evaluate analysis significance and confidence
        - Decide on data refresh requirements
        - Plan coordination with other agents
        - Schedule follow-up analysis if needed
        - NO TRADING DECISIONS - only action planning
        """
        # PSEUDOCODE:
        # 1. Assess analysis significance and confidence levels
        # 2. Evaluate data freshness and completeness
        # 3. Determine if additional data sources are needed
        # 4. Plan coordination with related agents
        # 5. Schedule follow-up analysis if patterns detected
        # 6. Prioritize actions based on significance
        # 7. Return action plan with priorities
        # 8. NO TRADING DECISIONS - only action planning
        
        next_actions = {
            'immediate_actions': [],
            'scheduled_actions': [],
            'coordination_needed': [],
            'priority_level': 'low'
        }
        
        # AI Reasoning: Evaluate analysis significance
        significance_score = analysis_results.get('significance_score', 0.0)
        confidence_level = analysis_results.get('confidence_level', 0.0)
        
        if significance_score > 0.7 and confidence_level > 0.6:
            next_actions['priority_level'] = 'high'
            next_actions['immediate_actions'].append({
                'action': 'notify_orchestrator',
                'reason': 'high_significance_options_activity',
                'data': analysis_results
            })
            
            next_actions['coordination_needed'].append({
                'agent': 'market_news_agent',
                'reason': 'correlate_with_news_events',
                'priority': 'high'
            })
            
            next_actions['coordination_needed'].append({
                'agent': 'insider_trading_agent',
                'reason': 'check_for_insider_activity',
                'priority': 'medium'
            })
        
        elif significance_score > 0.4:
            next_actions['priority_level'] = 'medium'
            next_actions['scheduled_actions'].append({
                'action': 'follow_up_analysis',
                'schedule_minutes': 30,
                'reason': 'moderate_significance_pattern'
            })
        
        # AI Reasoning: Plan data refresh based on activity level
        if len(analysis_results.get('unusual_activity', [])) > 0:
            next_actions['scheduled_actions'].append({
                'action': 'refresh_options_data',
                'schedule_minutes': 15,
                'reason': 'active_options_flow'
            })
        
        return next_actions

    async def fetch_and_process_options(self):
        """
        AI Reasoning: Fetch and process options flow data from multiple sources
        - Retrieve options data from selected sources
        - Process and normalize data formats
        - Apply pattern recognition algorithms
        - Store significant events in knowledge base
        - Coordinate with other agents as needed
        - NO TRADING DECISIONS - only data processing
        """
        # PSEUDOCODE:
        # 1. Select high-priority tickers for options analysis
        # 2. Check knowledge base for existing data
        # 3. Select optimal data sources for each ticker
        # 4. Fetch options flow data from APIs
        # 5. Process and normalize data formats
        # 6. Apply pattern recognition and analysis
        # 7. Store significant events in knowledge base
        # 8. Determine next actions and coordinate with agents
        # 9. NO TRADING DECISIONS - only data processing
        
        try:
            # AI Reasoning: Select tickers for analysis (example tickers)
            priority_tickers = ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA']
            
            for ticker in priority_tickers:
                # AI Reasoning: Check existing data and determine update needs
                existing_data = await self.check_knowledge_base_for_existing_data(ticker)
                
                if not existing_data['needs_update']:
                    logger.info(f"Recent options data exists for {ticker}, skipping update")
                    continue
                
                # AI Reasoning: Select optimal data sources
                data_sources = await self.select_optimal_data_sources(ticker, 'unusual_activity')
                
                # AI Reasoning: Fetch options flow data
                flow_data = await self.fetch_options_data(ticker, data_sources)
                
                if flow_data:
                    # AI Reasoning: Analyze patterns and detect unusual activity
                    analysis_results = await self.analyze_options_flow_patterns(flow_data)
                    
                    # AI Reasoning: Store significant events in knowledge base
                    if analysis_results['significance_score'] > 0.3:
                        await self.store_in_knowledge_base(ticker, analysis_results)
                    
                    # AI Reasoning: Determine next actions
                    next_actions = await self.determine_next_best_action(analysis_results, ticker)
                    
                    # AI Reasoning: Execute immediate actions
                    for action in next_actions['immediate_actions']:
                        if action['action'] == 'notify_orchestrator':
                            await self.notify_orchestrator(action['data'])
                    
                    # AI Reasoning: Schedule follow-up actions
                    for action in next_actions['scheduled_actions']:
                        if action['action'] == 'follow_up_analysis':
                            asyncio.create_task(self.schedule_follow_up_analysis(ticker, action['schedule_minutes']))
                
                # AI Reasoning: Rate limiting between tickers
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error in fetch_and_process_options: {e}")
            # AI Reasoning: Implement error recovery and retry logic
            await self.handle_error_recovery(e)
    
    async def fetch_options_data(self, ticker: str, data_sources: List[str]) -> List[OptionsFlowEvent]:
        """
        AI Reasoning: Fetch options data from selected sources
        - Retrieve data from multiple APIs
        - Handle rate limiting and errors
        - Normalize data formats
        - Apply quality filters
        - NO TRADING DECISIONS - only data retrieval
        """
        # PSEUDOCODE:
        # 1. Initialize data collection from selected sources
        # 2. Handle API authentication and rate limiting
        # 3. Retrieve options flow data from each source
        # 4. Apply data quality filters and validation
        # 5. Normalize data formats across sources
        # 6. Merge and deduplicate data
        # 7. Return processed options flow events
        # 8. NO TRADING DECISIONS - only data retrieval
        
        flow_events = []
        
        async with aiohttp.ClientSession() as session:
            for source in data_sources:
                try:
                    if source == 'cboe' and self.data_sources[source]['api_key']:
                        # AI Reasoning: Fetch CBOE options data
                        data = await self.fetch_cboe_data(session, ticker)
                        if data:
                            flow_events.extend(self.parse_cboe_data(data, ticker))
                    
                    elif source == 'squeezemetrics' and self.data_sources[source]['api_key']:
                        # AI Reasoning: Fetch SqueezeMetrics data
                        data = await self.fetch_squeezemetrics_data(session, ticker)
                        if data:
                            flow_events.extend(self.parse_squeezemetrics_data(data, ticker))
                    
                    # AI Reasoning: Rate limiting between sources
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching data from {source}: {e}")
                    continue
        
        return flow_events
    
    async def fetch_cboe_data(self, session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
        """AI Reasoning: Fetch options data from CBOE API"""
        # PSEUDOCODE: Implement CBOE API integration
        return None
    
    async def fetch_squeezemetrics_data(self, session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
        """AI Reasoning: Fetch options data from SqueezeMetrics API"""
        # PSEUDOCODE: Implement SqueezeMetrics API integration
        return None
    
    def parse_cboe_data(self, data: Dict, ticker: str) -> List[OptionsFlowEvent]:
        """AI Reasoning: Parse and normalize CBOE options data"""
        # PSEUDOCODE: Implement CBOE data parsing
        return []
    
    def parse_squeezemetrics_data(self, data: Dict, ticker: str) -> List[OptionsFlowEvent]:
        """AI Reasoning: Parse and normalize SqueezeMetrics data"""
        # PSEUDOCODE: Implement SqueezeMetrics data parsing
        return []
    
    async def store_in_knowledge_base(self, ticker: str, analysis_results: Dict[str, Any]):
        """
        AI Reasoning: Store significant options flow events in knowledge base
        - Store events with proper metadata
        - Include analysis results and confidence scores
        - Tag events for easy retrieval
        - NO TRADING DECISIONS - only data storage
        """
        # PSEUDOCODE:
        # 1. Prepare event data with analysis results
        # 2. Include metadata and confidence scores
        # 3. Store in knowledge base with proper indexing
        # 4. Tag events for correlation analysis
        # 5. Update event tracking and statistics
        # 6. NO TRADING DECISIONS - only data storage
        
        try:
            event_data = {
                'ticker': ticker,
                'event_type': 'options_flow_analysis',
                'analysis_results': analysis_results,
                'timestamp': datetime.utcnow().isoformat(),
                'agent_version': '1.0',
                'confidence_score': analysis_results.get('confidence_level', 0.0)
            }
            
            with engine.connect() as conn:
                query = text("""
                    INSERT INTO events (source_agent, event_type, event_time, data)
                    VALUES (:source_agent, :event_type, :event_time, :data)
                """)
                
                conn.execute(query, {
                    'source_agent': self.agent_name,
                    'event_type': 'options_flow_analysis',
                    'event_time': datetime.utcnow(),
                    'data': json.dumps(event_data)
                })
                conn.commit()
                
            logger.info(f"Stored options flow analysis for {ticker}")
            
        except Exception as e:
            logger.error(f"Error storing options flow data: {e}")
    
    async def notify_orchestrator(self, data: Dict[str, Any]):
        """
        AI Reasoning: Notify orchestrator of significant options flow events
        - Send high-significance events to orchestrator
        - Include analysis results and confidence scores
        - Request coordination with other agents if needed
        - NO TRADING DECISIONS - only coordination
        """
        # PSEUDOCODE:
        # 1. Prepare notification data with analysis results
        # 2. Include confidence scores and significance levels
        # 3. Send to orchestrator via MCP
        # 4. Request coordination with related agents
        # 5. NO TRADING DECISIONS - only coordination
        
        try:
            notification = {
                'agent': self.agent_name,
                'event_type': 'significant_options_flow',
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'priority': 'high' if data.get('significance_score', 0.0) > 0.7 else 'medium'
            }
            
            # AI Reasoning: Send via MCP to orchestrator
            # await self.send_mcp_message('orchestrator', notification)
            logger.info(f"Notified orchestrator of significant options flow: {notification}")
            
        except Exception as e:
            logger.error(f"Error notifying orchestrator: {e}")
    
    async def schedule_follow_up_analysis(self, ticker: str, delay_minutes: int):
        """
        AI Reasoning: Schedule follow-up analysis for options flow patterns
        - Schedule delayed analysis for pattern confirmation
        - Monitor pattern evolution over time
        - Update analysis results as new data arrives
        - NO TRADING DECISIONS - only analysis scheduling
        """
        # PSEUDOCODE:
        # 1. Wait for specified delay period
        # 2. Re-fetch options flow data for the ticker
        # 3. Compare with previous analysis results
        # 4. Update pattern confidence and significance
        # 5. Store updated analysis in knowledge base
        # 6. NO TRADING DECISIONS - only analysis scheduling
        
        await asyncio.sleep(delay_minutes * 60)
        
        try:
            # AI Reasoning: Re-analyze options flow for pattern confirmation
            data_sources = await self.select_optimal_data_sources(ticker, 'flow_patterns')
            flow_data = await self.fetch_options_data(ticker, data_sources)
            
            if flow_data:
                analysis_results = await self.analyze_options_flow_patterns(flow_data)
                
                # AI Reasoning: Update knowledge base with follow-up analysis
                if analysis_results['significance_score'] > 0.2:
                    await self.store_in_knowledge_base(ticker, analysis_results)
                
                logger.info(f"Completed follow-up options analysis for {ticker}")
                
        except Exception as e:
            logger.error(f"Error in follow-up analysis for {ticker}: {e}")
    
    async def handle_error_recovery(self, error: Exception):
        """
        AI Reasoning: Handle errors and implement recovery strategies
        - Log errors with context and severity
        - Implement retry logic with exponential backoff
        - Fall back to alternative data sources
        - Maintain system stability and data quality
        - NO TRADING DECISIONS - only error recovery
        """
        # PSEUDOCODE:
        # 1. Log error with full context and stack trace
        # 2. Assess error severity and impact
        # 3. Implement appropriate recovery strategy
        # 4. Retry with exponential backoff if applicable
        # 5. Fall back to alternative data sources
        # 6. Update system health metrics
        # 7. NO TRADING DECISIONS - only error recovery
        
        logger.error(f"Options flow agent error: {error}")
        
        # AI Reasoning: Implement retry logic for transient errors
        if "rate limit" in str(error).lower():
            logger.info("Rate limit hit, implementing backoff strategy")
            await asyncio.sleep(60)  # Wait 1 minute before retry
        elif "connection" in str(error).lower():
            logger.info("Connection error, retrying with exponential backoff")
            await asyncio.sleep(30)  # Wait 30 seconds before retry

    async def listen_for_mcp_messages(self):
        """
        AI Reasoning: Listen for MCP messages from orchestrator and other agents
        - Handle requests for options flow analysis
        - Respond to coordination requests
        - Process priority analysis requests
        - NO TRADING DECISIONS - only message handling
        """
        # PSEUDOCODE:
        # 1. Listen for incoming MCP messages
        # 2. Parse message type and priority
        # 3. Handle analysis requests for specific tickers
        # 4. Respond with current analysis results
        # 5. Coordinate with requesting agents
        # 6. NO TRADING DECISIONS - only message handling
        
        try:
            # AI Reasoning: Check for MCP messages
            # message = await self.receive_mcp_message()
            # if message:
            #     await self.handle_mcp_message(message)
            
        await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in MCP message handling: {e}")
    
    async def run(self):
        """
        AI Reasoning: Main agent execution loop
        - Coordinate data fetching and analysis
        - Handle MCP communication
        - Implement error recovery and monitoring
        - Maintain system health and performance
        - NO TRADING DECISIONS - only system operation
        """
        # PSEUDOCODE:
        # 1. Initialize agent and establish connections
        # 2. Start MCP message listening
        # 3. Begin periodic options flow analysis
        # 4. Handle errors and implement recovery
        # 5. Monitor system health and performance
        # 6. Coordinate with other agents as needed
        # 7. NO TRADING DECISIONS - only system operation
        
        logger.info("Starting Options Flow Agent")
        
        while True:
            try:
                # AI Reasoning: Run main analysis cycle
                await self.fetch_and_process_options()
                
                # AI Reasoning: Handle MCP communication
                await self.listen_for_mcp_messages()
                
                # AI Reasoning: Wait before next cycle
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in main agent loop: {e}")
                await self.handle_error_recovery(e)
                await asyncio.sleep(60)  # Wait before retry

async def main():
    """AI Reasoning: Main entry point for Options Flow Agent"""
    agent = OptionsFlowAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 