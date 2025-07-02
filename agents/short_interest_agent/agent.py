"""
Short Interest Analysis Agent

AI Reasoning: This agent analyzes short interest data and patterns for:
1. Short interest ratio calculations and trends
2. Days to cover analysis and implications
3. Short squeeze potential identification
4. Institutional short position tracking
5. Short interest correlation with price movements
6. Regulatory short interest reporting analysis

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
NO TRADING DECISIONS should be made. All short interest analysis is for
informational purposes only.

AI REASONING: The agent should:
1. Monitor short interest data and trends
2. Analyze short squeeze potential and patterns
3. Track institutional short positions
4. Assess short interest correlation with price movements
5. NEVER make buy/sell recommendations
6. NEVER provide trading advice
"""

@dataclass
class ShortInterestData:
    """AI Reasoning: Comprehensive short interest data with analysis metadata"""
    ticker: str
    short_interest: int
    shares_outstanding: int
    short_interest_ratio: float
    days_to_cover: float
    timestamp: datetime
    data_source: str
    confidence_score: float = 0.0
    ai_analysis_notes: List[str] = None

@dataclass
class ShortSqueezeAnalysis:
    """AI Reasoning: Short squeeze potential analysis with risk assessment"""
    ticker: str
    squeeze_probability: float
    risk_factors: List[str]
    trigger_conditions: List[str]
    historical_comparison: Dict[str, Any]
    confidence_level: float
    ai_relevance_score: float = 0.0

class ShortInterestAgent:
    """
    AI Reasoning: Intelligent short interest analysis system
    - Monitor short interest data and calculate key metrics
    - Analyze short squeeze potential and risk factors
    - Track institutional short positions and changes
    - Correlate short interest with price movements
    - Coordinate with other agents for comprehensive analysis
    - NO TRADING DECISIONS - only data aggregation and analysis
    """
    
    def __init__(self):
        # AI Reasoning: Short interest data sources and reliability scoring
        self.data_sources = {
            'finra': {
                'reliability': 0.95,
                'update_frequency': 'bi_monthly',
                'data_types': ['short_interest', 'days_to_cover'],
                'api_key': None  # Public data
            },
            'nasdaq': {
                'reliability': 0.90,
                'update_frequency': 'bi_monthly',
                'data_types': ['short_interest', 'short_interest_ratio'],
                'api_key': None  # Public data
            },
            'yahoo_finance': {
                'reliability': 0.85,
                'update_frequency': 'daily',
                'data_types': ['short_interest', 'shares_outstanding'],
                'api_key': None  # Public data
            },
            'bloomberg': {
                'reliability': 0.88,
                'update_frequency': 'daily',
                'data_types': ['institutional_short_positions'],
                'api_key': os.getenv('BLOOMBERG_API_KEY')
            }
        }
        
        # AI Reasoning: Short interest analysis thresholds and patterns
        self.analysis_thresholds = {
            'high_short_interest': {'ratio': 0.20, 'significance': 'high'},
            'extreme_short_interest': {'ratio': 0.50, 'significance': 'critical'},
            'squeeze_candidate': {'ratio': 0.30, 'days_to_cover': 5.0, 'significance': 'high'},
            'institutional_short': {'position_size': 1000000, 'significance': 'medium'}
        }
        
        # AI Reasoning: Short squeeze risk factors and indicators
        self.squeeze_risk_factors = {
            'high_short_interest_ratio': {'weight': 0.3, 'threshold': 0.30},
            'low_days_to_cover': {'weight': 0.25, 'threshold': 3.0},
            'positive_catalyst': {'weight': 0.2, 'threshold': 0.7},
            'institutional_short_concentration': {'weight': 0.15, 'threshold': 0.40},
            'price_momentum': {'weight': 0.1, 'threshold': 0.15}
        }
        
        self.agent_name = "short_interest_agent"
    
    async def check_knowledge_base_for_existing_data(self, ticker: str, data_type: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Check knowledge base for existing short interest data
        - Query existing short interest data and trends
        - Assess data freshness and completeness
        - Determine if new data fetch is needed
        - Identify data gaps and inconsistencies
        - NO TRADING DECISIONS - only data validation
        """
        # PSEUDOCODE:
        # 1. Query knowledge base for ticker's recent short interest data
        # 2. Check last update timestamp and data freshness
        # 3. Assess data completeness against expected patterns
        # 4. Identify missing or outdated information
        # 5. Calculate confidence in existing data quality
        # 6. Determine if new data fetch is warranted
        # 7. Return existing data with quality assessment
        # 8. NO TRADING DECISIONS - only data validation
        
        try:
            with engine.connect() as conn:
                # AI Reasoning: Intelligent query based on data type and time range
                if data_type:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'short_interest_agent' 
                        AND data->>'ticker' = :ticker 
                        AND data->>'data_type' = :data_type
                        ORDER BY event_time DESC 
                        LIMIT 20
                    """)
                    result = conn.execute(query, {"ticker": ticker, "data_type": data_type})
                else:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'short_interest_agent' 
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
                    'data_freshness_days': None,
                    'completeness_score': 0.0,
                    'confidence_level': 0.0
                }
                
                if existing_data:
                    latest_time = existing_data[0]['event_time']
                    data_quality['data_freshness_days'] = (datetime.utcnow() - latest_time).days
                    
                    # AI Reasoning: Calculate completeness based on expected data types
                    data_types = [event['data'].get('data_type') for event in existing_data]
                    data_quality['completeness_score'] = len(set(data_types)) / len(self.data_sources)
                    
                    # AI Reasoning: Assess confidence based on data consistency
                    data_quality['confidence_level'] = min(1.0, data_quality['completeness_score'] * 0.9)
                
                return {
                    'existing_data': existing_data,
                    'data_quality': data_quality,
                    'needs_update': data_quality['data_freshness_days'] is None or data_quality['data_freshness_days'] > 15
                }
                
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            return {'existing_data': [], 'data_quality': {}, 'needs_update': True}
    
    async def select_optimal_data_sources(self, ticker: str, analysis_type: str) -> List[str]:
        """
        AI Reasoning: Select optimal data sources for short interest analysis
        - Evaluate data source reliability and freshness
        - Match data sources to analysis requirements
        - Prioritize sources based on data quality
        - Consider update frequency and availability
        - NO TRADING DECISIONS - only source optimization
        """
        # PSEUDOCODE:
        # 1. Analyze required data types for the analysis
        # 2. Evaluate available data sources and their capabilities
        # 3. Check data source reliability and update frequency
        # 4. Assess data availability and completeness
        # 5. Prioritize sources based on data quality and timeliness
        # 6. Select optimal combination of data sources
        # 7. Return prioritized list of data sources
        # 8. NO TRADING DECISIONS - only source optimization
        
        selected_sources = []
        
        # AI Reasoning: Match analysis type to data source capabilities
        if analysis_type == 'short_interest_ratio':
            selected_sources = ['finra', 'nasdaq', 'yahoo_finance']
        elif analysis_type == 'institutional_short':
            selected_sources = ['bloomberg', 'finra']
        elif analysis_type == 'squeeze_analysis':
            selected_sources = ['finra', 'nasdaq', 'bloomberg']
        else:
            selected_sources = ['finra', 'nasdaq', 'yahoo_finance', 'bloomberg']
        
        # AI Reasoning: Filter by reliability and availability
        reliable_sources = [
            source for source in selected_sources 
            if self.data_sources[source]['reliability'] > 0.85
        ]
        
        return reliable_sources[:3]  # Limit to top 3 sources
    
    async def calculate_short_interest_metrics(self, short_data: ShortInterestData) -> Dict[str, Any]:
        """
        AI Reasoning: Calculate comprehensive short interest metrics
        - Calculate short interest ratio and days to cover
        - Assess significance relative to historical data
        - Identify unusual patterns and trends
        - NO TRADING DECISIONS - only metric calculation
        """
        # PSEUDOCODE:
        # 1. Calculate short interest ratio (short interest / shares outstanding)
        # 2. Calculate days to cover (short interest / average daily volume)
        # 3. Compare metrics to historical averages and thresholds
        # 4. Identify significant deviations and patterns
        # 5. Score metrics by significance and confidence
        # 6. Generate comprehensive metrics analysis
        # 7. NO TRADING DECISIONS - only metric calculation
        
        metrics_analysis = {
            'short_interest_ratio': short_data.short_interest_ratio,
            'days_to_cover': short_data.days_to_cover,
            'significance_level': 'normal',
            'historical_percentile': 0.0,
            'trend_direction': 'stable',
            'confidence_score': short_data.confidence_score,
            'analysis_notes': []
        }
        
        # AI Reasoning: Assess significance based on thresholds
        if short_data.short_interest_ratio > self.analysis_thresholds['extreme_short_interest']['ratio']:
            metrics_analysis['significance_level'] = 'critical'
            metrics_analysis['analysis_notes'].append('Extreme short interest ratio detected')
        elif short_data.short_interest_ratio > self.analysis_thresholds['high_short_interest']['ratio']:
            metrics_analysis['significance_level'] = 'high'
            metrics_analysis['analysis_notes'].append('High short interest ratio detected')
        
        # AI Reasoning: Assess days to cover significance
        if short_data.days_to_cover < self.analysis_thresholds['squeeze_candidate']['days_to_cover']:
            metrics_analysis['significance_level'] = 'high'
            metrics_analysis['analysis_notes'].append('Low days to cover - potential squeeze candidate')
        
        # AI Reasoning: Calculate trend direction (placeholder for historical comparison)
        # In a real implementation, this would compare with historical data
        metrics_analysis['trend_direction'] = 'stable'
        
        return metrics_analysis
    
    async def analyze_squeeze_potential(self, short_data: ShortInterestData, price_data: Dict[str, Any] = None) -> ShortSqueezeAnalysis:
        """
        AI Reasoning: Analyze short squeeze potential and risk factors
        - Calculate squeeze probability based on multiple factors
        - Identify trigger conditions and risk factors
        - Assess historical comparison and patterns
        - NO TRADING DECISIONS - only squeeze analysis
        """
        # PSEUDOCODE:
        # 1. Calculate squeeze probability using weighted risk factors
        # 2. Identify specific trigger conditions and catalysts
        # 3. Compare with historical squeeze patterns
        # 4. Assess institutional short concentration
        # 5. Analyze price momentum and volume patterns
        # 6. Generate comprehensive squeeze analysis
        # 7. NO TRADING DECISIONS - only squeeze analysis
        
        squeeze_probability = 0.0
        risk_factors = []
        trigger_conditions = []
        
        # AI Reasoning: Calculate squeeze probability using weighted factors
        if short_data.short_interest_ratio > self.squeeze_risk_factors['high_short_interest_ratio']['threshold']:
            squeeze_probability += self.squeeze_risk_factors['high_short_interest_ratio']['weight']
            risk_factors.append('high_short_interest_ratio')
            trigger_conditions.append('short_interest_above_30_percent')
        
        if short_data.days_to_cover < self.squeeze_risk_factors['low_days_to_cover']['threshold']:
            squeeze_probability += self.squeeze_risk_factors['low_days_to_cover']['weight']
            risk_factors.append('low_days_to_cover')
            trigger_conditions.append('days_to_cover_below_3')
        
        # AI Reasoning: Assess price momentum (placeholder)
        if price_data and price_data.get('momentum', 0) > self.squeeze_risk_factors['price_momentum']['threshold']:
            squeeze_probability += self.squeeze_risk_factors['price_momentum']['weight']
            risk_factors.append('positive_price_momentum')
            trigger_conditions.append('price_momentum_above_threshold')
        
        # AI Reasoning: Historical comparison (placeholder)
        historical_comparison = {
            'similar_squeezes': 0,
            'success_rate': 0.0,
            'average_duration': 0,
            'confidence': 0.0
        }
        
        # AI Reasoning: Calculate confidence level
        confidence_level = min(1.0, squeeze_probability * 1.2)
        
        return ShortSqueezeAnalysis(
            ticker=short_data.ticker,
            squeeze_probability=squeeze_probability,
            risk_factors=risk_factors,
            trigger_conditions=trigger_conditions,
            historical_comparison=historical_comparison,
            confidence_level=confidence_level,
            ai_relevance_score=squeeze_probability
        )
    
    async def correlate_with_price_movements(self, short_data: ShortInterestData, price_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        AI Reasoning: Correlate short interest with price movements
        - Analyze price movement patterns relative to short interest
        - Identify correlation strength and direction
        - Assess timing relationships and lag effects
        - NO TRADING DECISIONS - only correlation analysis
        """
        # PSEUDOCODE:
        # 1. Align short interest data with price history
        # 2. Calculate correlation coefficients and significance
        # 3. Identify timing relationships and lag effects
        # 4. Analyze price movement patterns around short interest changes
        # 5. Assess correlation strength and direction
        # 6. Generate correlation analysis report
        # 7. NO TRADING DECISIONS - only correlation analysis
        
        correlation_analysis = {
            'correlation_coefficient': 0.0,
            'correlation_significance': 'none',
            'price_impact': 'neutral',
            'timing_relationship': 'no_pattern',
            'lag_effects': [],
            'confidence_score': 0.0,
            'analysis_notes': []
        }
        
        if not price_history or len(price_history) < 10:
            correlation_analysis['analysis_notes'].append('Insufficient price history for correlation analysis')
            return correlation_analysis
        
        # AI Reasoning: Calculate correlation coefficient (simplified)
        # In a real implementation, this would use statistical correlation methods
        short_interest_values = [short_data.short_interest_ratio] * len(price_history)
        price_values = [entry.get('close', 0) for entry in price_history]
        
        # AI Reasoning: Simple correlation calculation (placeholder)
        if len(price_values) > 1:
            # Calculate correlation using numpy or similar in real implementation
            correlation_analysis['correlation_coefficient'] = 0.1  # Placeholder
            correlation_analysis['correlation_significance'] = 'weak'
            
            if abs(correlation_analysis['correlation_coefficient']) > 0.7:
                correlation_analysis['correlation_significance'] = 'strong'
            elif abs(correlation_analysis['correlation_coefficient']) > 0.3:
                correlation_analysis['correlation_significance'] = 'moderate'
        
        # AI Reasoning: Assess price impact
        if correlation_analysis['correlation_coefficient'] > 0.3:
            correlation_analysis['price_impact'] = 'positive'
            correlation_analysis['analysis_notes'].append('Positive correlation with price movements')
        elif correlation_analysis['correlation_coefficient'] < -0.3:
            correlation_analysis['price_impact'] = 'negative'
            correlation_analysis['analysis_notes'].append('Negative correlation with price movements')
        
        return correlation_analysis
    
    async def determine_next_best_action(self, analysis_results: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        AI Reasoning: Determine next best action based on short interest analysis
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
        significance_level = analysis_results.get('significance_level', 'normal')
        squeeze_probability = analysis_results.get('squeeze_probability', 0.0)
        
        if significance_level == 'critical' or squeeze_probability > 0.7:
            next_actions['priority_level'] = 'high'
            next_actions['immediate_actions'].append({
                'action': 'notify_orchestrator',
                'reason': 'critical_short_interest_activity',
                'data': analysis_results
            })
            
            next_actions['coordination_needed'].append({
                'agent': 'options_flow_agent',
                'reason': 'check_options_activity_for_squeeze_signals',
                'priority': 'high'
            })
            
            next_actions['coordination_needed'].append({
                'agent': 'market_news_agent',
                'reason': 'correlate_with_news_events',
                'priority': 'high'
            })
        
        elif significance_level == 'high' or squeeze_probability > 0.4:
            next_actions['priority_level'] = 'medium'
            next_actions['scheduled_actions'].append({
                'action': 'follow_up_analysis',
                'schedule_hours': 24,
                'reason': 'high_significance_short_interest_pattern'
            })
        
        # AI Reasoning: Plan data refresh based on significance
        if significance_level in ['high', 'critical']:
            next_actions['scheduled_actions'].append({
                'action': 'refresh_short_interest_data',
                'schedule_hours': 12,
                'reason': 'active_short_interest_pattern'
            })
        
        return next_actions
    
    async def fetch_and_process_short_interest_data(self):
        """
        AI Reasoning: Fetch and process short interest data from multiple sources
        - Retrieve short interest data from selected sources
        - Process and normalize data formats
        - Apply pattern recognition algorithms
        - Store significant events in knowledge base
        - Coordinate with other agents as needed
        - NO TRADING DECISIONS - only data processing
        """
        # PSEUDOCODE:
        # 1. Select high-priority tickers for short interest analysis
        # 2. Check knowledge base for existing data
        # 3. Select optimal data sources for each ticker
        # 4. Fetch short interest data from APIs
        # 5. Process and normalize data formats
        # 6. Apply pattern recognition and analysis
        # 7. Store significant events in knowledge base
        # 8. Determine next actions and coordinate with agents
        # 9. NO TRADING DECISIONS - only data processing
        
        try:
            # AI Reasoning: Select tickers for analysis (example tickers)
            priority_tickers = ['GME', 'AMC', 'TSLA', 'NVDA', 'AAPL']
            
            for ticker in priority_tickers:
                # AI Reasoning: Check existing data and determine update needs
                existing_data = await self.check_knowledge_base_for_existing_data(ticker)
                
                if not existing_data['needs_update']:
                    logger.info(f"Recent short interest data exists for {ticker}, skipping update")
                    continue
                
                # AI Reasoning: Select optimal data sources
                data_sources = await self.select_optimal_data_sources(ticker, 'short_interest_ratio')
                
                # AI Reasoning: Fetch short interest data
                short_interest_data = await self.fetch_short_interest_data(ticker, data_sources)
                
                if short_interest_data:
                    # AI Reasoning: Calculate short interest metrics
                    metrics_analysis = await self.calculate_short_interest_metrics(short_interest_data)
                    
                    # AI Reasoning: Analyze squeeze potential
                    squeeze_analysis = await self.analyze_squeeze_potential(short_interest_data)
                    
                    # AI Reasoning: Correlate with price movements
                    price_correlation = await self.correlate_with_price_movements(short_interest_data, [])
                    
                    # AI Reasoning: Store significant events in knowledge base
                    if metrics_analysis['significance_level'] in ['high', 'critical']:
                        await self.store_in_knowledge_base(ticker, {
                            'metrics_analysis': metrics_analysis,
                            'squeeze_analysis': squeeze_analysis,
                            'price_correlation': price_correlation
                        })
                    
                    # AI Reasoning: Determine next actions
                    next_actions = await self.determine_next_best_action(metrics_analysis, ticker)
                    
                    # AI Reasoning: Execute immediate actions
                    for action in next_actions['immediate_actions']:
                        if action['action'] == 'notify_orchestrator':
                            await self.notify_orchestrator(action['data'])
                    
                    # AI Reasoning: Schedule follow-up actions
                    for action in next_actions['scheduled_actions']:
                        if action['action'] == 'follow_up_analysis':
                            asyncio.create_task(self.schedule_follow_up_analysis(ticker, action['schedule_hours']))
                
                # AI Reasoning: Rate limiting between tickers
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error in fetch_and_process_short_interest_data: {e}")
            # AI Reasoning: Implement error recovery and retry logic
            await self.handle_error_recovery(e)
    
    async def fetch_short_interest_data(self, ticker: str, data_sources: List[str]) -> Optional[ShortInterestData]:
        """
        AI Reasoning: Fetch short interest data from selected sources
        - Retrieve data from multiple APIs
        - Handle rate limiting and errors
        - Normalize data formats
        - Apply quality filters
        - NO TRADING DECISIONS - only data retrieval
        """
        # PSEUDOCODE:
        # 1. Initialize data collection from selected sources
        # 2. Handle API authentication and rate limiting
        # 3. Retrieve short interest data from each source
        # 4. Apply data quality filters and validation
        # 5. Normalize data formats across sources
        # 6. Merge and deduplicate data
        # 7. Return processed short interest data
        # 8. NO TRADING DECISIONS - only data retrieval
        
        short_interest_data = None
        
        async with aiohttp.ClientSession() as session:
            for source in data_sources:
                try:
                    if source == 'finra':
                        # AI Reasoning: Fetch FINRA short interest data
                        data = await self.fetch_finra_short_interest(session, ticker)
                        if data:
                            short_interest_data = self.parse_finra_short_interest(data, ticker)
                    
                    elif source == 'nasdaq':
                        # AI Reasoning: Fetch NASDAQ short interest data
                        data = await self.fetch_nasdaq_short_interest(session, ticker)
                        if data:
                            short_interest_data = self.parse_nasdaq_short_interest(data, ticker)
                    
                    elif source == 'yahoo_finance':
                        # AI Reasoning: Fetch Yahoo Finance short interest data
                        data = await self.fetch_yahoo_short_interest(session, ticker)
                        if data:
                            short_interest_data = self.parse_yahoo_short_interest(data, ticker)
                    
                    # AI Reasoning: Rate limiting between sources
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching data from {source}: {e}")
                    continue
        
        return short_interest_data
    
    async def fetch_finra_short_interest(self, session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
        """AI Reasoning: Fetch short interest data from FINRA"""
        # PSEUDOCODE: Implement FINRA short interest API integration
        return None
    
    async def fetch_nasdaq_short_interest(self, session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
        """AI Reasoning: Fetch short interest data from NASDAQ"""
        # PSEUDOCODE: Implement NASDAQ short interest API integration
        return None
    
    async def fetch_yahoo_short_interest(self, session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
        """AI Reasoning: Fetch short interest data from Yahoo Finance"""
        # PSEUDOCODE: Implement Yahoo Finance short interest integration
        return None
    
    def parse_finra_short_interest(self, data: Dict, ticker: str) -> ShortInterestData:
        """AI Reasoning: Parse and normalize FINRA short interest data"""
        # PSEUDOCODE: Implement FINRA data parsing
        return ShortInterestData(
            ticker=ticker,
            short_interest=1000000,
            shares_outstanding=10000000,
            short_interest_ratio=0.10,
            days_to_cover=2.5,
            timestamp=datetime.utcnow(),
            data_source='finra',
            confidence_score=0.9
        )
    
    def parse_nasdaq_short_interest(self, data: Dict, ticker: str) -> ShortInterestData:
        """AI Reasoning: Parse and normalize NASDAQ short interest data"""
        # PSEUDOCODE: Implement NASDAQ data parsing
        return ShortInterestData(
            ticker=ticker,
            short_interest=1000000,
            shares_outstanding=10000000,
            short_interest_ratio=0.10,
            days_to_cover=2.5,
            timestamp=datetime.utcnow(),
            data_source='nasdaq',
            confidence_score=0.9
        )
    
    def parse_yahoo_short_interest(self, data: Dict, ticker: str) -> ShortInterestData:
        """AI Reasoning: Parse and normalize Yahoo Finance short interest data"""
        # PSEUDOCODE: Implement Yahoo Finance data parsing
        return ShortInterestData(
            ticker=ticker,
            short_interest=1000000,
            shares_outstanding=10000000,
            short_interest_ratio=0.10,
            days_to_cover=2.5,
            timestamp=datetime.utcnow(),
            data_source='yahoo_finance',
            confidence_score=0.85
        )
    
    async def store_in_knowledge_base(self, ticker: str, analysis_results: Dict[str, Any]):
        """
        AI Reasoning: Store significant short interest events in knowledge base
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
                'event_type': 'short_interest_analysis',
                'analysis_results': analysis_results,
                'timestamp': datetime.utcnow().isoformat(),
                'agent_version': '1.0',
                'confidence_score': analysis_results.get('metrics_analysis', {}).get('confidence_score', 0.0)
            }
            
            with engine.connect() as conn:
                query = text("""
                    INSERT INTO events (source_agent, event_type, event_time, data)
                    VALUES (:source_agent, :event_type, :event_time, :data)
                """)
                
                conn.execute(query, {
                    'source_agent': self.agent_name,
                    'event_type': 'short_interest_analysis',
                    'event_time': datetime.utcnow(),
                    'data': json.dumps(event_data)
                })
                conn.commit()
                
            logger.info(f"Stored short interest analysis for {ticker}")
            
        except Exception as e:
            logger.error(f"Error storing short interest data: {e}")
    
    async def notify_orchestrator(self, data: Dict[str, Any]):
        """
        AI Reasoning: Notify orchestrator of significant short interest events
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
                'event_type': 'significant_short_interest_activity',
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'priority': 'high' if data.get('significance_level') == 'critical' else 'medium'
            }
            
            # AI Reasoning: Send via MCP to orchestrator
            # await self.send_mcp_message('orchestrator', notification)
            logger.info(f"Notified orchestrator of significant short interest activity: {notification}")
            
        except Exception as e:
            logger.error(f"Error notifying orchestrator: {e}")
    
    async def schedule_follow_up_analysis(self, ticker: str, delay_hours: int):
        """
        AI Reasoning: Schedule follow-up analysis for short interest patterns
        - Schedule delayed analysis for pattern confirmation
        - Monitor pattern evolution over time
        - Update analysis results as new data arrives
        - NO TRADING DECISIONS - only analysis scheduling
        """
        # PSEUDOCODE:
        # 1. Wait for specified delay period
        # 2. Re-fetch short interest data for the ticker
        # 3. Compare with previous analysis results
        # 4. Update pattern confidence and significance
        # 5. Store updated analysis in knowledge base
        # 6. NO TRADING DECISIONS - only analysis scheduling
        
        await asyncio.sleep(delay_hours * 3600)
        
        try:
            # AI Reasoning: Re-analyze short interest data for pattern confirmation
            data_sources = await self.select_optimal_data_sources(ticker, 'short_interest_ratio')
            short_interest_data = await self.fetch_short_interest_data(ticker, data_sources)
            
            if short_interest_data:
                metrics_analysis = await self.calculate_short_interest_metrics(short_interest_data)
                
                # AI Reasoning: Update knowledge base with follow-up analysis
                if metrics_analysis['significance_level'] in ['high', 'critical']:
                    await self.store_in_knowledge_base(ticker, {'metrics_analysis': metrics_analysis})
                
                logger.info(f"Completed follow-up short interest analysis for {ticker}")
                
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
        
        logger.error(f"Short interest agent error: {error}")
        
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
        - Handle requests for short interest analysis
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
        - Coordinate short interest data fetching and analysis
        - Handle MCP communication
        - Implement error recovery and monitoring
        - Maintain system health and performance
        - NO TRADING DECISIONS - only system operation
        """
        # PSEUDOCODE:
        # 1. Initialize agent and establish connections
        # 2. Start MCP message listening
        # 3. Begin periodic short interest analysis
        # 4. Handle errors and implement recovery
        # 5. Monitor system health and performance
        # 6. Coordinate with other agents as needed
        # 7. NO TRADING DECISIONS - only system operation
        
        logger.info("Starting Short Interest Agent")
        
        while True:
            try:
                # AI Reasoning: Run main analysis cycle
                await self.fetch_and_process_short_interest_data()
                
                # AI Reasoning: Handle MCP communication
                await self.listen_for_mcp_messages()
                
                # AI Reasoning: Wait before next cycle
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in main agent loop: {e}")
                await self.handle_error_recovery(e)
                await asyncio.sleep(60)  # Wait before retry

async def main():
    """AI Reasoning: Main entry point for Short Interest Agent"""
    agent = ShortInterestAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 