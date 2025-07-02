"""
Event Impact Analysis Agent

AI Reasoning: This agent analyzes the impact of market events on:
1. Stock prices and market movements
2. Sector and industry performance
3. Volatility and market sentiment
4. Trading volume and liquidity
5. Cross-asset correlations
6. Market microstructure effects

NO TRADING DECISIONS - Only event impact analysis for informational purposes.
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
SYSTEM POLICY: This agent is STRICTLY for event impact analysis.
NO TRADING DECISIONS should be made. All event impact analysis is for
informational purposes only.

AI REASONING: The agent should:
1. Analyze the impact of market events on various assets
2. Assess event significance and market reaction
3. Monitor post-event market behavior
4. Identify event-driven patterns and correlations
5. NEVER make buy/sell recommendations
6. NEVER provide trading advice
"""

@dataclass
class MarketEvent:
    """AI Reasoning: Comprehensive market event with impact metadata"""
    event_id: str
    event_type: str  # earnings, news, economic_data, corporate_action, market_event
    ticker: Optional[str] = None
    sector: Optional[str] = None
    event_time: datetime = None
    event_description: str = ""
    expected_impact: str = "neutral"  # positive, negative, neutral
    ai_significance_score: float = 0.0
    ai_analysis_notes: List[str] = None

@dataclass
class EventImpact:
    """AI Reasoning: Event impact analysis with market reaction data"""
    event_id: str
    impact_type: str  # price_movement, volume_spike, volatility_change, sentiment_shift
    pre_event_data: Dict[str, Any]
    post_event_data: Dict[str, Any]
    impact_magnitude: float
    impact_duration: str  # immediate, short_term, medium_term, long_term
    confidence_score: float
    ai_relevance_score: float = 0.0

class EventImpactAgent:
    """
    AI Reasoning: Intelligent event impact analysis system
    - Analyze the impact of market events on various assets
    - Assess event significance and market reaction patterns
    - Monitor post-event market behavior and recovery
    - Identify event-driven correlations and patterns
    - Coordinate with other agents for comprehensive analysis
    - NO TRADING DECISIONS - only event impact analysis
    """
    
    def __init__(self):
        # AI Reasoning: Event types and impact categories
        self.event_categories = {
            'earnings': {
                'impact_scope': 'company_specific',
                'analysis_horizon': 'short_term',
                'key_metrics': ['price_change', 'volume_spike', 'volatility_increase'],
                'significance_threshold': 0.7
            },
            'news': {
                'impact_scope': 'sector_wide',
                'analysis_horizon': 'immediate',
                'key_metrics': ['sentiment_change', 'price_movement', 'volume_increase'],
                'significance_threshold': 0.6
            },
            'economic_data': {
                'impact_scope': 'market_wide',
                'analysis_horizon': 'medium_term',
                'key_metrics': ['market_reaction', 'sector_rotation', 'volatility_change'],
                'significance_threshold': 0.8
            },
            'corporate_action': {
                'impact_scope': 'company_specific',
                'analysis_horizon': 'long_term',
                'key_metrics': ['price_impact', 'volume_pattern', 'sentiment_shift'],
                'significance_threshold': 0.75
            }
        }
        
        # AI Reasoning: Impact analysis thresholds and patterns
        self.impact_thresholds = {
            'price_movement': {'significant': 0.05, 'major': 0.10, 'extreme': 0.20},
            'volume_spike': {'significant': 2.0, 'major': 5.0, 'extreme': 10.0},
            'volatility_change': {'significant': 0.20, 'major': 0.50, 'extreme': 1.0},
            'sentiment_shift': {'significant': 0.30, 'major': 0.60, 'extreme': 0.80}
        }
        
        # AI Reasoning: Analysis timeframes and monitoring periods
        self.analysis_timeframes = {
            'immediate': {'pre_event_hours': 1, 'post_event_hours': 2, 'monitoring_frequency': '5min'},
            'short_term': {'pre_event_hours': 24, 'post_event_hours': 48, 'monitoring_frequency': '1hour'},
            'medium_term': {'pre_event_hours': 168, 'post_event_hours': 336, 'monitoring_frequency': '6hour'},
            'long_term': {'pre_event_hours': 720, 'post_event_hours': 1440, 'monitoring_frequency': '1day'}
        }
        
        self.agent_name = "event_impact_agent"
    
    async def check_knowledge_base_for_existing_data(self, event_id: str, impact_type: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Check knowledge base for existing event impact data
        - Query existing impact analysis and market reaction data
        - Assess data freshness and completeness
        - Determine if new analysis is needed
        - Identify data gaps and inconsistencies
        - NO TRADING DECISIONS - only data validation
        """
        # PSEUDOCODE:
        # 1. Query knowledge base for event's recent impact analysis
        # 2. Check last analysis timestamp and data freshness
        # 3. Assess analysis completeness against expected metrics
        # 4. Identify missing or outdated impact data
        # 5. Calculate confidence in existing analysis quality
        # 6. Determine if new analysis is warranted
        # 7. Return existing data with quality assessment
        # 8. NO TRADING DECISIONS - only data validation
        
        try:
            with engine.connect() as conn:
                # AI Reasoning: Intelligent query based on impact type and time range
                if impact_type:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'event_impact_agent' 
                        AND data->>'event_id' = :event_id 
                        AND data->>'impact_type' = :impact_type
                        ORDER BY event_time DESC 
                        LIMIT 20
                    """)
                    result = conn.execute(query, {"event_id": event_id, "impact_type": impact_type})
                else:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'event_impact_agent' 
                        AND data->>'event_id' = :event_id
                        ORDER BY event_time DESC 
                        LIMIT 50
                    """)
                    result = conn.execute(query, {"event_id": event_id})
                
                existing_data = [dict(row) for row in result]
                
                # AI Reasoning: Assess data quality and freshness
                data_quality = {
                    'total_records': len(existing_data),
                    'latest_analysis': existing_data[0]['event_time'] if existing_data else None,
                    'analysis_freshness_hours': None,
                    'completeness_score': 0.0,
                    'confidence_level': 0.0
                }
                
                if existing_data:
                    latest_time = existing_data[0]['event_time']
                    data_quality['analysis_freshness_hours'] = (datetime.utcnow() - latest_time).total_seconds() / 3600
                    
                    # AI Reasoning: Calculate completeness based on expected impact types
                    impact_types = [event['data'].get('impact_type') for event in existing_data]
                    data_quality['completeness_score'] = len(set(impact_types)) / len(self.impact_thresholds)
                    
                    # AI Reasoning: Assess confidence based on analysis consistency
                    data_quality['confidence_level'] = min(1.0, data_quality['completeness_score'] * 0.9)
                
                return {
                    'existing_data': existing_data,
                    'data_quality': data_quality,
                    'needs_analysis': data_quality['analysis_freshness_hours'] is None or data_quality['analysis_freshness_hours'] > 2.0
                }
                
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            return {'existing_data': [], 'data_quality': {}, 'needs_analysis': True}
    
    async def select_events_for_analysis(self, event_type: str = None) -> List[MarketEvent]:
        """
        AI Reasoning: Select events for impact analysis based on priority and timing
        - Evaluate event significance and timing
        - Match events to analysis requirements
        - Prioritize events based on expected impact
        - Consider analysis resources and constraints
        - NO TRADING DECISIONS - only event selection
        """
        # PSEUDOCODE:
        # 1. Analyze event significance and expected impact
        # 2. Evaluate event timing and market conditions
        # 3. Check event category and analysis requirements
        # 4. Assess analysis resources and time constraints
        # 5. Prioritize events based on significance and urgency
        # 6. Select optimal set of events for analysis
        # 7. Return prioritized list of events to analyze
        # 8. NO TRADING DECISIONS - only event selection
        
        # AI Reasoning: Example events for analysis
        priority_events = [
            MarketEvent(
                event_id="earnings_aapl_q4_2024",
                event_type="earnings",
                ticker="AAPL",
                sector="technology",
                event_time=datetime.utcnow() - timedelta(hours=2),
                event_description="Apple Q4 2024 earnings release",
                expected_impact="positive",
                ai_significance_score=0.8
            ),
            MarketEvent(
                event_id="fed_rate_decision_dec_2024",
                event_type="economic_data",
                ticker=None,
                sector=None,
                event_time=datetime.utcnow() - timedelta(hours=1),
                event_description="Federal Reserve interest rate decision",
                expected_impact="neutral",
                ai_significance_score=0.9
            ),
            MarketEvent(
                event_id="merger_announcement_tech_2024",
                event_type="corporate_action",
                ticker="TECH",
                sector="technology",
                event_time=datetime.utcnow() - timedelta(hours=3),
                event_description="Major tech company merger announcement",
                expected_impact="positive",
                ai_significance_score=0.7
            )
        ]
        
        if event_type:
            # AI Reasoning: Filter events by type
            filtered_events = [event for event in priority_events if event.event_type == event_type]
            return filtered_events[:3]  # Limit to top 3 events per type
        
        return priority_events[:5]  # Return top 5 priority events
    
    async def analyze_price_impact(self, event: MarketEvent, pre_event_data: Dict[str, Any], post_event_data: Dict[str, Any]) -> EventImpact:
        """
        AI Reasoning: Analyze price impact of market events
        - Calculate price changes before and after events
        - Assess price movement significance and patterns
        - Identify price reaction timing and magnitude
        - Evaluate price impact duration and recovery
        - NO TRADING DECISIONS - only price analysis
        """
        # PSEUDOCODE:
        # 1. Extract price data from pre and post event periods
        # 2. Calculate price changes and percentage movements
        # 3. Assess price movement significance against thresholds
        # 4. Identify price reaction patterns and timing
        # 5. Evaluate price impact duration and recovery patterns
        # 6. Generate comprehensive price impact analysis
        # 7. Score impact by magnitude and significance
        # 8. NO TRADING DECISIONS - only price analysis
        
        pre_price = pre_event_data.get('price', 100.0)
        post_price = post_event_data.get('price', 105.0)
        price_change = (post_price - pre_price) / pre_price
        
        # AI Reasoning: Assess price impact significance
        if abs(price_change) > self.impact_thresholds['price_movement']['extreme']:
            impact_level = 'extreme'
        elif abs(price_change) > self.impact_thresholds['price_movement']['major']:
            impact_level = 'major'
        elif abs(price_change) > self.impact_thresholds['price_movement']['significant']:
            impact_level = 'significant'
        else:
            impact_level = 'minor'
        
        price_impact = EventImpact(
            event_id=event.event_id,
            impact_type='price_movement',
            pre_event_data=pre_event_data,
            post_event_data=post_event_data,
            impact_magnitude=abs(price_change),
            impact_duration='immediate' if abs(price_change) > 0.05 else 'short_term',
            confidence_score=0.85,
            ai_relevance_score=0.8
        )
        
        return price_impact
    
    async def analyze_volume_impact(self, event: MarketEvent, pre_event_data: Dict[str, Any], post_event_data: Dict[str, Any]) -> EventImpact:
        """
        AI Reasoning: Analyze volume impact of market events
        - Calculate volume changes before and after events
        - Assess volume spike significance and patterns
        - Identify volume reaction timing and magnitude
        - Evaluate volume impact duration and normalization
        - NO TRADING DECISIONS - only volume analysis
        """
        # PSEUDOCODE:
        # 1. Extract volume data from pre and post event periods
        # 2. Calculate volume changes and relative increases
        # 3. Assess volume spike significance against thresholds
        # 4. Identify volume reaction patterns and timing
        # 5. Evaluate volume impact duration and normalization
        # 6. Generate comprehensive volume impact analysis
        # 7. Score impact by magnitude and significance
        # 8. NO TRADING DECISIONS - only volume analysis
        
        pre_volume = pre_event_data.get('volume', 1000000)
        post_volume = post_event_data.get('volume', 3000000)
        volume_ratio = post_volume / pre_volume if pre_volume > 0 else 1.0
        
        # AI Reasoning: Assess volume impact significance
        if volume_ratio > self.impact_thresholds['volume_spike']['extreme']:
            impact_level = 'extreme'
        elif volume_ratio > self.impact_thresholds['volume_spike']['major']:
            impact_level = 'major'
        elif volume_ratio > self.impact_thresholds['volume_spike']['significant']:
            impact_level = 'significant'
        else:
            impact_level = 'minor'
        
        volume_impact = EventImpact(
            event_id=event.event_id,
            impact_type='volume_spike',
            pre_event_data=pre_event_data,
            post_event_data=post_event_data,
            impact_magnitude=volume_ratio,
            impact_duration='immediate' if volume_ratio > 3.0 else 'short_term',
            confidence_score=0.80,
            ai_relevance_score=0.75
        )
        
        return volume_impact
    
    async def analyze_volatility_impact(self, event: MarketEvent, pre_event_data: Dict[str, Any], post_event_data: Dict[str, Any]) -> EventImpact:
        """
        AI Reasoning: Analyze volatility impact of market events
        - Calculate volatility changes before and after events
        - Assess volatility spike significance and patterns
        - Identify volatility reaction timing and magnitude
        - Evaluate volatility impact duration and normalization
        - NO TRADING DECISIONS - only volatility analysis
        """
        # PSEUDOCODE:
        # 1. Extract volatility data from pre and post event periods
        # 2. Calculate volatility changes and percentage movements
        # 3. Assess volatility spike significance against thresholds
        # 4. Identify volatility reaction patterns and timing
        # 5. Evaluate volatility impact duration and normalization
        # 6. Generate comprehensive volatility impact analysis
        # 7. Score impact by magnitude and significance
        # 8. NO TRADING DECISIONS - only volatility analysis
        
        pre_volatility = pre_event_data.get('volatility', 0.20)
        post_volatility = post_event_data.get('volatility', 0.35)
        volatility_change = (post_volatility - pre_volatility) / pre_volatility if pre_volatility > 0 else 0.0
        
        # AI Reasoning: Assess volatility impact significance
        if abs(volatility_change) > self.impact_thresholds['volatility_change']['extreme']:
            impact_level = 'extreme'
        elif abs(volatility_change) > self.impact_thresholds['volatility_change']['major']:
            impact_level = 'major'
        elif abs(volatility_change) > self.impact_thresholds['volatility_change']['significant']:
            impact_level = 'significant'
        else:
            impact_level = 'minor'
        
        volatility_impact = EventImpact(
            event_id=event.event_id,
            impact_type='volatility_change',
            pre_event_data=pre_event_data,
            post_event_data=post_event_data,
            impact_magnitude=abs(volatility_change),
            impact_duration='immediate' if abs(volatility_change) > 0.50 else 'short_term',
            confidence_score=0.82,
            ai_relevance_score=0.78
        )
        
        return volatility_impact
    
    async def determine_next_best_action(self, impact_results: List[EventImpact], event: MarketEvent) -> Dict[str, Any]:
        """
        AI Reasoning: Determine next best action based on event impact analysis
        - Evaluate impact results and identify significant effects
        - Decide on follow-up analysis requirements
        - Plan coordination with other agents
        - Schedule additional monitoring if needed
        - NO TRADING DECISIONS - only action planning
        """
        # PSEUDOCODE:
        # 1. Assess impact results and identify significant effects
        # 2. Evaluate impact magnitude and duration
        # 3. Determine if follow-up analysis is needed
        # 4. Plan coordination with related agents
        # 5. Schedule additional monitoring for significant impacts
        # 6. Prioritize actions based on impact significance
        # 7. Return action plan with priorities
        # 8. NO TRADING DECISIONS - only action planning
        
        next_actions = {
            'immediate_actions': [],
            'scheduled_actions': [],
            'coordination_needed': [],
            'priority_level': 'low'
        }
        
        # AI Reasoning: Evaluate impact significance
        for impact in impact_results:
            if impact.impact_magnitude > 0.10 and impact.confidence_score > 0.8:
                next_actions['priority_level'] = 'high'
                next_actions['immediate_actions'].append({
                    'action': 'notify_orchestrator',
                    'reason': 'significant_event_impact',
                    'data': impact
                })
                
                next_actions['coordination_needed'].append({
                    'agent': 'market_news_agent',
                    'reason': 'correlate_with_news_coverage',
                    'priority': 'high'
                })
                
                next_actions['coordination_needed'].append({
                    'agent': 'options_flow_agent',
                    'reason': 'check_options_activity_impact',
                    'priority': 'medium'
                })
        
        # AI Reasoning: Schedule follow-up monitoring for significant events
        if event.ai_significance_score > 0.7:
            next_actions['scheduled_actions'].append({
                'action': 'follow_up_monitoring',
                'schedule_hours': 24,
                'reason': 'monitor_event_impact_evolution'
            })
        
        return next_actions
    
    async def fetch_and_process_events(self):
        """
        AI Reasoning: Fetch and analyze event impacts
        - Retrieve events from various sources
        - Execute comprehensive impact analysis
        - Analyze market reactions and patterns
        - Store significant results in knowledge base
        - Coordinate with other agents as needed
        - NO TRADING DECISIONS - only event analysis
        """
        # PSEUDOCODE:
        # 1. Select events for impact analysis based on priority and timing
        # 2. Check knowledge base for existing impact analysis
        # 3. Execute comprehensive event impact analysis
        # 4. Analyze market reactions and patterns
        # 5. Calculate impact metrics and significance scores
        # 6. Store significant results in knowledge base
        # 7. Determine next actions and coordinate with agents
        # 8. NO TRADING DECISIONS - only event analysis
        
        try:
            # AI Reasoning: Select events for analysis
            events_to_analyze = await self.select_events_for_analysis()
            
            for event in events_to_analyze:
                # AI Reasoning: Check existing analysis data and determine analysis needs
                existing_data = await self.check_knowledge_base_for_existing_data(event.event_id)
                
                if not existing_data['needs_analysis']:
                    logger.info(f"Recent impact analysis exists for {event.event_id}, skipping analysis")
                    continue
                
                # AI Reasoning: Fetch market data for impact analysis
                pre_event_data = await self.fetch_market_data(event, 'pre_event')
                post_event_data = await self.fetch_market_data(event, 'post_event')
                
                if pre_event_data and post_event_data:
                    # AI Reasoning: Execute comprehensive impact analysis
                    impact_results = []
                    
                    # Price impact analysis
                    price_impact = await self.analyze_price_impact(event, pre_event_data, post_event_data)
                    impact_results.append(price_impact)
                    
                    # Volume impact analysis
                    volume_impact = await self.analyze_volume_impact(event, pre_event_data, post_event_data)
                    impact_results.append(volume_impact)
                    
                    # Volatility impact analysis
                    volatility_impact = await self.analyze_volatility_impact(event, pre_event_data, post_event_data)
                    impact_results.append(volatility_impact)
                    
                    # AI Reasoning: Store significant impact results in knowledge base
                    for impact in impact_results:
                        if impact.confidence_score > 0.7:
                            await self.store_in_knowledge_base(event.event_id, impact)
                    
                    # AI Reasoning: Determine next actions
                    next_actions = await self.determine_next_best_action(impact_results, event)
                    
                    # AI Reasoning: Execute immediate actions
                    for action in next_actions['immediate_actions']:
                        if action['action'] == 'notify_orchestrator':
                            await self.notify_orchestrator(action['data'])
                    
                    # AI Reasoning: Schedule follow-up actions
                    for action in next_actions['scheduled_actions']:
                        if action['action'] == 'follow_up_monitoring':
                            asyncio.create_task(self.schedule_follow_up_monitoring(event.event_id, action['schedule_hours']))
                
                # AI Reasoning: Rate limiting between events
                await asyncio.sleep(3)
                
        except Exception as e:
            logger.error(f"Error in fetch_and_process_events: {e}")
            # AI Reasoning: Implement error recovery and retry logic
            await self.handle_error_recovery(e)
    
    async def fetch_market_data(self, event: MarketEvent, period: str) -> Optional[Dict[str, Any]]:
        """
        AI Reasoning: Fetch market data for event impact analysis
        - Retrieve price, volume, and volatility data
        - Handle data source errors and rate limiting
        - Normalize data formats across sources
        - Apply quality filters and validation
        - NO TRADING DECISIONS - only data retrieval
        """
        # PSEUDOCODE:
        # 1. Determine data requirements based on event type and period
        # 2. Select appropriate data sources and timeframes
        # 3. Fetch market data from APIs or databases
        # 4. Apply data quality filters and validation
        # 5. Normalize data formats across sources
        # 6. Return processed market data
        # 7. NO TRADING DECISIONS - only data retrieval
        
        # AI Reasoning: Mock market data for demonstration
        if period == 'pre_event':
            return {
                'price': 100.0,
                'volume': 1000000,
                'volatility': 0.20,
                'timestamp': event.event_time - timedelta(hours=1)
            }
        else:  # post_event
            return {
                'price': 105.0,
                'volume': 3000000,
                'volatility': 0.35,
                'timestamp': event.event_time + timedelta(hours=1)
            }
    
    async def store_in_knowledge_base(self, event_id: str, impact: EventImpact):
        """
        AI Reasoning: Store significant event impact results in knowledge base
        - Store impact results with proper metadata
        - Include analysis metrics and confidence scores
        - Tag results for easy retrieval and analysis
        - NO TRADING DECISIONS - only data storage
        """
        # PSEUDOCODE:
        # 1. Prepare impact result data with metadata
        # 2. Include analysis metrics and confidence scores
        # 3. Store in knowledge base with proper indexing
        # 4. Tag results for correlation analysis
        # 5. Update impact tracking and statistics
        # 6. NO TRADING DECISIONS - only data storage
        
        try:
            event_data = {
                'event_id': event_id,
                'impact_type': impact.impact_type,
                'impact_magnitude': impact.impact_magnitude,
                'impact_duration': impact.impact_duration,
                'confidence_score': impact.confidence_score,
                'pre_event_data': impact.pre_event_data,
                'post_event_data': impact.post_event_data,
                'timestamp': datetime.utcnow().isoformat(),
                'agent_version': '1.0'
            }
            
            with engine.connect() as conn:
                query = text("""
                    INSERT INTO events (source_agent, event_type, event_time, data)
                    VALUES (:source_agent, :event_type, :event_time, :data)
                """)
                
                conn.execute(query, {
                    'source_agent': self.agent_name,
                    'event_type': 'event_impact_analysis',
                    'event_time': datetime.utcnow(),
                    'data': json.dumps(event_data)
                })
                conn.commit()
                
            logger.info(f"Stored event impact analysis for {event_id}")
            
        except Exception as e:
            logger.error(f"Error storing event impact data: {e}")
    
    async def notify_orchestrator(self, data: Dict[str, Any]):
        """
        AI Reasoning: Notify orchestrator of significant event impacts
        - Send critical impact results to orchestrator
        - Include impact metrics and confidence scores
        - Request coordination with other agents if needed
        - NO TRADING DECISIONS - only coordination
        """
        # PSEUDOCODE:
        # 1. Prepare notification data with impact results
        # 2. Include confidence scores and significance levels
        # 3. Send to orchestrator via MCP
        # 4. Request coordination with related agents
        # 5. NO TRADING DECISIONS - only coordination
        
        try:
            notification = {
                'agent': self.agent_name,
                'event_type': 'significant_event_impact',
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'priority': 'high' if data.get('confidence_score', 0.0) > 0.8 else 'medium'
            }
            
            # AI Reasoning: Send via MCP to orchestrator
            # await self.send_mcp_message('orchestrator', notification)
            logger.info(f"Notified orchestrator of significant event impact: {notification}")
            
        except Exception as e:
            logger.error(f"Error notifying orchestrator: {e}")
    
    async def schedule_follow_up_monitoring(self, event_id: str, delay_hours: int):
        """
        AI Reasoning: Schedule follow-up monitoring for event impact evolution
        - Schedule delayed monitoring for impact confirmation
        - Monitor impact evolution over time
        - Update impact analysis as new data arrives
        - NO TRADING DECISIONS - only monitoring scheduling
        """
        # PSEUDOCODE:
        # 1. Wait for specified delay period
        # 2. Re-analyze event impact with updated data
        # 3. Compare with previous impact analysis
        # 4. Update impact metrics and confidence
        # 5. Store updated impact analysis in knowledge base
        # 6. NO TRADING DECISIONS - only monitoring scheduling
        
        await asyncio.sleep(delay_hours * 3600)
        
        try:
            # AI Reasoning: Re-analyze event impact for evolution tracking
            existing_data = await self.check_knowledge_base_for_existing_data(event_id)
            
            if existing_data['existing_data']:
                # AI Reasoning: Update impact analysis with new data
                logger.info(f"Completed follow-up monitoring for {event_id}")
                
        except Exception as e:
            logger.error(f"Error in follow-up monitoring for {event_id}: {e}")
    
    async def handle_error_recovery(self, error: Exception):
        """
        AI Reasoning: Handle errors and implement recovery strategies
        - Log errors with context and severity
        - Implement retry logic with exponential backoff
        - Fall back to alternative analysis methods
        - Maintain system stability and data quality
        - NO TRADING DECISIONS - only error recovery
        """
        # PSEUDOCODE:
        # 1. Log error with full context and stack trace
        # 2. Assess error severity and impact
        # 3. Implement appropriate recovery strategy
        # 4. Retry with exponential backoff if applicable
        # 5. Fall back to alternative analysis methods
        # 6. Update system health metrics
        # 7. NO TRADING DECISIONS - only error recovery
        
        logger.error(f"Event impact agent error: {error}")
        
        # AI Reasoning: Implement retry logic for transient errors
        if "timeout" in str(error).lower():
            logger.info("Analysis timeout, implementing backoff strategy")
            await asyncio.sleep(60)  # Wait 1 minute before retry
        elif "data" in str(error).lower():
            logger.info("Data error, retrying with alternative sources")
            await asyncio.sleep(30)  # Wait 30 seconds before retry
    
    async def listen_for_mcp_messages(self):
        """
        AI Reasoning: Listen for MCP messages from orchestrator and other agents
        - Handle requests for event impact analysis
        - Respond to coordination requests
        - Process priority analysis requests
        - NO TRADING DECISIONS - only message handling
        """
        # PSEUDOCODE:
        # 1. Listen for incoming MCP messages
        # 2. Parse message type and priority
        # 3. Handle analysis requests for specific events
        # 4. Respond with current impact analysis results
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
        - Coordinate event impact analysis
        - Handle MCP communication
        - Implement error recovery and monitoring
        - Maintain system health and performance
        - NO TRADING DECISIONS - only system operation
        """
        # PSEUDOCODE:
        # 1. Initialize agent and establish connections
        # 2. Start MCP message listening
        # 3. Begin periodic event impact analysis
        # 4. Handle errors and implement recovery
        # 5. Monitor system health and performance
        # 6. Coordinate with other agents as needed
        # 7. NO TRADING DECISIONS - only system operation
        
        logger.info("Starting Event Impact Agent")
        
        while True:
            try:
                # AI Reasoning: Run main analysis cycle
                await self.fetch_and_process_events()
                
                # AI Reasoning: Handle MCP communication
                await self.listen_for_mcp_messages()
                
                # AI Reasoning: Wait before next cycle
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in main agent loop: {e}")
                await self.handle_error_recovery(e)
                await asyncio.sleep(60)  # Wait before retry

async def main():
    """AI Reasoning: Main entry point for Event Impact Agent"""
    agent = EventImpactAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 