"""
Investor Portfolio Tracking Agent - Multi-Tool Enhanced
- Integrates LangChain, Computer Use, LlamaIndex, Haystack, AutoGen
- Tracks notable investor portfolios including:
1. Congress people (e.g., Nancy Pelosi, disclosure requirements)
2. Hedge fund managers (e.g., Bill Ackman, Ray Dalio, Warren Buffett)
3. Institutional investors (pension funds, endowments)
4. Corporate insiders and executives
5. Celebrity investors and influencers

NO TRADING DECISIONS - Only data aggregation and analysis for informational purposes.
"""

# ============================================================================
# LANGCHAIN INTEGRATION IMPORTS
# ============================================================================
# PSEUDOCODE: Import LangChain components for agent orchestration
# from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.tools import BaseTool
# from langchain.callbacks import LangChainTracer
# from langchain.schema import BaseMessage, HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool
# from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
# from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

# ============================================================================
# COMPUTER USE IMPORTS
# ============================================================================
# PSEUDOCODE: Import Computer Use for dynamic tool selection
# from computer_use import ComputerUseToolSelector

# ============================================================================
# LLAMA INDEX IMPORTS
# ============================================================================
# PSEUDOCODE: Import LlamaIndex for RAG and knowledge base
# from llama_index import VectorStoreIndex, SimpleDirectoryReader

# ============================================================================
# HAYSTACK IMPORTS
# ============================================================================
# PSEUDOCODE: Import Haystack for document QA
# from haystack.pipelines import ExtractiveQAPipeline

# ============================================================================
# AUTOGEN IMPORTS
# ============================================================================
# PSEUDOCODE: Import AutoGen for multi-agent system
# from autogen import MultiAgentSystem

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

@dataclass
class InvestorProfile:
    """AI Reasoning: Comprehensive investor profile with tracking metadata"""
    name: str
    type: str  # congress, hedge_fund, institutional, insider, celebrity
    entity_id: str
    disclosure_requirements: List[str]
    tracking_frequency: str  # daily, weekly, monthly
    data_sources: List[str]
    last_updated: datetime
    portfolio_size: Optional[float] = None
    holdings_count: Optional[int] = None
    ai_confidence_score: float = 0.0

@dataclass
class PortfolioHolding:
    """AI Reasoning: Individual holding with analysis metadata"""
    ticker: str
    shares: Optional[int] = None
    value: Optional[float] = None
    percentage: Optional[float] = None
    acquisition_date: Optional[datetime] = None
    disclosure_date: Optional[datetime] = None
    transaction_type: Optional[str] = None  # buy, sell, hold
    ai_relevance_score: float = 0.0
    ai_analysis_notes: List[str] = None

class InvestorPortfolioAgent:
    """
    AI Reasoning: Intelligent investor portfolio tracking system with multi-tool integration
    - Track portfolio changes and holdings across different investor types
    - Analyze investment patterns and identify trends using LangChain orchestration
    - Monitor disclosure compliance and timing with Computer Use optimization
    - Identify potential conflicts of interest with LlamaIndex RAG
    - Coordinate with other agents for comprehensive analysis via AutoGen
    - NO TRADING DECISIONS - only data aggregation and analysis
    """
    
    def __init__(self):
        # LangChain LLM and memory for portfolio analysis
        # self.llm = ChatOpenAI(...)
        # self.memory = ConversationBufferWindowMemory(...)
        
        # Computer Use: dynamic data source selection
        # self.tool_selector = ComputerUseToolSelector(...)

        # LlamaIndex: RAG for portfolio knowledge base
        # self.llama_index = VectorStoreIndex.from_documents(...)
        # self.query_engine = self.llama_index.as_query_engine()

        # Haystack: document QA for portfolio analysis
        # self.haystack_pipeline = ExtractiveQAPipeline(...)

        # AutoGen: multi-agent coordination
        # self.multi_agent_system = MultiAgentSystem([...])

        # AI Reasoning: Investor profiles with tracking priorities
        self.investor_profiles = {
            # Congress people with disclosure requirements
            'nancy_pelosi': InvestorProfile(
                name="Nancy Pelosi",
                type="congress",
                entity_id="pelosi_nancy",
                disclosure_requirements=["STOCK Act", "45-day disclosure"],
                tracking_frequency="daily",
                data_sources=["House.gov", "OpenSecrets", "SEC Form 4"],
                last_updated=datetime.utcnow(),
                ai_confidence_score=0.9
            ),
            'bill_ackman': InvestorProfile(
                name="Bill Ackman",
                type="hedge_fund",
                entity_id="ackman_bill",
                disclosure_requirements=["13F", "13D/G"],
                tracking_frequency="quarterly",
                data_sources=["SEC 13F", "Pershing Square", "WhaleWisdom"],
                last_updated=datetime.utcnow(),
                ai_confidence_score=0.8
            ),
            'warren_buffett': InvestorProfile(
                name="Warren Buffett",
                type="hedge_fund",
                entity_id="buffett_warren",
                disclosure_requirements=["13F", "13D/G"],
                tracking_frequency="quarterly",
                data_sources=["SEC 13F", "Berkshire Hathaway", "WhaleWisdom"],
                last_updated=datetime.utcnow(),
                ai_confidence_score=0.9
            )
        }
        
        # AI Reasoning: Data source priorities and reliability scores
        self.data_sources = {
            'sec_13f': {'reliability': 0.95, 'update_frequency': 'quarterly', 'delay_days': 45},
            'sec_form4': {'reliability': 0.98, 'update_frequency': 'daily', 'delay_days': 2},
            'house_gov': {'reliability': 0.90, 'update_frequency': 'daily', 'delay_days': 1},
            'opensecrets': {'reliability': 0.85, 'update_frequency': 'weekly', 'delay_days': 7},
            'whalewisdom': {'reliability': 0.80, 'update_frequency': 'daily', 'delay_days': 1},
            'finviz': {'reliability': 0.75, 'update_frequency': 'daily', 'delay_days': 1}
        }
        
        # AI Reasoning: Portfolio analysis patterns and thresholds
        self.analysis_patterns = {
            'large_position_changes': {'threshold': 0.05, 'significance': 'high'},
            'new_positions': {'threshold': 0.01, 'significance': 'medium'},
            'sector_concentration': {'threshold': 0.30, 'significance': 'high'},
            'timing_patterns': {'threshold': 7, 'significance': 'medium'},  # days
            'conflict_indicators': {'threshold': 0.80, 'significance': 'critical'}
        }
        
        logger.info("Investor Portfolio Agent initialized with multi-tool integration")
    
    async def check_knowledge_base_for_existing_data(self, investor_id: str, ticker: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Check knowledge base for existing portfolio data
        - Query existing holdings and transactions
        - Assess data freshness and completeness
        - Determine if new data fetch is needed
        - Identify data gaps and inconsistencies
        - NO TRADING DECISIONS - only data validation
        """
        # PSEUDOCODE:
        # 1. Query knowledge base for investor's recent portfolio data
        # 2. Check last update timestamp and data freshness
        # 3. Assess data completeness against expected holdings
        # 4. Identify missing or outdated information
        # 5. Calculate confidence in existing data quality
        # 6. Determine if new data fetch is warranted
        # 7. Return existing data with quality assessment
        # 8. NO TRADING DECISIONS - only data validation
        
        try:
            with engine.connect() as conn:
                # AI Reasoning: Intelligent query based on investor type and requirements
                if ticker:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'investor_portfolio_agent' 
                        AND data->>'investor_id' = :investor_id 
                        AND data->>'ticker' = :ticker
                        ORDER BY event_time DESC 
                        LIMIT 10
                    """)
                    result = conn.execute(query, {"investor_id": investor_id, "ticker": ticker})
                else:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'investor_portfolio_agent' 
                        AND data->>'investor_id' = :investor_id
                        ORDER BY event_time DESC 
                        LIMIT 50
                    """)
                    result = conn.execute(query, {"investor_id": investor_id})
                
                existing_data = [dict(row) for row in result]
                
                # AI Reasoning: Assess data quality and freshness
                data_quality = {
                    'total_records': len(existing_data),
                    'latest_update': existing_data[0]['event_time'] if existing_data else None,
                    'data_freshness_hours': None,
                    'completeness_score': 0.0,
                    'confidence_score': 0.0
                }
                
                if existing_data:
                    latest_time = existing_data[0]['event_time']
                    data_quality['data_freshness_hours'] = (datetime.utcnow() - latest_time).total_seconds() / 3600
                    
                    # AI Reasoning: Calculate completeness based on investor type
                    investor_profile = self.investor_profiles.get(investor_id)
                    if investor_profile:
                        if investor_profile.type == 'congress':
                            # Congress disclosures should be very recent
                            data_quality['completeness_score'] = 0.9 if data_quality['data_freshness_hours'] < 24 else 0.3
                        elif investor_profile.type == 'hedge_fund':
                            # 13F filings are quarterly, so more lenient
                            data_quality['completeness_score'] = 0.8 if data_quality['data_freshness_hours'] < 720 else 0.4
                
                return {
                    'existing_data': existing_data,
                    'data_quality': data_quality,
                    'needs_update': data_quality['completeness_score'] < 0.7
                }
                
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            return {
                'existing_data': [],
                'data_quality': {'completeness_score': 0.0, 'confidence_score': 0.0},
                'needs_update': True
            }
    
    async def process_query_with_multi_tools(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        AI Reasoning: Process portfolio analysis query using multi-tool integration
        - Use LangChain for intelligent query parsing and orchestration
        - Apply Computer Use for dynamic data source selection
        - Leverage LlamaIndex for RAG and knowledge base lookups
        - Use Haystack for document QA and analysis
        - Coordinate with AutoGen for complex multi-agent workflows
        - NO TRADING DECISIONS - only data analysis orchestration
        """
        # PSEUDOCODE: Enhanced query processing with multi-tool integration
        # 1. Use LangChain to parse and classify the query
        # 2. Apply Computer Use to select optimal data sources and tools
        # 3. Use LlamaIndex to search existing portfolio knowledge base
        # 4. Apply Haystack for document QA if needed
        # 5. Use AutoGen for complex multi-agent coordination
        # 6. Aggregate and validate results
        # 7. Update LangChain memory and LlamaIndex knowledge base
        # 8. Return comprehensive analysis with multi-tool integration details
        
        try:
            # PSEUDOCODE: LangChain query parsing
            # parsed_query = self.llm.parse_query(query)
            # query_type = parsed_query.get('type', 'portfolio_analysis')
            
            # PSEUDOCODE: Computer Use tool selection
            # selected_tools = self.tool_selector.select_tools(query, available_tools)
            
            # PSEUDOCODE: LlamaIndex knowledge base search
            # kb_results = self.query_engine.query(query)
            
            # PSEUDOCODE: Haystack document QA
            # qa_results = self.haystack_pipeline.run(query=query, documents=[...])
            
            # PSEUDOCODE: AutoGen multi-agent coordination
            # if self._is_complex_portfolio_analysis(query):
            #     multi_agent_results = self.multi_agent_system.run(query)
            
            # PSEUDOCODE: Aggregate results
            # aggregated_results = self._aggregate_multi_tool_results([
            #     parsed_query, selected_tools, kb_results, qa_results, multi_agent_results
            # ])
            
            # PSEUDOCODE: Update memory and knowledge base
            # self.memory.save_context({"input": query}, {"output": str(aggregated_results)})
            # self.llama_index.add_document(aggregated_results)
            
            # Placeholder for multi-tool integration
            aggregated_results = {
                "query": query,
                "analysis_type": "portfolio_analysis",
                "multi_tool_integration": {
                    "langchain_parsing": "Query parsed and classified",
                    "computer_use_selection": "Optimal tools selected",
                    "llama_index_rag": "Knowledge base searched",
                    "haystack_qa": "Document analysis completed",
                    "autogen_coordination": "Multi-agent workflow executed"
                },
                "results": {
                    "portfolio_data": [],
                    "analysis_insights": [],
                    "recommendations": []
                }
            }
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error in multi-tool query processing: {e}")
            return {
                "error": str(e),
                "query": query,
                "multi_tool_integration": "Failed"
            }

    async def select_optimal_data_sources(self, investor_profile: InvestorProfile, data_type: str) -> List[str]:
        """
        AI Reasoning: Select optimal data sources using Computer Use optimization
        - Consider data reliability and freshness requirements
        - Balance accuracy vs. timeliness with intelligent selection
        - Account for disclosure requirements and delays
        - Use Computer Use for dynamic source optimization
        - NO TRADING DECISIONS - only data source optimization
        """
        # PSEUDOCODE: Enhanced data source selection with Computer Use
        # 1. Use Computer Use to analyze investor type and requirements
        # 2. Apply intelligent selection based on data type and freshness needs
        # 3. Evaluate data source reliability scores with context
        # 4. Consider availability, costs, and access patterns
        # 5. Select optimal combination for redundancy and accuracy
        # 6. Prioritize sources based on data type (holdings vs. transactions)
        # 7. Return ranked list of optimal data sources
        # 8. NO TRADING DECISIONS - only source selection
        
        # PSEUDOCODE: Computer Use tool selection
        # selected_sources = self.tool_selector.select_data_sources(
        #     investor_profile, data_type, self.data_sources
        # )
        
        optimal_sources = []
        
        # AI Reasoning: Source selection based on investor type
        if investor_profile.type == 'congress':
            # Congress people: prioritize official disclosures
            optimal_sources = ['house_gov', 'sec_form4', 'opensecrets']
        elif investor_profile.type == 'hedge_fund':
            # Hedge funds: prioritize SEC filings
            optimal_sources = ['sec_13f', 'sec_form4', 'whalewisdom']
        elif investor_profile.type == 'institutional':
            # Institutional: prioritize regulatory filings
            optimal_sources = ['sec_13f', 'sec_form4', 'finviz']
        
        # AI Reasoning: Filter by data type requirements
        if data_type == 'holdings':
            # Holdings data: prefer comprehensive sources
            optimal_sources = [s for s in optimal_sources if self.data_sources[s]['reliability'] > 0.85]
        elif data_type == 'transactions':
            # Transaction data: prefer timely sources
            optimal_sources = [s for s in optimal_sources if self.data_sources[s]['delay_days'] < 7]
        
        return optimal_sources[:3]  # Return top 3 sources
    
    async def analyze_portfolio_changes(self, old_holdings: List[PortfolioHolding], new_holdings: List[PortfolioHolding]) -> Dict[str, Any]:
        """
        AI Reasoning: Analyze portfolio changes for significant patterns
        - Identify large position changes and new positions
        - Detect sector concentration and diversification changes
        - Analyze timing patterns and potential conflicts
        - Calculate significance scores for changes
        - NO TRADING DECISIONS - only pattern analysis
        """
        # PSEUDOCODE:
        # 1. Compare old vs. new holdings for each ticker
        # 2. Calculate percentage changes in position sizes
        # 3. Identify new positions and complete exits
        # 4. Analyze sector distribution changes
        # 5. Check for timing patterns (e.g., before earnings, after news)
        # 6. Assess potential conflicts of interest
        # 7. Calculate significance scores for each change
        # 8. Generate analysis summary with confidence levels
        # 9. NO TRADING DECISIONS - only pattern analysis
        
        changes = {
            'new_positions': [],
            'increased_positions': [],
            'decreased_positions': [],
            'exited_positions': [],
            'sector_changes': {},
            'significant_changes': [],
            'timing_analysis': {},
            'conflict_indicators': [],
            'overall_significance_score': 0.0
        }
        
        # AI Reasoning: Create ticker mapping for comparison
        old_holdings_map = {h.ticker: h for h in old_holdings}
        new_holdings_map = {h.ticker: h for h in new_holdings}
        
        # AI Reasoning: Analyze position changes
        for ticker, new_holding in new_holdings_map.items():
            if ticker not in old_holdings_map:
                # New position
                changes['new_positions'].append({
                    'ticker': ticker,
                    'shares': new_holding.shares,
                    'value': new_holding.value,
                    'significance_score': min(new_holding.percentage or 0, 1.0)
                })
            else:
                old_holding = old_holdings_map[ticker]
                if old_holding.shares and new_holding.shares:
                    change_pct = (new_holding.shares - old_holding.shares) / old_holding.shares
                    
                    if change_pct > self.analysis_patterns['large_position_changes']['threshold']:
                        changes['increased_positions'].append({
                            'ticker': ticker,
                            'change_pct': change_pct,
                            'old_shares': old_holding.shares,
                            'new_shares': new_holding.shares,
                            'significance_score': min(abs(change_pct), 1.0)
                        })
                    elif change_pct < -self.analysis_patterns['large_position_changes']['threshold']:
                        changes['decreased_positions'].append({
                            'ticker': ticker,
                            'change_pct': change_pct,
                            'old_shares': old_holding.shares,
                            'new_shares': new_holding.shares,
                            'significance_score': min(abs(change_pct), 1.0)
                        })
        
        # AI Reasoning: Identify exited positions
        for ticker, old_holding in old_holdings_map.items():
            if ticker not in new_holdings_map:
                changes['exited_positions'].append({
                    'ticker': ticker,
                    'old_shares': old_holding.shares,
                    'old_value': old_holding.value,
                    'significance_score': min(old_holding.percentage or 0, 1.0)
                })
        
        # AI Reasoning: Calculate overall significance
        total_changes = len(changes['new_positions']) + len(changes['increased_positions']) + len(changes['decreased_positions']) + len(changes['exited_positions'])
        if total_changes > 0:
            avg_significance = sum([
                sum(c['significance_score'] for c in changes['new_positions']),
                sum(c['significance_score'] for c in changes['increased_positions']),
                sum(c['significance_score'] for c in changes['decreased_positions']),
                sum(c['significance_score'] for c in changes['exited_positions'])
            ]) / total_changes
            changes['overall_significance_score'] = avg_significance
        
        return changes
    
    async def determine_next_best_action(self, analysis_results: Dict[str, Any], investor_profile: InvestorProfile) -> Dict[str, Any]:
        """
        AI Reasoning: Determine optimal next action based on analysis results
        - Decide whether to continue analysis, fetch more data, or trigger other agents
        - Consider significance of findings and investor importance
        - Balance thoroughness vs. resource efficiency
        - NO TRADING DECISIONS - only action planning
        """
        # PSEUDOCODE:
        # 1. Assess significance of portfolio changes
        # 2. Consider investor importance and tracking priority
        # 3. Evaluate data completeness and quality
        # 4. Determine if additional analysis is warranted
        # 5. Decide which other agents to trigger based on findings
        # 6. Calculate confidence in next action decision
        # 7. Return action plan with reasoning
        # 8. NO TRADING DECISIONS - only action planning
        
        next_action = {
            'action_type': 'continue_analysis',
            'confidence': 0.8,
            'reasoning': [],
            'agents_to_trigger': [],
            'estimated_completion_time': 30
        }
        
        # AI Reasoning: Assess significance and determine action
        significance_score = analysis_results.get('overall_significance_score', 0.0)
        
        if significance_score > 0.7:
            # High significance: trigger multiple agents for comprehensive analysis
            next_action['action_type'] = 'trigger_comprehensive_analysis'
            next_action['agents_to_trigger'] = [
                'sec_filings_agent',  # Check for recent filings
                'market_news_agent',  # Look for related news
                'event_impact_agent',  # Assess potential market impact
                'social_media_nlp_agent'  # Monitor social sentiment
            ]
            next_action['reasoning'].append(f"High significance score ({significance_score:.2f}) warrants comprehensive analysis")
        
        elif significance_score > 0.3:
            # Medium significance: trigger selective agents
            next_action['action_type'] = 'trigger_selective_analysis'
            next_action['agents_to_trigger'] = [
                'market_news_agent',  # Check for recent news
                'event_impact_agent'  # Assess potential impact
            ]
            next_action['reasoning'].append(f"Medium significance score ({significance_score:.2f}) warrants selective analysis")
        
        else:
            # Low significance: continue with current analysis
            next_action['action_type'] = 'continue_analysis'
            next_action['reasoning'].append(f"Low significance score ({significance_score:.2f}), continuing current analysis")
        
        # AI Reasoning: Consider investor importance
        if investor_profile.type == 'congress':
            # Congress people: always trigger news agent for potential conflicts
            if 'market_news_agent' not in next_action['agents_to_trigger']:
                next_action['agents_to_trigger'].append('market_news_agent')
            next_action['reasoning'].append("Congress person tracking - checking for potential conflicts of interest")
        
        # AI Reasoning: Consider data quality
        if analysis_results.get('data_quality', {}).get('completeness_score', 0.0) < 0.7:
            next_action['action_type'] = 'fetch_additional_data'
            next_action['reasoning'].append("Low data completeness, fetching additional data")
        
        return next_action
    
    async def fetch_portfolio_data(self, investor_id: str, data_sources: List[str]) -> List[PortfolioHolding]:
        """
        AI Reasoning: Fetch portfolio data from multiple sources with intelligent handling
        - Prioritize data sources based on reliability and freshness
        - Handle API rate limits and failures gracefully
        - Validate and cross-reference data from multiple sources
        - NO TRADING DECISIONS - only data collection
        """
        # PSEUDOCODE:
        # 1. Initialize portfolio holdings list
        # 2. For each data source, attempt to fetch data
        # 3. Handle API rate limits with exponential backoff
        # 4. Validate data format and completeness
        # 5. Cross-reference data across sources for consistency
        # 6. Calculate confidence scores for each holding
        # 7. Return consolidated portfolio data
        # 8. NO TRADING DECISIONS - only data collection
        
        holdings = []
        
        # AI Reasoning: Fetch from multiple sources for redundancy
        for source in data_sources:
            try:
                # PSEUDOCODE for each source:
                # - Construct API request with proper authentication
                # - Handle rate limiting and retries
                # - Parse response and extract holdings
                # - Validate data format and ranges
                # - Add to holdings list with source metadata
                
                # Placeholder for actual API calls
                mock_holdings = [
                    PortfolioHolding(
                        ticker="AAPL",
                        shares=10000,
                        value=1500000.0,
                        percentage=0.15,
                        acquisition_date=datetime.utcnow() - timedelta(days=30),
                        disclosure_date=datetime.utcnow(),
                        transaction_type="buy",
                        ai_relevance_score=0.8,
                        ai_analysis_notes=["Large position in tech sector"]
                    )
                ]
                
                holdings.extend(mock_holdings)
                
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                continue
        
        return holdings
    
    async def process_portfolio_update(self, investor_id: str, ticker: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Main portfolio processing function with comprehensive analysis
        - Check existing data and determine update needs
        - Fetch new data from optimal sources
        - Analyze changes and patterns
        - Determine next actions and agent coordination
        - NO TRADING DECISIONS - only data processing
        """
        # PSEUDOCODE:
        # 1. Validate investor ID and get profile
        # 2. Check knowledge base for existing data
        # 3. Determine if update is needed based on data freshness
        # 4. Select optimal data sources for this investor
        # 5. Fetch new portfolio data from selected sources
        # 6. Compare with existing data to identify changes
        # 7. Analyze changes for significance and patterns
        # 8. Determine next best action based on findings
        # 9. Store results in knowledge base
        # 10. Coordinate with other agents if needed
        # 11. NO TRADING DECISIONS - only data processing
        
        logger.info(f"Processing portfolio update for investor: {investor_id}")
        
        # AI Reasoning: Validate investor and get profile
        investor_profile = self.investor_profiles.get(investor_id)
        if not investor_profile:
            return {
                'success': False,
                'error': f'Unknown investor: {investor_id}',
                'next_action': 'error_handling'
            }
        
        # AI Reasoning: Check existing data
        existing_data_result = await self.check_knowledge_base_for_existing_data(investor_id, ticker)
        
        # AI Reasoning: Determine if update is needed
        if not existing_data_result['needs_update']:
            return {
                'success': True,
                'message': 'Data is current, no update needed',
                'existing_data': existing_data_result['existing_data'],
                'next_action': 'return_results'
            }
        
        # AI Reasoning: Select optimal data sources
        data_sources = await self.select_optimal_data_sources(investor_profile, 'holdings')
        
        # AI Reasoning: Fetch new portfolio data
        new_holdings = await self.fetch_portfolio_data(investor_id, data_sources)
        
        # AI Reasoning: Extract old holdings from existing data
        old_holdings = []
        if existing_data_result['existing_data']:
            # PSEUDOCODE: Parse existing holdings from knowledge base data
            pass
        
        # AI Reasoning: Analyze portfolio changes
        analysis_results = await self.analyze_portfolio_changes(old_holdings, new_holdings)
        
        # AI Reasoning: Determine next best action
        next_action = await self.determine_next_best_action(analysis_results, investor_profile)
        
        # AI Reasoning: Store results in knowledge base
        event_data = {
            'investor_id': investor_id,
            'investor_name': investor_profile.name,
            'investor_type': investor_profile.type,
            'holdings': [vars(h) for h in new_holdings],
            'analysis_results': analysis_results,
            'data_sources': data_sources,
            'next_action': next_action,
            'ai_confidence_score': investor_profile.ai_confidence_score
        }
        
        # PSEUDOCODE: Store in knowledge base
        # await self.store_in_knowledge_base(event_data)
        
        return {
            'success': True,
            'investor_id': investor_id,
            'holdings_count': len(new_holdings),
            'analysis_results': analysis_results,
            'next_action': next_action,
            'agents_to_trigger': next_action['agents_to_trigger'],
            'disclaimer': 'NO TRADING DECISIONS - Data for informational purposes only'
        }

# AI Reasoning: Main agent execution function
async def main():
    """
    AI Reasoning: Main execution function with intelligent workflow management
    - Initialize agent and validate configuration
    - Process portfolio updates for tracked investors
    - Handle errors and coordinate with other agents
    - NO TRADING DECISIONS - only data processing
    """
    # PSEUDOCODE:
    # 1. Initialize agent and validate configuration
    # 2. Load investor profiles and tracking priorities
    # 3. Check system health and data source availability
    # 4. Process portfolio updates for each tracked investor
    # 5. Handle errors and implement recovery strategies
    # 6. Coordinate with other agents via MCP
    # 7. Log results and update system status
    # 8. NO TRADING DECISIONS - only data processing
    
    logger.info("Starting Investor Portfolio Agent")
    
    agent = InvestorPortfolioAgent()
    
    # AI Reasoning: Process updates for tracked investors
    for investor_id in agent.investor_profiles.keys():
        try:
            result = await agent.process_portfolio_update(investor_id)
            logger.info(f"Portfolio update result for {investor_id}: {result}")
            
            # AI Reasoning: Trigger other agents if needed
            if result.get('agents_to_trigger'):
                logger.info(f"Triggering agents: {result['agents_to_trigger']}")
                # PSEUDOCODE: Send MCP messages to trigger other agents
                
        except Exception as e:
            logger.error(f"Error processing {investor_id}: {e}")
            # PSEUDOCODE: Implement error recovery strategies
    
    logger.info("Investor Portfolio Agent completed")

if __name__ == "__main__":
    asyncio.run(main()) 