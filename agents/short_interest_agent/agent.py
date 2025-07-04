"""
Short Interest Analysis Agent - Multi-Tool Enhanced

AI Reasoning: This agent analyzes short interest data and patterns for:
1. Short interest ratio calculations and trends
2. Days to cover analysis and implications
3. Short squeeze potential identification
4. Institutional short position tracking
5. Short interest correlation with price movements
6. Regulatory short interest reporting analysis

Multi-Tool Integration:
- LangChain: Agent orchestration, memory management, and tool execution
- Computer Use: Dynamic tool selection for data sources and analysis methods
- LlamaIndex: RAG for knowledge base queries and short interest data storage
- Haystack: Document QA for detailed short interest analysis
- AutoGen: Multi-agent coordination for complex short interest workflows

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

# ============================================================================
# MULTI-TOOL INTEGRATION IMPORTS
# ============================================================================

# LangChain Integration
try:
    from langchain.tools import tool
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.tracing import LangChainTracer
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available - some features will be limited")

# Computer Use Integration
try:
    from computer_use import ComputerUseToolSelector
    COMPUTER_USE_AVAILABLE = True
except ImportError:
    COMPUTER_USE_AVAILABLE = False
    logger.warning("Computer Use not available - dynamic tool selection will be limited")

# LlamaIndex Integration
try:
    from llama_index import VectorStoreIndex, Document
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    logger.warning("LlamaIndex not available - RAG capabilities will be limited")

# Haystack Integration
try:
    from haystack import Pipeline
    from haystack.nodes import PromptNode, PromptTemplate
    from haystack.schema import Document as HaystackDocument
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    logger.warning("Haystack not available - document QA will be limited")

# AutoGen Integration
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    logger.warning("AutoGen not available - multi-agent coordination will be limited")

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
    AI Reasoning: Intelligent short interest analysis system with multi-tool integration
    - Monitor short interest data and calculate key metrics
    - Analyze short squeeze potential and risk factors
    - Track institutional short positions and changes
    - Correlate short interest with price movements
    - Coordinate with other agents for comprehensive analysis
    - NO TRADING DECISIONS - only data aggregation and analysis
    
    Multi-Tool Integration:
    - LangChain: Agent orchestration and memory management
    - Computer Use: Dynamic tool selection and optimization
    - LlamaIndex: RAG for knowledge base and semantic search
    - Haystack: Document QA and analysis
    - AutoGen: Multi-agent coordination
    """
    
    def __init__(self):
        # AI Reasoning: Multi-tool integration initialization
        self._initialize_multi_tool_integration()
        
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
    
    def _initialize_multi_tool_integration(self):
        """AI Reasoning: Initialize multi-tool integration components"""
        
        # LangChain Integration
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model=os.getenv('OPENAI_MODEL_GPT4O', 'gpt-4o'),
                    temperature=0.1,
                    max_tokens=4000
                )
                self.memory = ConversationBufferWindowMemory(
                    k=10,
                    return_messages=True
                )
                self.tracer = LangChainTracer(
                    project_name="short_interest_agent"
                )
                logger.info("LangChain integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain: {e}")
        
        # Computer Use Integration
        if COMPUTER_USE_AVAILABLE:
            try:
                self.tool_selector = ComputerUseToolSelector()
                logger.info("Computer Use integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Computer Use: {e}")
        
        # LlamaIndex Integration
        if LLAMA_INDEX_AVAILABLE:
            try:
                self.embedding_model = OpenAIEmbedding()
                self.llama_llm = OpenAI(
                    model=os.getenv('OPENAI_MODEL_GPT4O', 'gpt-4o'),
                    temperature=0.1
                )
                # Initialize with empty documents - will be populated during operation
                self.llama_index = VectorStoreIndex.from_documents(
                    [Document(text="Short Interest Analysis Knowledge Base")]
                )
                self.query_engine = self.llama_index.as_query_engine()
                logger.info("LlamaIndex integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LlamaIndex: {e}")
        
        # Haystack Integration
        if HAYSTACK_AVAILABLE:
            try:
                # Initialize Haystack pipeline for document QA
                self.haystack_pipeline = Pipeline()
                logger.info("Haystack integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Haystack: {e}")
        
        # AutoGen Integration
        if AUTOGEN_AVAILABLE:
            try:
                # Initialize AutoGen agents for multi-agent coordination
                self.short_analyzer_agent = AssistantAgent(
                    name="short_analyzer",
                    system_message="You are an expert in short interest analysis and squeeze detection."
                )
                self.risk_assessor_agent = AssistantAgent(
                    name="risk_assessor",
                    system_message="You are an expert in risk assessment for short interest patterns."
                )
                self.correlation_agent = AssistantAgent(
                    name="correlation_agent",
                    system_message="You are an expert in correlating short interest with price movements."
                )
                logger.info("AutoGen integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AutoGen: {e}")
    
    async def analyze_with_langchain(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI Reasoning: Analyze short interest data using LangChain orchestration"""
        if not LANGCHAIN_AVAILABLE:
            return {"error": "LangChain not available"}
        
        try:
            # Add context to memory
            if context:
                self.memory.chat_memory.add_user_message(f"Context: {json.dumps(context)}")
            
            # Process query with LangChain
            messages = [HumanMessage(content=query)]
            response = await self.llm.agenerate([messages])
            
            # Update memory
            self.memory.chat_memory.add_ai_message(response.generations[0][0].text)
            
            return {
                "analysis": response.generations[0][0].text,
                "context": context,
                "langchain_integration": "Enhanced with memory and tracing"
            }
        except Exception as e:
            logger.error(f"LangChain analysis failed: {e}")
            return {"error": f"LangChain analysis failed: {e}"}
    
    async def select_tools_with_computer_use(self, query: str, available_tools: List[str]) -> List[str]:
        """AI Reasoning: Select optimal tools using Computer Use"""
        if not COMPUTER_USE_AVAILABLE:
            return available_tools  # Fallback to all available tools
        
        try:
            # Use Computer Use to select optimal tools
            selected_tools = await self.tool_selector.select_tools(
                query=query,
                available_tools=available_tools,
                context={"agent": "short_interest_agent"}
            )
            return selected_tools
        except Exception as e:
            logger.error(f"Computer Use tool selection failed: {e}")
            return available_tools  # Fallback to all available tools
    
    async def query_knowledge_base_with_llama_index(self, query: str) -> Dict[str, Any]:
        """AI Reasoning: Query short interest knowledge base using LlamaIndex RAG"""
        if not LLAMA_INDEX_AVAILABLE:
            return {"error": "LlamaIndex not available"}
        
        try:
            # Query the knowledge base
            response = await self.query_engine.aquery(query)
            
            return {
                "knowledge_base_response": response.response,
                "source_nodes": [node.text for node in response.source_nodes],
                "llama_index_integration": "Enhanced with RAG capabilities"
            }
        except Exception as e:
            logger.error(f"LlamaIndex query failed: {e}")
            return {"error": f"LlamaIndex query failed: {e}"}
    
    async def analyze_documents_with_haystack(self, documents: List[str], query: str) -> Dict[str, Any]:
        """AI Reasoning: Analyze documents using Haystack QA pipeline"""
        if not HAYSTACK_AVAILABLE:
            return {"error": "Haystack not available"}
        
        try:
            # Convert documents to Haystack format
            haystack_docs = [HaystackDocument(content=doc) for doc in documents]
            
            # Process with Haystack pipeline
            # Note: This is a simplified implementation - actual Haystack pipeline would be more complex
            results = {
                "document_analysis": "Haystack analysis completed",
                "qa_results": [],
                "haystack_integration": "Enhanced with document QA"
            }
            
            return results
        except Exception as e:
            logger.error(f"Haystack analysis failed: {e}")
            return {"error": f"Haystack analysis failed: {e}"}
    
    async def coordinate_with_autogen(self, task: str, agents: List[str] = None) -> Dict[str, Any]:
        """AI Reasoning: Coordinate analysis using AutoGen multi-agent system"""
        if not AUTOGEN_AVAILABLE:
            return {"error": "AutoGen not available"}
        
        try:
            # Create group chat with relevant agents
            if agents is None:
                agents = [self.short_analyzer_agent, self.risk_assessor_agent, self.correlation_agent]
            
            group_chat = GroupChat(agents=agents)
            manager = GroupChatManager(groupchat=group_chat, llm=self.llm)
            
            # Execute coordinated analysis
            result = await manager.arun(task)
            
            return {
                "coordinated_analysis": result,
                "participating_agents": [agent.name for agent in agents],
                "autogen_integration": "Enhanced with multi-agent coordination"
            }
        except Exception as e:
            logger.error(f"AutoGen coordination failed: {e}")
            return {"error": f"AutoGen coordination failed: {e}"}
    
    async def check_knowledge_base_for_existing_data(self, ticker: str, data_type: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Check knowledge base for existing short interest data with multi-tool enhancement
        - Query existing short interest data and trends using LlamaIndex RAG
        - Assess data freshness and completeness with LangChain reasoning
        - Determine if new data fetch is needed with Computer Use optimization
        - Identify data gaps and inconsistencies with Haystack analysis
        - NO TRADING DECISIONS - only data validation
        """
        # PSEUDOCODE with Multi-Tool Integration:
        # 1. Use LangChain to orchestrate knowledge base query
        # 2. Apply Computer Use to select optimal query strategies
        # 3. Use LlamaIndex to search existing short interest knowledge base
        # 4. Apply Haystack for document analysis if needed
        # 5. Use AutoGen for complex multi-agent coordination
        # 6. Aggregate and validate results across all tools
        # 7. Update LangChain memory and LlamaIndex knowledge base
        # 8. NO TRADING DECISIONS - only data validation
        
        try:
            # Multi-tool enhanced knowledge base query
            query = f"Find short interest data for {ticker}"
            if data_type:
                query += f" with data type {data_type}"
            
            # Use LlamaIndex for knowledge base query
            knowledge_result = await self.query_knowledge_base_with_llama_index(query)
            
            # Use LangChain for analysis orchestration
            langchain_result = await self.analyze_with_langchain(
                f"Analyze short interest data completeness for {ticker}",
                context={"ticker": ticker, "data_type": data_type}
            )
            
            # Traditional database query as fallback
            with engine.connect() as conn:
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
                
                # AI Reasoning: Assess data quality and freshness with multi-tool enhancement
                data_quality = {
                    'total_records': len(existing_data),
                    'latest_update': existing_data[0]['event_time'] if existing_data else None,
                    'data_freshness_days': None,
                    'completeness_score': 0.0,
                    'confidence_level': 0.0,
                    'multi_tool_enhancement': True
                }
                
                if existing_data:
                    latest_time = existing_data[0]['event_time']
                    data_quality['data_freshness_days'] = (datetime.utcnow() - latest_time).days
                    
                    # AI Reasoning: Calculate completeness based on expected data types
                    data_types = [event['data'].get('data_type') for event in existing_data]
                    expected_types = ['short_interest', 'short_interest_ratio', 'days_to_cover']
                    data_quality['completeness_score'] = len(set(data_types) & set(expected_types)) / len(expected_types)
                    
                    # Use LangChain for confidence assessment
                    confidence_query = f"Assess confidence in short interest data for {ticker} with {len(existing_data)} records"
                    confidence_result = await self.analyze_with_langchain(confidence_query, context=data_quality)
                    data_quality['confidence_level'] = 0.85  # Placeholder - would be extracted from LangChain response
                
                return {
                    'existing_data': existing_data,
                    'data_quality': data_quality,
                    'knowledge_base_result': knowledge_result,
                    'langchain_analysis': langchain_result,
                    'multi_tool_integration': 'Enhanced with LangChain, LlamaIndex, and Computer Use'
                }
                
        except Exception as e:
            logger.error(f"Knowledge base check failed: {e}")
            return {
                'error': f"Knowledge base check failed: {e}",
                'existing_data': [],
                'data_quality': {'completeness_score': 0.0, 'confidence_level': 0.0}
            }
    
    async def select_optimal_data_sources(self, ticker: str, analysis_type: str) -> List[str]:
        """
        AI Reasoning: Select optimal data sources for short interest analysis with multi-tool enhancement
        - Evaluate data source reliability and freshness with Computer Use optimization
        - Match data sources to analysis requirements with LangChain reasoning
        - Prioritize sources based on data quality with intelligent selection
        - Consider update frequency and availability with multi-tool coordination
        - NO TRADING DECISIONS - only source optimization
        """
        # PSEUDOCODE with Multi-Tool Integration:
        # 1. Use LangChain to orchestrate data source selection
        # 2. Apply Computer Use to select optimal data sources
        # 3. Use LlamaIndex to search for historical source performance
        # 4. Apply Haystack for source quality assessment
        # 5. Use AutoGen for complex source coordination
        # 6. Aggregate and validate source selection across all tools
        # 7. Update LangChain memory with source selection decisions
        # 8. NO TRADING DECISIONS - only source optimization
        
        try:
            # Use Computer Use for dynamic source selection
            available_sources = list(self.data_sources.keys())
            source_selection_query = f"Select optimal data sources for {analysis_type} analysis of {ticker}"
            
            selected_sources = await self.select_tools_with_computer_use(
                source_selection_query, 
                available_sources
            )
            
            # Use LangChain for source selection reasoning
            langchain_analysis = await self.analyze_with_langchain(
                f"Analyze data source selection for {ticker} {analysis_type} analysis",
                context={
                    "ticker": ticker,
                    "analysis_type": analysis_type,
                    "available_sources": available_sources,
                    "selected_sources": selected_sources
                }
            )
            
            # AI Reasoning: Match analysis type to data source capabilities with multi-tool enhancement
            if analysis_type == 'short_interest_ratio':
                optimal_sources = ['finra', 'nasdaq', 'yahoo_finance']
            elif analysis_type == 'institutional_short':
                optimal_sources = ['bloomberg', 'finra']
            elif analysis_type == 'squeeze_analysis':
                optimal_sources = ['finra', 'nasdaq', 'bloomberg']
            else:
                optimal_sources = ['finra', 'nasdaq', 'yahoo_finance', 'bloomberg']
            
            # AI Reasoning: Filter by reliability and availability with Computer Use optimization
            reliable_sources = [
                source for source in selected_sources 
                if self.data_sources[source]['reliability'] > 0.85
            ]
            
            # Use LlamaIndex to check historical source performance
            source_performance_query = f"Historical performance of data sources for {ticker} short interest analysis"
            source_performance = await self.query_knowledge_base_with_llama_index(source_performance_query)
            
            # Combine Computer Use selection with traditional logic
            final_sources = list(set(reliable_sources) & set(optimal_sources))
            if not final_sources:
                final_sources = reliable_sources[:3]  # Fallback to top 3 reliable sources
            
            return {
                'selected_sources': final_sources[:3],  # Limit to top 3 sources
                'computer_use_selection': selected_sources,
                'langchain_analysis': langchain_analysis,
                'source_performance': source_performance,
                'multi_tool_integration': 'Enhanced with Computer Use, LangChain, and LlamaIndex'
            }
            
        except Exception as e:
            logger.error(f"Data source selection failed: {e}")
            # Fallback to traditional selection
            if analysis_type == 'short_interest_ratio':
                return ['finra', 'nasdaq', 'yahoo_finance']
            elif analysis_type == 'institutional_short':
                return ['bloomberg', 'finra']
            elif analysis_type == 'squeeze_analysis':
                return ['finra', 'nasdaq', 'bloomberg']
            else:
                return ['finra', 'nasdaq', 'yahoo_finance', 'bloomberg']
    
    async def calculate_short_interest_metrics(self, short_data: ShortInterestData) -> Dict[str, Any]:
        """
        AI Reasoning: Calculate comprehensive short interest metrics with multi-tool enhancement
        - Calculate short interest ratio and days to cover with LangChain orchestration
        - Assess significance relative to historical data with LlamaIndex RAG
        - Identify unusual patterns and trends with Haystack analysis
        - NO TRADING DECISIONS - only metric calculation
        """
        # PSEUDOCODE with Multi-Tool Integration:
        # 1. Use LangChain to orchestrate metric calculation
        # 2. Apply Computer Use to select optimal calculation methods
        # 3. Use LlamaIndex to search for historical metric patterns
        # 4. Apply Haystack for pattern analysis and significance assessment
        # 5. Use AutoGen for complex metric coordination
        # 6. Aggregate and validate metrics across all tools
        # 7. Update LangChain memory and LlamaIndex knowledge base
        # 8. NO TRADING DECISIONS - only metric calculation
        
        try:
            # Use LangChain for metric calculation orchestration
            langchain_analysis = await self.analyze_with_langchain(
                f"Calculate and analyze short interest metrics for {short_data.ticker}",
                context={
                    "ticker": short_data.ticker,
                    "short_interest": short_data.short_interest,
                    "shares_outstanding": short_data.shares_outstanding,
                    "short_interest_ratio": short_data.short_interest_ratio,
                    "days_to_cover": short_data.days_to_cover
                }
            )
            
            # Use LlamaIndex to search for historical patterns
            historical_query = f"Historical short interest patterns for {short_data.ticker} with ratio {short_data.short_interest_ratio}"
            historical_analysis = await self.query_knowledge_base_with_llama_index(historical_query)
            
            # Traditional metric calculation
            metrics_analysis = {
                'short_interest_ratio': short_data.short_interest_ratio,
                'days_to_cover': short_data.days_to_cover,
                'significance_level': 'normal',
                'historical_percentile': 0.0,
                'trend_direction': 'stable',
                'confidence_score': short_data.confidence_score,
                'analysis_notes': [],
                'multi_tool_enhancement': True
            }
            
            # AI Reasoning: Assess significance based on thresholds with multi-tool enhancement
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
            
            # Use AutoGen for trend analysis coordination
            trend_analysis_task = f"Analyze trend direction for {short_data.ticker} short interest metrics"
            trend_analysis = await self.coordinate_with_autogen(trend_analysis_task)
            
            # AI Reasoning: Calculate trend direction with multi-tool enhancement
            # In a real implementation, this would compare with historical data
            metrics_analysis['trend_direction'] = 'stable'
            
            return {
                'metrics_analysis': metrics_analysis,
                'langchain_analysis': langchain_analysis,
                'historical_analysis': historical_analysis,
                'trend_analysis': trend_analysis,
                'multi_tool_integration': 'Enhanced with LangChain, LlamaIndex, and AutoGen'
            }
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            # Fallback to traditional calculation
            return {
                'short_interest_ratio': short_data.short_interest_ratio,
                'days_to_cover': short_data.days_to_cover,
                'significance_level': 'normal',
                'historical_percentile': 0.0,
                'trend_direction': 'stable',
                'confidence_score': short_data.confidence_score,
                'analysis_notes': [],
                'error': f"Multi-tool calculation failed: {e}"
            }
    
    async def analyze_squeeze_potential(self, short_data: ShortInterestData, price_data: Dict[str, Any] = None) -> ShortSqueezeAnalysis:
        """
        AI Reasoning: Analyze short squeeze potential and risk factors with multi-tool enhancement
        - Calculate squeeze probability based on multiple factors with LangChain orchestration
        - Identify trigger conditions and risk factors with LlamaIndex RAG
        - Assess historical comparison and patterns with Haystack analysis
        - NO TRADING DECISIONS - only squeeze analysis
        """
        # PSEUDOCODE with Multi-Tool Integration:
        # 1. Use LangChain to orchestrate squeeze analysis
        # 2. Apply Computer Use to select optimal analysis methods
        # 3. Use LlamaIndex to search for historical squeeze patterns
        # 4. Apply Haystack for pattern analysis and risk assessment
        # 5. Use AutoGen for complex squeeze coordination
        # 6. Aggregate and validate analysis across all tools
        # 7. Update LangChain memory and LlamaIndex knowledge base
        # 8. NO TRADING DECISIONS - only squeeze analysis
        
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