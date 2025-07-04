"""
Discovery Agent - Multi-Tool Enhanced
====================================

This agent continuously generates context-aware market questions and coordinates
with other agents to get comprehensive answers, serving as a blueprint for
future cloud integration with 24/7 autonomous question generation.

CRITICAL SYSTEM POLICY: NO TRADING DECISIONS OR RECOMMENDATIONS
This agent only performs question generation, agent coordination, and
knowledge synthesis. No trading advice is provided.
"""

import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import requests
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import uuid

# Multi-Tool Integration Imports
from langchain.llms import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain.tracing import LangChainTracer

from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.storage.storage_context import StorageContext

from haystack import Pipeline
from haystack.nodes import PreProcessor, EmbeddingRetriever, PromptNode
from haystack.schema import Document as HaystackDocument

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Computer Use Integration
try:
    from computer_use import ComputerUseToolSelector
    COMPUTER_USE_AVAILABLE = True
except ImportError:
    COMPUTER_USE_AVAILABLE = False
    ComputerUseToolSelector = None

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
SYSTEM POLICY: This agent is STRICTLY for question generation and agent coordination.
NO TRADING DECISIONS should be made. All question generation is for research and
investigation purposes only.

AI REASONING: The agent should:
1. Generate context-aware market questions continuously
2. Coordinate with other agents to get comprehensive answers
3. Synthesize information from multiple agents
4. Identify knowledge gaps and generate follow-up questions
5. NEVER make buy/sell recommendations
6. NEVER provide trading advice
"""

@dataclass
class MarketQuestion:
    """Represents a generated market question with AI reasoning"""
    question_id: str
    question_text: str
    question_type: str  # 'anomaly', 'correlation', 'trend', 'gap', 'investigation'
    priority: int = 1
    confidence: float = 0.0
    target_agents: List[str] = None
    context: Dict[str, Any] = None
    generated_at: datetime = None
    status: str = 'pending'  # 'pending', 'in_progress', 'completed', 'failed'
    
    def __post_init__(self):
        if self.target_agents is None:
            self.target_agents = []
        if self.context is None:
            self.context = {}
        if self.generated_at is None:
            self.generated_at = datetime.now()

@dataclass
class QuestionResult:
    """Represents the result of a question investigation"""
    question_id: str
    answers: Dict[str, Any]
    synthesis: str
    confidence: float
    follow_up_questions: List[str]
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()

class DiscoveryAgent:
    """
    Intelligent question generation and agent coordination system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "DiscoveryAgent"
        self.version = "1.0.0"
        self.question_queue = []
        self.question_history = []
        self.agent_capabilities = {
            'equity_research_agent': {
                'capabilities': ['analyst_ratings', 'research_reports', 'target_prices'],
                'question_types': ['analyst_opinion', 'research_coverage', 'price_targets']
            },
            'sec_filings_agent': {
                'capabilities': ['financial_statements', 'risk_factors', 'management_discussion'],
                'question_types': ['financial_health', 'regulatory_compliance', 'business_risks']
            },
            'market_news_agent': {
                'capabilities': ['news_analysis', 'sentiment_analysis', 'event_tracking'],
                'question_types': ['market_sentiment', 'news_impact', 'event_analysis']
            },
            'insider_trading_agent': {
                'capabilities': ['insider_transactions', 'ownership_changes', 'form_4_analysis'],
                'question_types': ['insider_activity', 'ownership_patterns', 'insider_sentiment']
            },
            'social_media_nlp_agent': {
                'capabilities': ['social_sentiment', 'trend_analysis', 'influencer_tracking'],
                'question_types': ['social_sentiment', 'trend_analysis', 'public_opinion']
            },
            'fundamental_pricing_agent': {
                'capabilities': ['valuation_models', 'financial_metrics', 'price_analysis'],
                'question_types': ['valuation_analysis', 'price_fairness', 'financial_metrics']
            },
            'kpi_tracker_agent': {
                'capabilities': ['performance_metrics', 'earnings_tracking', 'growth_analysis'],
                'question_types': ['performance_analysis', 'earnings_quality', 'growth_trends']
            },
            'event_impact_agent': {
                'capabilities': ['event_analysis', 'impact_assessment', 'catalyst_tracking'],
                'question_types': ['event_impact', 'catalyst_analysis', 'timeline_analysis']
            },
            'options_flow_agent': {
                'capabilities': ['options_activity', 'flow_analysis', 'unusual_activity'],
                'question_types': ['options_sentiment', 'flow_analysis', 'unusual_activity']
            },
            'macro_calendar_agent': {
                'capabilities': ['economic_events', 'fed_analysis', 'macro_trends'],
                'question_types': ['macro_impact', 'economic_analysis', 'policy_effects']
            },
            'ml_model_testing_agent': {
                'capabilities': ['model_validation', 'prediction_analysis', 'algorithm_testing'],
                'question_types': ['prediction_accuracy', 'model_performance', 'algorithm_analysis']
            },
            'comparative_analysis_agent': {
                'capabilities': ['peer_comparison', 'benchmark_analysis', 'relative_valuation'],
                'question_types': ['peer_analysis', 'benchmark_comparison', 'relative_performance']
            }
        }
        
        # MCP Communication setup
        self.mcp_endpoint = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8000/mcp')
        self.message_queue = []
        
        # Error handling and recovery
        self.error_count = 0
        self.max_retries = 3
        self.health_score = 1.0
        
        # Question generation metrics
        self.questions_generated = 0
        self.questions_completed = 0
        self.agent_coordination_count = 0
        
        # Multi-Tool Integration
        self._initialize_langchain()
        self._initialize_llama_index()
        self._initialize_haystack()
        self._initialize_autogen()
        self._initialize_computer_use()
        
        # Performance tracking
        self.health_score = 1.0
        self.last_update = datetime.now()
        self.error_count = 0
        
        logger.info(f"Initialized {self.name} v{self.version} with multi-tool integration")

    def _initialize_langchain(self):
        """Initialize LangChain for agent orchestration"""
        try:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10
            )
            
            # Create tools for question generation and coordination
            self.tools = [
                Tool(
                    name="generate_questions",
                    func=self._generate_questions_tool,
                    description="Generate context-aware market questions"
                ),
                Tool(
                    name="coordinate_agents",
                    func=self._coordinate_agents_tool,
                    description="Coordinate with multiple agents for comprehensive answers"
                ),
                Tool(
                    name="synthesize_answers",
                    func=self._synthesize_answers_tool,
                    description="Synthesize answers from multiple agents"
                )
            ]
            
            # Create agent executor
            prompt = PromptTemplate.from_template(
                "You are a discovery agent expert. Use the available tools to generate questions and coordinate with other agents.\n\n"
                "Available tools: {tools}\n"
                "Chat history: {chat_history}\n"
                "Question: {input}\n"
                "Answer:"
            )
            
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=5
            )
            
            logger.info("LangChain integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain: {e}")
            self.agent_executor = None

    def _initialize_llama_index(self):
        """Initialize LlamaIndex for knowledge base management"""
        try:
            # Initialize embedding model
            embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
                embed_model=embed_model
            )
            
            # Initialize storage context
            storage_context = StorageContext.from_defaults()
            
            # Create vector store index
            self.llama_index = VectorStoreIndex(
                [],
                service_context=service_context,
                storage_context=storage_context
            )
            
            # Create query engine
            self.query_engine = self.llama_index.as_query_engine(
                response_mode="compact",
                streaming=True
            )
            
            logger.info("LlamaIndex integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex: {e}")
            self.query_engine = None

    def _initialize_haystack(self):
        """Initialize Haystack for document QA"""
        try:
            # Create preprocessing pipeline
            self.preprocessor = PreProcessor(
                clean_empty_lines=True,
                clean_whitespace=True,
                clean_header_footer=True,
                split_by="word",
                split_length=500,
                split_overlap=50
            )
            
            # Create embedding retriever
            self.retriever = EmbeddingRetriever(
                document_store=None,  # Will be set when document store is available
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                model_format="sentence_transformers"
            )
            
            # Create prompt node for QA
            self.prompt_node = PromptNode(
                model_name_or_path="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                default_prompt_template="question-answering"
            )
            
            # Create QA pipeline
            self.qa_pipeline = Pipeline()
            self.qa_pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            self.qa_pipeline.add_node(component=self.prompt_node, name="PromptNode", inputs=["Retriever"])
            
            logger.info("Haystack integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Haystack: {e}")
            self.qa_pipeline = None

    def _initialize_autogen(self):
        """Initialize AutoGen for multi-agent coordination"""
        try:
            # Create question generation assistant
            self.question_assistant = AssistantAgent(
                name="question_generator",
                system_message="You are an expert question generator. Generate context-aware market questions for investigation.",
                llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
            )
            
            # Create coordination assistant
            self.coordination_assistant = AssistantAgent(
                name="coordination_specialist",
                system_message="You are an expert in coordinating with multiple agents to get comprehensive answers.",
                llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
            )
            
            # Create user proxy
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,
                llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
            )
            
            # Create group chat
            self.group_chat = GroupChat(
                agents=[self.user_proxy, self.question_assistant, self.coordination_assistant],
                messages=[],
                max_round=10
            )
            
            # Create group chat manager
            self.chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
            )
            
            logger.info("AutoGen integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen: {e}")
            self.chat_manager = None

    def _initialize_computer_use(self):
        """Initialize Computer Use for dynamic tool selection"""
        try:
            if COMPUTER_USE_AVAILABLE:
                self.tool_selector = ComputerUseToolSelector(
                    available_tools=self.tools,
                    optimization_strategy="performance"
                )
                logger.info("Computer Use integration initialized successfully")
            else:
                self.tool_selector = None
                logger.warning("Computer Use not available, using default tool selection")
                
        except Exception as e:
            logger.error(f"Failed to initialize Computer Use: {e}")
            self.tool_selector = None
        
    async def initialize(self):
        """Initialize the discovery agent"""
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # AI REASONING: Initialize question generation capabilities
        # PSEUDOCODE:
        # 1. Load question generation models and patterns
        # 2. Initialize agent capability mapping and coordination
        # 3. Set up question queue and priority management
        # 4. Configure continuous question generation parameters
        # 5. Initialize knowledge gap detection algorithms
        # 6. Set up MCP communication for agent coordination
        # 7. Configure question synthesis and follow-up logic
        # 8. Initialize monitoring and health tracking
        
        logger.info(f"{self.name} initialized successfully")
        
    async def generate_context_aware_questions(self) -> List[MarketQuestion]:
        """
        Generate context-aware market questions based on current data and trends
        """
        # AI REASONING: Context-aware question generation
        # PSEUDOCODE:
        # 1. Analyze current market conditions and trends
        # 2. Identify anomalies, patterns, and knowledge gaps
        # 3. Generate questions based on detected patterns:
        #    - Anomaly detection: "Why did X stock show unusual volume?"
        #    - Correlation analysis: "What's driving the correlation between A and B?"
        #    - Trend investigation: "What factors are behind this trend?"
        #    - Gap analysis: "What information is missing about this event?"
        # 4. Prioritize questions by significance and urgency
        # 5. Assign appropriate target agents for each question
        # 6. Calculate confidence scores for question relevance
        # 7. Queue questions for processing
        # 8. NO TRADING DECISIONS - only question generation
        
        questions = []
        
        # Generate anomaly-based questions
        anomaly_questions = await self._generate_anomaly_questions()
        questions.extend(anomaly_questions)
        
        # Generate correlation-based questions
        correlation_questions = await self._generate_correlation_questions()
        questions.extend(correlation_questions)
        
        # Generate trend-based questions
        trend_questions = await self._generate_trend_questions()
        questions.extend(trend_questions)
        
        # Generate gap-based questions
        gap_questions = await self._generate_gap_questions()
        questions.extend(gap_questions)
        
        # Add questions to queue
        self.question_queue.extend(questions)
        self.questions_generated += len(questions)
        
        logger.info(f"Generated {len(questions)} context-aware questions")
        return questions
    
    async def _generate_anomaly_questions(self) -> List[MarketQuestion]:
        """
        Generate questions based on detected anomalies
        """
        # AI REASONING: Anomaly-based question generation
        # PSEUDOCODE:
        # 1. Query knowledge base for recent anomalies
        # 2. Identify unusual patterns in volume, price, sentiment
        # 3. Generate specific questions about each anomaly
        # 4. Assign target agents based on anomaly type
        # 5. Calculate priority based on anomaly significance
        
        questions = []
        
        # Example anomaly questions (in production, would query knowledge base)
        anomaly_patterns = [
            {
                'pattern': 'unusual_volume',
                'question': 'What's driving the unusual volume spike in {ticker}?',
                'target_agents': ['market_news_agent', 'options_flow_agent', 'insider_trading_agent'],
                'priority': 8
            },
            {
                'pattern': 'price_movement',
                'question': 'What factors are behind the significant price movement in {ticker}?',
                'target_agents': ['equity_research_agent', 'market_news_agent', 'fundamental_pricing_agent'],
                'priority': 7
            },
            {
                'pattern': 'sentiment_shift',
                'question': 'What's causing the sentiment shift for {ticker}?',
                'target_agents': ['social_media_nlp_agent', 'market_news_agent', 'event_impact_agent'],
                'priority': 6
            }
        ]
        
        for pattern in anomaly_patterns:
            question = MarketQuestion(
                question_id=str(uuid.uuid4()),
                question_text=pattern['question'],
                question_type='anomaly',
                priority=pattern['priority'],
                confidence=0.8,
                target_agents=pattern['target_agents'],
                context={'pattern_type': pattern['pattern']}
            )
            questions.append(question)
        
        return questions
    
    async def _generate_correlation_questions(self) -> List[MarketQuestion]:
        """
        Generate questions based on detected correlations
        """
        # AI REASONING: Correlation-based question generation
        # PSEUDOCODE:
        # 1. Analyze correlations between different metrics
        # 2. Identify unexpected or significant correlations
        # 3. Generate questions about correlation drivers
        # 4. Assign comparative analysis agent for investigation
        
        questions = []
        
        correlation_questions = [
            {
                'question': 'What's driving the correlation between {metric1} and {metric2}?',
                'target_agents': ['comparative_analysis_agent', 'fundamental_pricing_agent'],
                'priority': 6
            },
            {
                'question': 'Why are {sector1} and {sector2} moving in opposite directions?',
                'target_agents': ['macro_calendar_agent', 'comparative_analysis_agent'],
                'priority': 7
            }
        ]
        
        for corr in correlation_questions:
            question = MarketQuestion(
                question_id=str(uuid.uuid4()),
                question_text=corr['question'],
                question_type='correlation',
                priority=corr['priority'],
                confidence=0.7,
                target_agents=corr['target_agents'],
                context={'analysis_type': 'correlation'}
            )
            questions.append(question)
        
        return questions
    
    async def _generate_trend_questions(self) -> List[MarketQuestion]:
        """
        Generate questions based on detected trends
        """
        # AI REASONING: Trend-based question generation
        # PSEUDOCODE:
        # 1. Identify emerging trends in market data
        # 2. Generate questions about trend sustainability
        # 3. Investigate factors driving trends
        # 4. Assign appropriate agents for trend analysis
        
        questions = []
        
        trend_questions = [
            {
                'question': 'What factors are sustaining this trend in {sector}?',
                'target_agents': ['kpi_tracker_agent', 'macro_calendar_agent', 'event_impact_agent'],
                'priority': 6
            },
            {
                'question': 'How long is this trend likely to continue?',
                'target_agents': ['ml_model_testing_agent', 'fundamental_pricing_agent'],
                'priority': 5
            }
        ]
        
        for trend in trend_questions:
            question = MarketQuestion(
                question_id=str(uuid.uuid4()),
                question_text=trend['question'],
                question_type='trend',
                priority=trend['priority'],
                confidence=0.6,
                target_agents=trend['target_agents'],
                context={'analysis_type': 'trend'}
            )
            questions.append(question)
        
        return questions
    
    async def _generate_gap_questions(self) -> List[MarketQuestion]:
        """
        Generate questions based on knowledge gaps
        """
        # AI REASONING: Gap-based question generation
        # PSEUDOCODE:
        # 1. Identify missing information in knowledge base
        # 2. Generate questions to fill knowledge gaps
        # 3. Prioritize gaps by importance and urgency
        # 4. Assign agents to investigate gaps
        
        questions = []
        
        gap_questions = [
            {
                'question': 'What information is missing about {event}?',
                'target_agents': ['sec_filings_agent', 'market_news_agent', 'event_impact_agent'],
                'priority': 7
            },
            {
                'question': 'Why is there limited coverage on {ticker}?',
                'target_agents': ['equity_research_agent', 'social_media_nlp_agent'],
                'priority': 5
            }
        ]
        
        for gap in gap_questions:
            question = MarketQuestion(
                question_id=str(uuid.uuid4()),
                question_text=gap['question'],
                question_type='gap',
                priority=gap['priority'],
                confidence=0.5,
                target_agents=gap['target_agents'],
                context={'analysis_type': 'gap'}
            )
            questions.append(question)
        
        return questions
    
    async def coordinate_with_agents(self, question: MarketQuestion) -> QuestionResult:
        """
        Coordinate with target agents to get comprehensive answers
        """
        # AI REASONING: Agent coordination and answer synthesis
        # PSEUDOCODE:
        # 1. Send question to each target agent via MCP
        # 2. Collect responses from all agents
        # 3. Synthesize answers into comprehensive response
        # 4. Identify conflicting information and resolve
        # 5. Generate follow-up questions based on gaps
        # 6. Calculate confidence in synthesized answer
        # 7. Update question status and store result
        # 8. NO TRADING DECISIONS - only information synthesis
        
        logger.info(f"Coordinating with agents for question: {question.question_id}")
        
        # Send question to target agents
        agent_responses = {}
        for agent in question.target_agents:
            response = await self._query_agent(agent, question)
            if response:
                agent_responses[agent] = response
        
        # Synthesize answers
        synthesis = await self._synthesize_answers(question, agent_responses)
        
        # Generate follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(question, agent_responses)
        
        # Calculate confidence
        confidence = self._calculate_synthesis_confidence(agent_responses, synthesis)
        
        result = QuestionResult(
            question_id=question.question_id,
            answers=agent_responses,
            synthesis=synthesis,
            confidence=confidence,
            follow_up_questions=follow_up_questions
        )
        
        # Update question status
        question.status = 'completed'
        self.questions_completed += 1
        self.agent_coordination_count += len(agent_responses)
        
        return result
    
    async def _query_agent(self, agent_name: str, question: MarketQuestion) -> Optional[Dict[str, Any]]:
        """
        Query a specific agent for answer to question
        """
        # AI REASONING: Agent querying and response handling
        # PSEUDOCODE:
        # 1. Format question for specific agent capabilities
        # 2. Send MCP message to agent
        # 3. Handle response and error cases
        # 4. Validate response quality and relevance
        # 5. Return formatted response or None
        
        try:
            # Format question for agent
            formatted_question = self._format_question_for_agent(agent_name, question)
            
            # Send MCP message (placeholder implementation)
            # response = await self.send_mcp_message(agent_name, formatted_question)
            
            # Mock response for demonstration
            response = {
                'agent': agent_name,
                'answer': f"Response from {agent_name} about {question.question_text}",
                'confidence': 0.8,
                'data_sources': ['source1', 'source2'],
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying agent {agent_name}: {e}")
            return None
    
    def _format_question_for_agent(self, agent_name: str, question: MarketQuestion) -> Dict[str, Any]:
        """
        Format question for specific agent capabilities
        """
        # AI REASONING: Question formatting for agent optimization
        # PSEUDOCODE:
        # 1. Extract agent capabilities and question types
        # 2. Format question to match agent's expertise
        # 3. Add relevant context and parameters
        # 4. Optimize question for agent's processing style
        
        agent_capabilities = self.agent_capabilities.get(agent_name, {})
        
        return {
            'question': question.question_text,
            'question_type': question.question_type,
            'target_capabilities': agent_capabilities.get('capabilities', []),
            'context': question.context,
            'priority': question.priority
        }
    
    async def _synthesize_answers(self, question: MarketQuestion, agent_responses: Dict[str, Any]) -> str:
        """
        Synthesize answers from multiple agents into comprehensive response
        """
        # AI REASONING: Answer synthesis and conflict resolution
        # PSEUDOCODE:
        # 1. Analyze responses from all agents
        # 2. Identify common themes and patterns
        # 3. Resolve conflicts and contradictions
        # 4. Combine complementary information
        # 5. Generate comprehensive synthesis
        # 6. Highlight areas of agreement and disagreement
        # 7. Provide confidence levels for different aspects
        
        if not agent_responses:
            return "No responses received from target agents."
        
        synthesis_parts = []
        
        # Analyze each agent's response
        for agent, response in agent_responses.items():
            synthesis_parts.append(f"{agent}: {response.get('answer', 'No answer provided')}")
        
        # Combine responses
        synthesis = f"Comprehensive analysis of '{question.question_text}':\n"
        synthesis += "\n".join(synthesis_parts)
        
        return synthesis
    
    async def _generate_follow_up_questions(self, question: MarketQuestion, agent_responses: Dict[str, Any]) -> List[str]:
        """
        Generate follow-up questions based on agent responses
        """
        # AI REASONING: Follow-up question generation
        # PSEUDOCODE:
        # 1. Analyze agent responses for gaps and inconsistencies
        # 2. Identify areas needing deeper investigation
        # 3. Generate specific follow-up questions
        # 4. Prioritize follow-up questions by importance
        # 5. Assign appropriate agents for follow-up
        
        follow_up_questions = []
        
        # Analyze responses for gaps
        if len(agent_responses) < len(question.target_agents):
            missing_agents = set(question.target_agents) - set(agent_responses.keys())
            for agent in missing_agents:
                follow_up_questions.append(f"Get response from {agent} about {question.question_text}")
        
        # Generate additional follow-up questions based on response content
        follow_up_questions.extend([
            "What are the underlying factors driving this pattern?",
            "How does this compare to historical patterns?",
            "What are the potential implications for future performance?"
        ])
        
        return follow_up_questions
    
    def _calculate_synthesis_confidence(self, agent_responses: Dict[str, Any], synthesis: str) -> float:
        """
        Calculate confidence in synthesized answer
        """
        # AI REASONING: Confidence calculation
        # PSEUDOCODE:
        # 1. Assess response quality from each agent
        # 2. Calculate agreement level between agents
        # 3. Consider data freshness and source reliability
        # 4. Factor in synthesis completeness
        # 5. Return overall confidence score
        
        if not agent_responses:
            return 0.0
        
        # Calculate average agent confidence
        agent_confidences = [response.get('confidence', 0.5) for response in agent_responses.values()]
        avg_confidence = sum(agent_confidences) / len(agent_confidences)
        
        # Factor in response completeness
        completeness_factor = len(agent_responses) / max(len(agent_responses), 1)
        
        return avg_confidence * completeness_factor
    
    async def process_question_queue(self):
        """
        Process the question queue and coordinate with agents
        """
        # AI REASONING: Queue processing and prioritization
        # PSEUDOCODE:
        # 1. Sort questions by priority and timestamp
        # 2. Process high-priority questions first
        # 3. Coordinate with agents for each question
        # 4. Update question status and results
        # 5. Generate new questions based on results
        # 6. Maintain queue health and performance
        
        # Sort questions by priority
        self.question_queue.sort(key=lambda q: (q.priority, q.generated_at), reverse=True)
        
        # Process questions (limit to avoid overwhelming system)
        questions_to_process = self.question_queue[:5]
        
        for question in questions_to_process:
            if question.status == 'pending':
                question.status = 'in_progress'
                
                # Coordinate with agents
                result = await self.coordinate_with_agents(question)
                
                # Store result
                self.question_history.append(result)
                
                # Generate follow-up questions
                for follow_up in result.follow_up_questions:
                    new_question = MarketQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=follow_up,
                        question_type='follow_up',
                        priority=question.priority - 1,
                        confidence=0.6,
                        target_agents=question.target_agents
                    )
                    self.question_queue.append(new_question)
        
        # Remove processed questions
        self.question_queue = [q for q in self.question_queue if q.status == 'pending']
    
    async def run(self):
        """
        Main execution loop for continuous question generation and coordination
        """
        logger.info(f"Starting {self.name} with continuous question generation")
        
        # PSEUDOCODE for main execution loop:
        # 1. Initialize question generation capabilities
        # 2. Start continuous question generation loop:
        #    - Generate context-aware questions
        #    - Process question queue
        #    - Coordinate with agents
        #    - Synthesize answers and generate follow-ups
        #    - Update system health and metrics
        # 3. Monitor system performance and adjust frequency
        # 4. Handle errors and recovery
        # 5. NO TRADING DECISIONS - only question generation and coordination
        
        while True:
            try:
                # Generate new questions
                await self.generate_context_aware_questions()
                
                # Process question queue
                await self.process_question_queue()
                
                # Update health metrics
                await self.update_health_metrics()
                
                # Sleep interval based on question generation frequency
                sleep_interval = self.calculate_sleep_interval()
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                await self.handle_error(e, "main_loop")
                await asyncio.sleep(60)
    
    async def update_health_metrics(self):
        """Update agent health and performance metrics"""
        # AI REASONING: Health monitoring and optimization
        # PSEUDOCODE:
        # 1. Calculate question generation rate and success
        # 2. Monitor agent coordination effectiveness
        # 3. Track response quality and synthesis accuracy
        # 4. Update health score based on performance
        # 5. Identify optimization opportunities
        
        self.health_score = min(1.0, self.questions_completed / max(self.questions_generated, 1))
        
        logger.info(f"Health metrics: {self.questions_generated} generated, {self.questions_completed} completed, health: {self.health_score:.2f}")
    
    def calculate_sleep_interval(self) -> int:
        """Calculate sleep interval based on system load and performance"""
        # AI REASONING: Dynamic interval calculation
        # PSEUDOCODE:
        # 1. Assess current queue size and processing load
        # 2. Consider agent availability and response times
        # 3. Factor in question generation success rate
        # 4. Adjust interval for optimal performance
        
        base_interval = 300  # 5 minutes
        
        # Adjust based on queue size
        if len(self.question_queue) > 20:
            base_interval = 60  # 1 minute if queue is large
        elif len(self.question_queue) < 5:
            base_interval = 600  # 10 minutes if queue is small
        
        return base_interval
    
    async def handle_error(self, error: Exception, context: str):
        """Handle errors and implement recovery strategies"""
        logger.error(f"Error in {context}: {error}")
        self.error_count += 1
        
        if self.error_count > self.max_retries:
            logger.critical(f"Too many errors, stopping agent")
            raise error
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} cleanup completed")

# Example usage
async def main():
    config = {
        "question_generation_interval": 300,  # 5 minutes
        "max_questions_per_cycle": 10,
        "priority_threshold": 5
    }
    
    agent = DiscoveryAgent(config)
    await agent.initialize()
    
    try:
        # Run the agent
        await agent.run()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 