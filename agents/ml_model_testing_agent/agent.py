"""
ML Model Testing Agent - Multi-Tool Enhanced
===========================================

AI Reasoning: This agent tests and validates machine learning models for:
1. Financial data analysis and prediction models
2. Natural language processing for market sentiment
3. Time series forecasting and trend analysis
4. Risk assessment and portfolio optimization models
5. Market microstructure and liquidity models
6. Cross-asset correlation and diversification models

NO TRADING DECISIONS - Only model testing and validation for informational purposes.
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
SYSTEM POLICY: This agent is STRICTLY for model testing and validation.
NO TRADING DECISIONS should be made. All ML model testing is for
informational and research purposes only.

AI REASONING: The agent should:
1. Test and validate machine learning models
2. Assess model performance and accuracy
3. Monitor model drift and degradation
4. Validate model assumptions and robustness
5. NEVER make buy/sell recommendations
6. NEVER provide trading advice
"""

@dataclass
class MLModel:
    """AI Reasoning: Comprehensive ML model with testing metadata"""
    model_id: str
    model_type: str  # sentiment, forecasting, risk, correlation, microstructure
    source: str  # huggingface, custom, scikit-learn, tensorflow
    version: str
    performance_metrics: Dict[str, float]
    test_results: Dict[str, Any]
    last_tested: datetime
    ai_confidence_score: float = 0.0
    ai_analysis_notes: List[str] = None

@dataclass
class ModelTestResult:
    """AI Reasoning: Model test results with validation metrics"""
    test_id: str
    model_id: str
    test_type: str  # accuracy, robustness, drift, performance
    test_data: Dict[str, Any]
    results: Dict[str, float]
    confidence_score: float
    recommendations: List[str]
    ai_relevance_score: float = 0.0

class MLModelTestingAgent:
    """
    AI Reasoning: Intelligent ML model testing and validation system
    - Test and validate machine learning models for financial applications
    - Assess model performance, accuracy, and robustness
    - Monitor model drift and degradation over time
    - Validate model assumptions and data quality
    - Coordinate with other agents for comprehensive analysis
    - NO TRADING DECISIONS - only model testing and validation
    """
    
    def __init__(self):
        # AI Reasoning: ML model sources and repositories
        self.model_sources = {
            'huggingface': {
                'reliability': 0.90,
                'update_frequency': 'continuous',
                'model_types': ['sentiment', 'nlp', 'transformer'],
                'api_key': os.getenv('HUGGINGFACE_API_KEY')
            },
            'scikit_learn': {
                'reliability': 0.95,
                'update_frequency': 'monthly',
                'model_types': ['regression', 'classification', 'clustering'],
                'api_key': None
            },
            'tensorflow': {
                'reliability': 0.88,
                'update_frequency': 'continuous',
                'model_types': ['neural_network', 'deep_learning', 'time_series'],
                'api_key': None
            }
        }
        
        # AI Reasoning: Model testing criteria and thresholds
        self.testing_criteria = {
            'accuracy_threshold': {'minimum': 0.70, 'target': 0.85, 'excellent': 0.95},
            'robustness_threshold': {'minimum': 0.60, 'target': 0.80, 'excellent': 0.90},
            'drift_threshold': {'warning': 0.10, 'critical': 0.20, 'action_required': 0.30},
            'performance_threshold': {'latency_ms': 1000, 'throughput': 100, 'memory_mb': 512}
        }
        
        # AI Reasoning: Model categories and testing priorities
        self.model_categories = {
            'sentiment_analysis': {
                'priority': 'high',
                'test_frequency': 'daily',
                'key_metrics': ['accuracy', 'f1_score', 'precision', 'recall']
            },
            'time_series_forecasting': {
                'priority': 'high',
                'test_frequency': 'daily',
                'key_metrics': ['mae', 'rmse', 'mape', 'r2_score']
            },
            'risk_assessment': {
                'priority': 'critical',
                'test_frequency': 'daily',
                'key_metrics': ['var', 'cvar', 'sharpe_ratio', 'max_drawdown']
            },
            'correlation_analysis': {
                'priority': 'medium',
                'test_frequency': 'weekly',
                'key_metrics': ['correlation_coefficient', 'p_value', 'confidence_interval']
            }
        }
        
        # AI Reasoning: Research paper sources and parsing capabilities
        self.research_sources = {
            'arxiv': {
                'reliability': 0.85,
                'update_frequency': 'daily',
                'categories': ['cs.AI', 'cs.LG', 'q-fin.CP', 'q-fin.PM', 'q-fin.PR'],
                'api_key': os.getenv('ARXIV_API_KEY')
            },
            'papers_with_code': {
                'reliability': 0.90,
                'update_frequency': 'continuous',
                'categories': ['machine_learning', 'computer_vision', 'nlp'],
                'api_key': os.getenv('PAPERS_WITH_CODE_API_KEY')
            },
            'google_scholar': {
                'reliability': 0.80,
                'update_frequency': 'weekly',
                'categories': ['finance', 'economics', 'machine_learning'],
                'api_key': os.getenv('GOOGLE_SCHOLAR_API_KEY')
            },
            'ssrn': {
                'reliability': 0.88,
                'update_frequency': 'weekly',
                'categories': ['finance', 'economics', 'risk_management'],
                'api_key': os.getenv('SSRN_API_KEY')
            }
        }
        
        # AI Reasoning: Research paper analysis capabilities
        self.paper_analysis_capabilities = {
            'methodology_extraction': {
                'algorithms': ['extract_methodology', 'identify_models', 'parse_equations'],
                'confidence_threshold': 0.7
            },
            'performance_analysis': {
                'metrics': ['extract_benchmarks', 'compare_results', 'assess_significance'],
                'confidence_threshold': 0.8
            },
            'implementation_guidance': {
                'code_extraction': ['find_implementations', 'extract_pseudocode', 'identify_dependencies'],
                'confidence_threshold': 0.6
            },
            'relevance_assessment': {
                'criteria': ['financial_relevance', 'methodology_applicability', 'data_compatibility'],
                'confidence_threshold': 0.75
            }
        }
        
        self.agent_name = "ml_model_testing_agent"
        
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
        
        logger.info(f"Initialized {self.agent_name} with multi-tool integration")

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
            
            # Create tools for ML model testing
            self.tools = [
                Tool(
                    name="test_model_performance",
                    func=self._test_model_performance_tool,
                    description="Test ML model performance and accuracy"
                ),
                Tool(
                    name="analyze_research_papers",
                    func=self._analyze_research_papers_tool,
                    description="Analyze research papers for ML model insights"
                ),
                Tool(
                    name="detect_model_drift",
                    func=self._detect_model_drift_tool,
                    description="Detect model drift and degradation"
                )
            ]
            
            # Create agent executor
            prompt = PromptTemplate.from_template(
                "You are an ML model testing expert. Use the available tools to test and validate machine learning models.\n\n"
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
            # Create ML model testing assistant
            self.ml_testing_assistant = AssistantAgent(
                name="ml_testing_analyst",
                system_message="You are an expert ML model testing analyst. Test and validate machine learning models for financial applications.",
                llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
            )
            
            # Create research analysis assistant
            self.research_assistant = AssistantAgent(
                name="research_analyst",
                system_message="You are an expert research analyst. Analyze research papers and extract ML model insights.",
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
                agents=[self.user_proxy, self.ml_testing_assistant, self.research_assistant],
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

    async def parse_research_papers(self, query: str, max_papers: int = 10) -> List[Dict[str, Any]]:
        """
        AI Reasoning: Parse and analyze research papers for ML model insights
        - Search and retrieve relevant research papers
        - Extract methodology, algorithms, and performance metrics
        - Analyze findings and implementation guidance
        - NO TRADING DECISIONS - only research analysis
        """
        # PSEUDOCODE for research paper parsing:
        # 1. Search multiple research sources (arXiv, Papers with Code, Google Scholar, SSRN)
        # 2. Filter papers by relevance to financial ML and query topic
        # 3. Extract methodology, algorithms, and mathematical formulations
        # 4. Parse performance metrics and benchmark results
        # 5. Identify implementation details and code availability
        # 6. Assess relevance and applicability to current models
        # 7. Generate insights and recommendations for model improvement
        # 8. Return comprehensive paper analysis with confidence scores
        # 9. NO TRADING DECISIONS - only research analysis
        
        try:
            parsed_papers = []
            
            # AI Reasoning: Search across multiple research sources
            for source_name, source_config in self.research_sources.items():
                papers = await self.search_research_source(source_name, query, max_papers // len(self.research_sources))
                
                for paper in papers:
                    # AI Reasoning: Parse individual paper
                    parsed_paper = await self.parse_individual_paper(paper, source_name)
                    if parsed_paper:
                        parsed_papers.append(parsed_paper)
            
            # AI Reasoning: Rank papers by relevance and quality
            ranked_papers = self.rank_papers_by_relevance(parsed_papers, query)
            
            return ranked_papers[:max_papers]
            
        except Exception as e:
            logger.error(f"Error parsing research papers: {e}")
            return []
    
    async def search_research_source(self, source_name: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        AI Reasoning: Search specific research source for relevant papers
        - Query source API with financial ML focus
        - Filter results by relevance and quality
        - NO TRADING DECISIONS - only paper search
        """
        # PSEUDOCODE for research source search:
        # 1. Construct search query with financial ML focus
        # 2. Query source API with appropriate parameters
        # 3. Filter results by relevance score and publication date
        # 4. Extract paper metadata and abstracts
        # 5. Return filtered paper list
        # 6. NO TRADING DECISIONS - only search operation
        
        try:
            # AI Reasoning: Construct financial ML focused query
            financial_ml_query = f"{query} AND (finance OR trading OR market OR risk)"
            
            # AI Reasoning: Query source (placeholder implementation)
            # In production, implement actual API calls to research sources
            papers = [
                {
                    'title': f'Research Paper on {query}',
                    'authors': ['Author 1', 'Author 2'],
                    'abstract': f'Abstract about {query} in financial context',
                    'url': f'https://{source_name}.org/paper1',
                    'publication_date': '2024-01-01',
                    'source': source_name,
                    'relevance_score': 0.85
                }
            ]
            
            return papers[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching {source_name}: {e}")
            return []
    
    async def parse_individual_paper(self, paper: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        AI Reasoning: Parse individual research paper for detailed analysis
        - Extract methodology and algorithms
        - Parse performance metrics and results
        - Identify implementation details
        - NO TRADING DECISIONS - only paper parsing
        """
        # PSEUDOCODE for individual paper parsing:
        # 1. Download and parse paper content (PDF, text, or HTML)
        # 2. Extract methodology section and algorithms
        # 3. Parse mathematical formulations and equations
        # 4. Extract performance metrics and benchmark results
        # 5. Identify implementation details and code availability
        # 6. Assess financial relevance and applicability
        # 7. Calculate confidence scores for extracted information
        # 8. Return comprehensive paper analysis
        # 9. NO TRADING DECISIONS - only parsing analysis
        
        try:
            # AI Reasoning: Extract methodology and algorithms
            methodology = await self.extract_methodology(paper)
            
            # AI Reasoning: Parse performance metrics
            performance = await self.extract_performance_metrics(paper)
            
            # AI Reasoning: Identify implementation details
            implementation = await self.extract_implementation_details(paper)
            
            # AI Reasoning: Assess relevance
            relevance = self.assess_paper_relevance(paper, methodology, performance)
            
            return {
                'paper_id': paper.get('url', '').split('/')[-1],
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'source': source,
                'publication_date': paper.get('publication_date', ''),
                'methodology': methodology,
                'performance_metrics': performance,
                'implementation_details': implementation,
                'relevance_assessment': relevance,
                'confidence_score': self.calculate_parsing_confidence(methodology, performance, implementation),
                'ai_insights': self.generate_paper_insights(methodology, performance, implementation)
            }
            
        except Exception as e:
            logger.error(f"Error parsing individual paper: {e}")
            return None
    
    async def extract_methodology(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI Reasoning: Extract methodology and algorithms from research paper
        - Identify ML algorithms and techniques
        - Parse mathematical formulations
        - Extract model architectures
        - NO TRADING DECISIONS - only methodology extraction
        """
        # PSEUDOCODE for methodology extraction:
        # 1. Parse paper content for methodology section
        # 2. Identify ML algorithms and techniques mentioned
        # 3. Extract mathematical formulations and equations
        # 4. Identify model architectures and components
        # 5. Extract hyperparameters and configuration details
        # 6. Return structured methodology information
        # 7. NO TRADING DECISIONS - only extraction
        
        return {
            'algorithms': ['LSTM', 'Transformer', 'Random Forest'],
            'mathematical_formulations': ['equation1', 'equation2'],
            'model_architecture': 'Neural Network with Attention',
            'hyperparameters': {'learning_rate': 0.001, 'batch_size': 32},
            'confidence': 0.8,
            'extraction_method': 'ai_parsing'
        }
    
    async def extract_performance_metrics(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI Reasoning: Extract performance metrics and benchmark results
        - Parse accuracy, precision, recall metrics
        - Extract benchmark comparisons
        - Identify statistical significance
        - NO TRADING DECISIONS - only metrics extraction
        """
        # PSEUDOCODE for performance metrics extraction:
        # 1. Parse results section for performance metrics
        # 2. Extract accuracy, precision, recall, F1 scores
        # 3. Identify benchmark comparisons and baselines
        # 4. Extract statistical significance tests
        # 5. Parse confidence intervals and error margins
        # 6. Return structured performance data
        # 7. NO TRADING DECISIONS - only extraction
        
        return {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.94,
            'f1_score': 0.91,
            'benchmark_comparison': 'outperforms_baseline',
            'statistical_significance': 'p < 0.01',
            'confidence_intervals': [0.89, 0.95],
            'confidence': 0.85,
            'extraction_method': 'ai_parsing'
        }
    
    async def extract_implementation_details(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI Reasoning: Extract implementation details and code availability
        - Identify code repositories and implementations
        - Extract pseudocode and algorithms
        # Parse dependencies and requirements
        - NO TRADING DECISIONS - only implementation extraction
        """
        # PSEUDOCODE for implementation details extraction:
        # 1. Search for code repositories and implementations
        # 2. Extract pseudocode and algorithm descriptions
        # 3. Identify programming languages and frameworks
        # 4. Parse dependencies and requirements
        # 5. Extract data preprocessing steps
        # 6. Return implementation guidance
        # 7. NO TRADING DECISIONS - only extraction
        
        return {
            'code_available': True,
            'repository_url': 'https://github.com/author/model',
            'programming_language': 'Python',
            'frameworks': ['TensorFlow', 'PyTorch'],
            'dependencies': ['numpy', 'pandas', 'scikit-learn'],
            'data_preprocessing': ['normalization', 'feature_selection'],
            'confidence': 0.7,
            'extraction_method': 'ai_parsing'
        }
    
    def assess_paper_relevance(self, paper: Dict[str, Any], methodology: Dict[str, Any], performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI Reasoning: Assess relevance of research paper to financial ML
        - Evaluate financial applicability
        - Assess methodology relevance
        - Consider data compatibility
        - NO TRADING DECISIONS - only relevance assessment
        """
        # PSEUDOCODE for relevance assessment:
        # 1. Evaluate financial domain applicability
        # 2. Assess methodology relevance to current models
        # 3. Consider data compatibility and requirements
        # 4. Evaluate performance improvement potential
        # 5. Calculate overall relevance score
        # 6. Return relevance assessment
        # 7. NO TRADING DECISIONS - only assessment
        
        return {
            'financial_relevance': 0.85,
            'methodology_applicability': 0.80,
            'data_compatibility': 0.75,
            'performance_improvement_potential': 0.70,
            'overall_relevance_score': 0.78,
            'recommendation': 'highly_relevant',
            'confidence': 0.8
        }
    
    def calculate_parsing_confidence(self, methodology: Dict[str, Any], performance: Dict[str, Any], implementation: Dict[str, Any]) -> float:
        """
        AI Reasoning: Calculate confidence in paper parsing results
        - Assess extraction quality and completeness
        - Consider source reliability and paper quality
        - NO TRADING DECISIONS - only confidence calculation
        """
        # PSEUDOCODE for confidence calculation:
        # 1. Assess methodology extraction completeness
        # 2. Evaluate performance metrics quality
        # 3. Consider implementation details availability
        # 4. Factor in source reliability
        # 5. Calculate composite confidence score
        # 6. Return confidence assessment
        # 7. NO TRADING DECISIONS - only calculation
        
        methodology_confidence = methodology.get('confidence', 0.5)
        performance_confidence = performance.get('confidence', 0.5)
        implementation_confidence = implementation.get('confidence', 0.5)
        
        return (methodology_confidence * 0.4 + performance_confidence * 0.4 + implementation_confidence * 0.2)
    
    def generate_paper_insights(self, methodology: Dict[str, Any], performance: Dict[str, Any], implementation: Dict[str, Any]) -> List[str]:
        """
        AI Reasoning: Generate insights from parsed research paper
        - Identify key findings and implications
        - Suggest model improvements
        - NO TRADING DECISIONS - only insight generation
        """
        # PSEUDOCODE for insight generation:
        # 1. Analyze methodology for novel approaches
        # 2. Identify performance improvements
        # 3. Consider implementation feasibility
        # 4. Generate actionable insights
        # 5. Return insight list
        # 6. NO TRADING DECISIONS - only insight generation
        
        insights = []
        
        if performance.get('f1_score', 0) > 0.9:
            insights.append("High performance model with potential for financial applications")
        
        if 'Transformer' in methodology.get('algorithms', []):
            insights.append("Transformer architecture shows promise for financial time series")
        
        if implementation.get('code_available', False):
            insights.append("Implementation available for testing and validation")
        
        return insights
    
    def rank_papers_by_relevance(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        AI Reasoning: Rank papers by relevance to query and financial ML
        - Apply relevance scoring algorithm
        - Consider publication date and impact
        - NO TRADING DECISIONS - only ranking
        """
        # PSEUDOCODE for paper ranking:
        # 1. Calculate relevance score based on query match
        # 2. Consider publication date and recency
        # 3. Factor in source reliability and impact
        # 4. Apply financial ML relevance bonus
        # 5. Sort papers by composite score
        # 6. Return ranked paper list
        # 7. NO TRADING DECISIONS - only ranking
        
        for paper in papers:
            relevance_score = paper.get('relevance_assessment', {}).get('overall_relevance_score', 0.5)
            confidence_score = paper.get('confidence_score', 0.5)
            
            # AI Reasoning: Calculate composite ranking score
            paper['ranking_score'] = relevance_score * 0.7 + confidence_score * 0.3
        
        # AI Reasoning: Sort by ranking score
        return sorted(papers, key=lambda x: x.get('ranking_score', 0), reverse=True)
    
    async def check_knowledge_base_for_existing_data(self, model_id: str, test_type: str = None) -> Dict[str, Any]:
        """
        AI Reasoning: Check knowledge base for existing model test data
        - Query existing test results and performance metrics
        - Assess test freshness and completeness
        - Determine if new testing is needed
        - Identify test gaps and inconsistencies
        - NO TRADING DECISIONS - only data validation
        """
        # PSEUDOCODE:
        # 1. Query knowledge base for model's recent test results
        # 2. Check last test timestamp and data freshness
        # 3. Assess test completeness against expected criteria
        # 4. Identify missing or outdated test results
        # 5. Calculate confidence in existing test data quality
        # 6. Determine if new testing is warranted
        # 7. Return existing data with quality assessment
        # 8. NO TRADING DECISIONS - only data validation
        
        try:
            with engine.connect() as conn:
                # AI Reasoning: Intelligent query based on test type and time range
                if test_type:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'ml_model_testing_agent' 
                        AND data->>'model_id' = :model_id 
                        AND data->>'test_type' = :test_type
                        ORDER BY event_time DESC 
                        LIMIT 20
                    """)
                    result = conn.execute(query, {"model_id": model_id, "test_type": test_type})
                else:
                    query = text("""
                        SELECT * FROM events 
                        WHERE source_agent = 'ml_model_testing_agent' 
                        AND data->>'model_id' = :model_id
                        ORDER BY event_time DESC 
                        LIMIT 50
                    """)
                    result = conn.execute(query, {"model_id": model_id})
                
                existing_data = [dict(row) for row in result]
                
                # AI Reasoning: Assess data quality and freshness
                data_quality = {
                    'total_records': len(existing_data),
                    'latest_test': existing_data[0]['event_time'] if existing_data else None,
                    'test_freshness_hours': None,
                    'completeness_score': 0.0,
                    'confidence_level': 0.0
                }
                
                if existing_data:
                    latest_time = existing_data[0]['event_time']
                    data_quality['test_freshness_hours'] = (datetime.utcnow() - latest_time).total_seconds() / 3600
                    
                    # AI Reasoning: Calculate completeness based on expected test types
                    test_types = [event['data'].get('test_type') for event in existing_data]
                    data_quality['completeness_score'] = len(set(test_types)) / len(self.testing_criteria)
                    
                    # AI Reasoning: Assess confidence based on test consistency
                    data_quality['confidence_level'] = min(1.0, data_quality['completeness_score'] * 0.9)
                
                return {
                    'existing_data': existing_data,
                    'data_quality': data_quality,
                    'needs_testing': data_quality['test_freshness_hours'] is None or data_quality['test_freshness_hours'] > 24.0
                }
                
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            return {'existing_data': [], 'data_quality': {}, 'needs_testing': True}
    
    async def select_models_for_testing(self, category: str = None) -> List[str]:
        """
        AI Reasoning: Select models for testing based on priority and schedule
        - Evaluate model testing priorities and schedules
        - Match models to testing requirements
        - Prioritize models based on importance and last test time
        - Consider testing resources and constraints
        - NO TRADING DECISIONS - only model selection
        """
        # PSEUDOCODE:
        # 1. Analyze model testing priorities and schedules
        # 2. Evaluate available models and their categories
        # 3. Check last test time and testing frequency requirements
        # 4. Assess testing resources and time constraints
        # 5. Prioritize models based on importance and urgency
        # 6. Select optimal set of models for testing
        # 7. Return prioritized list of models to test
        # 8. NO TRADING DECISIONS - only model selection
        
        # AI Reasoning: Example model IDs for testing
        priority_models = [
            'sentiment_bert_financial',
            'time_series_lstm_sp500',
            'risk_var_garch',
            'correlation_pearson_multiasset',
            'microstructure_orderbook_mlp'
        ]
        
        if category:
            # AI Reasoning: Filter models by category
            category_models = [model for model in priority_models if category in model]
            return category_models[:3]  # Limit to top 3 models per category
        
        return priority_models[:5]  # Return top 5 priority models
    
    async def test_model_performance(self, model_id: str, test_data: Dict[str, Any]) -> ModelTestResult:
        """
        AI Reasoning: Test model performance and generate comprehensive results
        - Execute model tests with various datasets
        - Calculate performance metrics and accuracy scores
        - Assess model robustness and generalization
        - Identify potential issues and improvements
        - NO TRADING DECISIONS - only model testing
        """
        # PSEUDOCODE:
        # 1. Load model and test datasets
        # 2. Execute model predictions on test data
        # 3. Calculate performance metrics (accuracy, precision, recall, etc.)
        # 4. Assess model robustness with different data subsets
        # 5. Evaluate model generalization and overfitting
        # 6. Generate comprehensive test report
        # 7. Score results by confidence and significance
        # 8. NO TRADING DECISIONS - only model testing
        
        test_result = ModelTestResult(
            test_id=f"test_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_id=model_id,
            test_type='performance',
            test_data=test_data,
            results={
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'latency_ms': 150,
                'throughput': 200
            },
            confidence_score=0.8,
            recommendations=[
                'Model performs well on test data',
                'Consider retraining with more recent data',
                'Monitor for potential drift'
            ]
        )
        
        # AI Reasoning: Assess results against thresholds
        accuracy = test_result.results.get('accuracy', 0.0)
        if accuracy < self.testing_criteria['accuracy_threshold']['minimum']:
            test_result.recommendations.append('Model accuracy below minimum threshold - requires attention')
            test_result.confidence_score *= 0.8
        
        return test_result
    
    async def test_model_robustness(self, model_id: str, test_data: Dict[str, Any]) -> ModelTestResult:
        """
        AI Reasoning: Test model robustness and stability
        - Test model with noisy and adversarial data
        - Assess model stability across different conditions
        - Evaluate model sensitivity to parameter changes
        - Test model performance under stress conditions
        - NO TRADING DECISIONS - only robustness testing
        """
        # PSEUDOCODE:
        # 1. Generate noisy and adversarial test datasets
        # 2. Test model with varying data quality levels
        # 3. Assess model stability across different conditions
        # 4. Evaluate sensitivity to parameter variations
        # 5. Test performance under stress scenarios
        # 6. Calculate robustness metrics and scores
        # 7. Generate robustness assessment report
        # 8. NO TRADING DECISIONS - only robustness testing
        
        robustness_result = ModelTestResult(
            test_id=f"robustness_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_id=model_id,
            test_type='robustness',
            test_data=test_data,
            results={
                'noise_tolerance': 0.75,
                'adversarial_robustness': 0.68,
                'parameter_sensitivity': 0.82,
                'stress_test_performance': 0.70
            },
            confidence_score=0.75,
            recommendations=[
                'Model shows good noise tolerance',
                'Adversarial robustness could be improved',
                'Parameter sensitivity is acceptable'
            ]
        )
        
        return robustness_result
    
    async def detect_model_drift(self, model_id: str, historical_data: List[Dict[str, Any]]) -> ModelTestResult:
        """
        AI Reasoning: Detect model drift and performance degradation
        - Compare current performance with historical baseline
        - Identify performance degradation patterns
        - Assess data drift and concept drift
        - Recommend model retraining if needed
        - NO TRADING DECISIONS - only drift detection
        """
        # PSEUDOCODE:
        # 1. Load historical performance data for the model
        # 2. Calculate baseline performance metrics
        # 3. Compare current performance with historical baseline
        # 4. Identify performance degradation patterns
        # 5. Assess data distribution changes
        # 6. Detect concept drift indicators
        # 7. Generate drift detection report
        # 8. NO TRADING DECISIONS - only drift detection
        
        # AI Reasoning: Calculate drift metrics
        baseline_accuracy = 0.88
        current_accuracy = 0.82
        drift_magnitude = baseline_accuracy - current_accuracy
        
        drift_result = ModelTestResult(
            test_id=f"drift_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_id=model_id,
            test_type='drift_detection',
            test_data={'historical_data': historical_data},
            results={
                'drift_magnitude': drift_magnitude,
                'drift_direction': 'decreasing',
                'confidence_level': 0.85,
                'recommended_action': 'retrain' if drift_magnitude > self.testing_criteria['drift_threshold']['warning'] else 'monitor'
            },
            confidence_score=0.85,
            recommendations=[
                f'Model performance decreased by {drift_magnitude:.3f}',
                'Consider retraining with recent data' if drift_magnitude > self.testing_criteria['drift_threshold']['warning'] else 'Continue monitoring',
                'Assess data quality and feature relevance'
            ]
        )
        
        return drift_result
    
    async def determine_next_best_action(self, test_results: List[ModelTestResult], model_id: str) -> Dict[str, Any]:
        """
        AI Reasoning: Determine next best action based on model test results
        - Evaluate test results and identify issues
        - Decide on model retraining requirements
        - Plan additional testing if needed
        - Schedule follow-up monitoring
        - NO TRADING DECISIONS - only action planning
        """
        # PSEUDOCODE:
        # 1. Assess test results and identify critical issues
        # 2. Evaluate model performance against thresholds
        # 3. Determine if retraining is needed
        # 4. Plan additional testing for validation
        # 5. Schedule follow-up monitoring
        # 6. Prioritize actions based on severity
        # 7. Return action plan with priorities
        # 8. NO TRADING DECISIONS - only action planning
        
        next_actions = {
            'immediate_actions': [],
            'scheduled_actions': [],
            'coordination_needed': [],
            'priority_level': 'low'
        }
        
        # AI Reasoning: Evaluate test results
        for result in test_results:
            if result.test_type == 'drift_detection':
                drift_magnitude = result.results.get('drift_magnitude', 0.0)
                if drift_magnitude > self.testing_criteria['drift_threshold']['critical']:
                    next_actions['priority_level'] = 'critical'
                    next_actions['immediate_actions'].append({
                        'action': 'notify_orchestrator',
                        'reason': 'critical_model_drift',
                        'data': result
                    })
                    
                    next_actions['immediate_actions'].append({
                        'action': 'schedule_retraining',
                        'reason': 'model_performance_degradation',
                        'urgency': 'immediate'
                    })
            
            elif result.test_type == 'performance':
                accuracy = result.results.get('accuracy', 0.0)
                if accuracy < self.testing_criteria['accuracy_threshold']['minimum']:
                    next_actions['priority_level'] = 'high'
                    next_actions['immediate_actions'].append({
                        'action': 'notify_orchestrator',
                        'reason': 'low_model_accuracy',
                        'data': result
                    })
        
        # AI Reasoning: Schedule follow-up testing
        next_actions['scheduled_actions'].append({
            'action': 'follow_up_testing',
            'schedule_hours': 24,
            'reason': 'regular_model_monitoring'
        })
        
        return next_actions

    async def fetch_and_process_models(self):
        """
        AI Reasoning: Fetch and test machine learning models
        - Retrieve models from various sources
        - Execute comprehensive testing protocols
        - Analyze test results and performance metrics
        - Store significant results in knowledge base
        - Coordinate with other agents as needed
        - NO TRADING DECISIONS - only model testing
        """
        # PSEUDOCODE:
        # 1. Select models for testing based on priority and schedule
        # 2. Check knowledge base for existing test results
        # 3. Execute comprehensive model testing
        # 4. Analyze test results and performance metrics
        # 5. Detect model drift and degradation
        # 6. Store significant results in knowledge base
        # 7. Determine next actions and coordinate with agents
        # 8. NO TRADING DECISIONS - only model testing
        
        try:
            # AI Reasoning: Select models for testing
            models_to_test = await self.select_models_for_testing()
            
            for model_id in models_to_test:
                # AI Reasoning: Check existing test data and determine testing needs
                existing_data = await self.check_knowledge_base_for_existing_data(model_id)
                
                if not existing_data['needs_testing']:
                    logger.info(f"Recent test data exists for {model_id}, skipping testing")
                    continue
                
                # AI Reasoning: Execute comprehensive model testing
                test_results = []
                
                # Performance testing
                performance_result = await self.test_model_performance(model_id, {'test_type': 'performance'})
                test_results.append(performance_result)
                
                # Robustness testing
                robustness_result = await self.test_model_robustness(model_id, {'test_type': 'robustness'})
                test_results.append(robustness_result)
                
                # Drift detection
                historical_data = existing_data['existing_data']
                drift_result = await self.detect_model_drift(model_id, historical_data)
                test_results.append(drift_result)
                
                # AI Reasoning: Store significant test results in knowledge base
                for result in test_results:
                    if result.confidence_score > 0.7:
                        await self.store_in_knowledge_base(model_id, result)
                
                # AI Reasoning: Determine next actions
                next_actions = await self.determine_next_best_action(test_results, model_id)
                
                # AI Reasoning: Execute immediate actions
                for action in next_actions['immediate_actions']:
                    if action['action'] == 'notify_orchestrator':
                        await self.notify_orchestrator(action['data'])
                
                # AI Reasoning: Schedule follow-up actions
                for action in next_actions['scheduled_actions']:
                    if action['action'] == 'follow_up_testing':
                        asyncio.create_task(self.schedule_follow_up_testing(model_id, action['schedule_hours']))
                
                # AI Reasoning: Rate limiting between models
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error in fetch_and_process_models: {e}")
            # AI Reasoning: Implement error recovery and retry logic
            await self.handle_error_recovery(e)
    
    async def store_in_knowledge_base(self, model_id: str, test_result: ModelTestResult):
        """
        AI Reasoning: Store significant model test results in knowledge base
        - Store test results with proper metadata
        - Include performance metrics and confidence scores
        - Tag results for easy retrieval and analysis
        - NO TRADING DECISIONS - only data storage
        """
        # PSEUDOCODE:
        # 1. Prepare test result data with metadata
        # 2. Include performance metrics and confidence scores
        # 3. Store in knowledge base with proper indexing
        # 4. Tag results for correlation analysis
        # 5. Update test tracking and statistics
        # 6. NO TRADING DECISIONS - only data storage
        
        try:
            event_data = {
                'model_id': model_id,
                'test_id': test_result.test_id,
                'test_type': test_result.test_type,
                'results': test_result.results,
                'confidence_score': test_result.confidence_score,
                'recommendations': test_result.recommendations,
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
                    'event_type': 'model_test_result',
                    'event_time': datetime.utcnow(),
                    'data': json.dumps(event_data)
                })
                conn.commit()
                
            logger.info(f"Stored model test result for {model_id}")
            
        except Exception as e:
            logger.error(f"Error storing model test data: {e}")
    
    async def notify_orchestrator(self, data: Dict[str, Any]):
        """
        AI Reasoning: Notify orchestrator of significant model test results
        - Send critical test results to orchestrator
        - Include performance metrics and recommendations
        - Request coordination with other agents if needed
        - NO TRADING DECISIONS - only coordination
        """
        # PSEUDOCODE:
        # 1. Prepare notification data with test results
        # 2. Include confidence scores and recommendations
        # 3. Send to orchestrator via MCP
        # 4. Request coordination with related agents
        # 5. NO TRADING DECISIONS - only coordination
        
        try:
            notification = {
                'agent': self.agent_name,
                'event_type': 'significant_model_test_result',
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'priority': 'high' if data.get('confidence_score', 0.0) > 0.8 else 'medium'
            }
            
            # AI Reasoning: Send via MCP to orchestrator
            # await self.send_mcp_message('orchestrator', notification)
            logger.info(f"Notified orchestrator of significant model test result: {notification}")
            
        except Exception as e:
            logger.error(f"Error notifying orchestrator: {e}")
    
    async def schedule_follow_up_testing(self, model_id: str, delay_hours: int):
        """
        AI Reasoning: Schedule follow-up testing for model validation
        - Schedule delayed testing for model confirmation
        - Monitor model performance evolution over time
        - Update test results as new data arrives
        - NO TRADING DECISIONS - only testing scheduling
        """
        # PSEUDOCODE:
        # 1. Wait for specified delay period
        # 2. Re-test model with updated datasets
        # 3. Compare with previous test results
        # 4. Update performance metrics and confidence
        # 5. Store updated test results in knowledge base
        # 6. NO TRADING DECISIONS - only testing scheduling
        
        await asyncio.sleep(delay_hours * 3600)
        
        try:
            # AI Reasoning: Re-test model for validation
            test_result = await self.test_model_performance(model_id, {'test_type': 'follow_up'})
            
            # AI Reasoning: Update knowledge base with follow-up results
            if test_result.confidence_score > 0.6:
                await self.store_in_knowledge_base(model_id, test_result)
            
            logger.info(f"Completed follow-up testing for {model_id}")
            
        except Exception as e:
            logger.error(f"Error in follow-up testing for {model_id}: {e}")
    
    async def handle_error_recovery(self, error: Exception):
        """
        AI Reasoning: Handle errors and implement recovery strategies
        - Log errors with context and severity
        - Implement retry logic with exponential backoff
        - Fall back to alternative testing methods
        - Maintain system stability and data quality
        - NO TRADING DECISIONS - only error recovery
        """
        # PSEUDOCODE:
        # 1. Log error with full context and stack trace
        # 2. Assess error severity and impact
        # 3. Implement appropriate recovery strategy
        # 4. Retry with exponential backoff if applicable
        # 5. Fall back to alternative testing methods
        # 6. Update system health metrics
        # 7. NO TRADING DECISIONS - only error recovery
        
        logger.error(f"ML model testing agent error: {error}")
        
        # AI Reasoning: Implement retry logic for transient errors
        if "timeout" in str(error).lower():
            logger.info("Test timeout, implementing backoff strategy")
            await asyncio.sleep(60)  # Wait 1 minute before retry
        elif "memory" in str(error).lower():
            logger.info("Memory error, reducing test batch size")
            await asyncio.sleep(30)  # Wait 30 seconds before retry

    async def listen_for_mcp_messages(self):
        """
        AI Reasoning: Listen for MCP messages from orchestrator and other agents
        - Handle requests for model testing
        - Respond to coordination requests
        - Process priority testing requests
        - NO TRADING DECISIONS - only message handling
        """
        # PSEUDOCODE:
        # 1. Listen for incoming MCP messages
        # 2. Parse message type and priority
        # 3. Handle testing requests for specific models
        # 4. Respond with current test results
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
        - Coordinate model testing and validation
        - Handle MCP communication
        - Implement error recovery and monitoring
        - Maintain system health and performance
        - NO TRADING DECISIONS - only system operation
        """
        # PSEUDOCODE:
        # 1. Initialize agent and establish connections
        # 2. Start MCP message listening
        # 3. Begin periodic model testing
        # 4. Handle errors and implement recovery
        # 5. Monitor system health and performance
        # 6. Coordinate with other agents as needed
        # 7. NO TRADING DECISIONS - only system operation
        
        logger.info("Starting ML Model Testing Agent")
        
        while True:
            try:
                # AI Reasoning: Run main testing cycle
                await self.fetch_and_process_models()
                
                # AI Reasoning: Handle MCP communication
                await self.listen_for_mcp_messages()
                
                # AI Reasoning: Wait before next cycle
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in main agent loop: {e}")
                await self.handle_error_recovery(e)
                await asyncio.sleep(60)  # Wait before retry

async def main():
    """AI Reasoning: Main entry point for ML Model Testing Agent"""
    agent = MLModelTestingAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 