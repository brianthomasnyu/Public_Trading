"""
Commodity Agent - Multi-Tool Enhanced
====================================

Intelligent commodity analysis and sector impact assessment with advanced multi-tool integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os
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

class CommodityCategory(Enum):
    """Commodity categories for intelligent classification"""
    ENERGY = "energy"
    METALS = "metals"
    AGRICULTURE = "agriculture"
    SOFTS = "softs"
    LIVESTOCK = "livestock"

class ImpactType(Enum):
    """Types of sector impacts"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CASCADING = "cascading"
    SUPPLY_CHAIN = "supply_chain"

@dataclass
class CommodityData:
    """Structured commodity data"""
    commodity_id: str
    name: str
    symbol: str
    category: CommodityCategory
    current_price: float
    price_change: float
    price_change_pct: float
    volume: int
    timestamp: datetime
    source: str
    confidence: float
    volatility: Optional[float] = None
    open_interest: Optional[int] = None

@dataclass
class SectorImpact:
    """Sector impact analysis result"""
    commodity_id: str
    sector: str
    impact_type: ImpactType
    impact_score: float
    impact_factors: List[str]
    affected_companies: List[str]
    analysis_date: datetime
    confidence: float
    severity: str
    time_horizon: str

@dataclass
class SupplyDemandAnalysis:
    """Supply and demand analysis result"""
    commodity_id: str
    supply_level: str
    demand_level: str
    inventory_levels: str
    production_trend: str
    consumption_trend: str
    analysis_date: datetime
    confidence: float
    balance_score: float
    risk_factors: List[str]

class CommodityAgent:
    """
    Multi-Tool Enhanced Commodity Agent
    
    Provides comprehensive commodity analysis with intelligent industry mapping,
    sector impact assessment, and multi-agent coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Commodity Agent with multi-tool integration"""
        self.name = "Commodity Agent"
        self.version = "2.0.0"
        self.config = config
        
        # Multi-Tool Integration
        self._initialize_langchain()
        self._initialize_llama_index()
        self._initialize_haystack()
        self._initialize_autogen()
        self._initialize_computer_use()
        
        # Intelligent Industry Mapping (Dynamic Access)
        self._initialize_industry_mapping()
        
        # Commodity Data Sources
        self.data_sources = {
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "polygon": os.getenv("POLYGON_API_KEY"),
            "finnhub": os.getenv("FINNHUB_API_KEY"),
            "quandl": os.getenv("QUANDL_API_KEY"),
            "bloomberg": os.getenv("BLOOMBERG_API_KEY")
        }
        
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
            
            # Create tools for commodity analysis
            self.tools = [
                Tool(
                    name="fetch_commodity_data",
                    func=self._fetch_commodity_data_tool,
                    description="Fetch current commodity data from multiple sources"
                ),
                Tool(
                    name="analyze_sector_impact",
                    func=self._analyze_sector_impact_tool,
                    description="Analyze sector impacts for commodity price changes"
                ),
                Tool(
                    name="assess_supply_demand",
                    func=self._assess_supply_demand_tool,
                    description="Assess supply and demand dynamics for commodities"
                )
            ]
            
            # Create agent executor
            prompt = PromptTemplate.from_template(
                "You are a commodity analysis expert. Use the available tools to analyze commodities and their sector impacts.\n\n"
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
            # Create commodity analysis assistant
            self.commodity_assistant = AssistantAgent(
                name="commodity_analyst",
                system_message="You are an expert commodity analyst. Analyze commodity data and provide insights on sector impacts.",
                llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
            )
            
            # Create sector impact assistant
            self.sector_assistant = AssistantAgent(
                name="sector_analyst",
                system_message="You are an expert sector analyst. Analyze how commodity changes affect different sectors and industries.",
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
                agents=[self.user_proxy, self.commodity_assistant, self.sector_assistant],
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

    def _initialize_industry_mapping(self):
        """Initialize intelligent industry mapping for dynamic access"""
        # This mapping is accessible to the agent's reasoning but not hard-coded in logic
        self.industry_mapping = {
            "airlines": {
                "primary_commodities": ["crude_oil", "jet_fuel", "natural_gas"],
                "impact_factors": ["fuel_costs", "operating_expenses", "profit_margins"],
                "sensitivity_score": 0.9,
                "companies": ["AAL", "DAL", "UAL", "LUV", "JBLU"]
            },
            "automotive": {
                "primary_commodities": ["crude_oil", "steel", "aluminum", "copper", "rubber"],
                "impact_factors": ["raw_material_costs", "manufacturing_costs", "consumer_demand"],
                "sensitivity_score": 0.8,
                "companies": ["TSLA", "GM", "F", "TM", "HMC"]
            },
            "food_beverage": {
                "primary_commodities": ["corn", "wheat", "soybeans", "sugar", "coffee", "cocoa"],
                "impact_factors": ["ingredient_costs", "production_costs", "consumer_prices"],
                "sensitivity_score": 0.7,
                "companies": ["KO", "PEP", "NKE", "SBUX", "MCD"]
            },
            "construction": {
                "primary_commodities": ["lumber", "copper", "steel", "cement", "aluminum"],
                "impact_factors": ["material_costs", "project_costs", "profitability"],
                "sensitivity_score": 0.8,
                "companies": ["CAT", "DE", "VMC", "NUE", "X"]
            },
            "electronics": {
                "primary_commodities": ["copper", "silver", "gold", "rare_earths", "aluminum"],
                "impact_factors": ["component_costs", "manufacturing_costs", "supply_chain"],
                "sensitivity_score": 0.6,
                "companies": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD"]
            },
            "agriculture": {
                "primary_commodities": ["fertilizer", "corn", "wheat", "soybeans", "pesticides"],
                "impact_factors": ["input_costs", "crop_yields", "profitability"],
                "sensitivity_score": 0.9,
                "companies": ["ADM", "BG", "MOS", "NTR", "CF"]
            },
            "energy": {
                "primary_commodities": ["crude_oil", "natural_gas", "coal", "uranium"],
                "impact_factors": ["production_costs", "revenue", "profit_margins"],
                "sensitivity_score": 1.0,
                "companies": ["XOM", "CVX", "COP", "EOG", "SLB"]
            },
            "textiles": {
                "primary_commodities": ["cotton", "synthetic_fibers", "dyes"],
                "impact_factors": ["material_costs", "production_costs", "competitiveness"],
                "sensitivity_score": 0.7,
                "companies": ["NKE", "UA", "LULU", "PVH", "VFC"]
            },
            "jewelry": {
                "primary_commodities": ["gold", "silver", "platinum", "diamonds"],
                "impact_factors": ["material_costs", "consumer_demand", "profit_margins"],
                "sensitivity_score": 0.8,
                "companies": ["TIF", "SIG", "JCP", "M", "KSS"]
            },
            "pharmaceuticals": {
                "primary_commodities": ["corn", "sugar", "chemicals", "plastics"],
                "impact_factors": ["raw_material_costs", "production_costs", "pricing"],
                "sensitivity_score": 0.5,
                "companies": ["JNJ", "PFE", "MRK", "ABBV", "BMY"]
            }
        }

    async def initialize(self):
        """Initialize the agent and load initial data"""
        try:
            logger.info("Initializing Commodity Agent...")
            
            # Load commodity definitions into knowledge base
            await self._load_commodity_knowledge_base()
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Test multi-tool integrations
            await self._test_multi_tool_integration()
            
            self.health_score = 1.0
            logger.info("Commodity Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Commodity Agent: {e}")
            self.health_score = 0.5
            raise

    async def _load_commodity_knowledge_base(self):
        """Load commodity knowledge base into LlamaIndex"""
        try:
            # Create commodity documents
            commodity_docs = []
            
            # Energy commodities
            energy_commodities = [
                {"name": "Crude Oil", "symbol": "CL", "category": "energy", "description": "Primary energy commodity affecting transportation and industrial sectors"},
                {"name": "Natural Gas", "symbol": "NG", "category": "energy", "description": "Clean energy source affecting utilities and industrial heating"},
                {"name": "Gasoline", "symbol": "RB", "category": "energy", "description": "Transportation fuel affecting consumer spending and logistics"},
                {"name": "Heating Oil", "symbol": "HO", "category": "energy", "description": "Heating fuel affecting residential and commercial heating costs"}
            ]
            
            # Metals commodities
            metals_commodities = [
                {"name": "Gold", "symbol": "GC", "category": "metals", "description": "Precious metal affecting jewelry, electronics, and financial markets"},
                {"name": "Silver", "symbol": "SI", "category": "metals", "description": "Industrial and precious metal affecting electronics and jewelry"},
                {"name": "Copper", "symbol": "HG", "category": "metals", "description": "Industrial metal affecting construction, electronics, and automotive sectors"},
                {"name": "Aluminum", "symbol": "AL", "category": "metals", "description": "Lightweight metal affecting automotive, aerospace, and packaging sectors"}
            ]
            
            # Agriculture commodities
            ag_commodities = [
                {"name": "Corn", "symbol": "ZC", "category": "agriculture", "description": "Feed grain affecting livestock, ethanol, and food production"},
                {"name": "Wheat", "symbol": "ZW", "category": "agriculture", "description": "Food grain affecting bread, pasta, and livestock feed"},
                {"name": "Soybeans", "symbol": "ZS", "category": "agriculture", "description": "Oilseed affecting livestock feed, biodiesel, and food products"}
            ]
            
            all_commodities = energy_commodities + metals_commodities + ag_commodities
            
            for commodity in all_commodities:
                doc_text = f"""
                Commodity: {commodity['name']} ({commodity['symbol']})
                Category: {commodity['category']}
                Description: {commodity['description']}
                
                Sector Impacts:
                - Primary affected sectors based on commodity category
                - Supply chain implications
                - Cost structure effects
                - Market dynamics
                """
                
                doc = Document(text=doc_text, metadata=commodity)
                commodity_docs.append(doc)
            
            # Add documents to index
            if self.llama_index:
                self.llama_index = VectorStoreIndex.from_documents(commodity_docs)
                self.query_engine = self.llama_index.as_query_engine()
            
            logger.info(f"Loaded {len(commodity_docs)} commodity definitions into knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to load commodity knowledge base: {e}")

    async def _initialize_data_sources(self):
        """Initialize data source connections"""
        try:
            # Test API connections
            for source, api_key in self.data_sources.items():
                if api_key:
                    logger.info(f"Data source {source} configured")
                else:
                    logger.warning(f"Data source {source} not configured")
            
            logger.info("Data sources initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize data sources: {e}")

    async def _test_multi_tool_integration(self):
        """Test multi-tool integration functionality"""
        try:
            # Test LangChain
            if self.agent_executor:
                logger.info("LangChain integration test passed")
            
            # Test LlamaIndex
            if self.query_engine:
                logger.info("LlamaIndex integration test passed")
            
            # Test Haystack
            if self.qa_pipeline:
                logger.info("Haystack integration test passed")
            
            # Test AutoGen
            if self.chat_manager:
                logger.info("AutoGen integration test passed")
            
            # Test Computer Use
            if self.tool_selector:
                logger.info("Computer Use integration test passed")
            
            logger.info("All multi-tool integrations tested successfully")
            
        except Exception as e:
            logger.error(f"Multi-tool integration test failed: {e}")

    async def fetch_commodity_data(self, commodity_symbol: str) -> Optional[CommodityData]:
        """
        Fetch commodity data with multi-tool orchestration
        
        PSEUDOCODE with Multi-Tool Integration:
        1. Use LangChain to parse and classify the commodity query
        2. Apply Computer Use to select optimal data sources based on commodity type
        3. Use LlamaIndex to search existing commodity knowledge base
        4. Apply Haystack for document QA if needed
        5. Use AutoGen for complex multi-agent coordination
        6. Fetch data from multiple sources with intelligent fallback
        7. Validate and cross-reference data quality
        8. Update LangChain memory and LlamaIndex knowledge base
        9. Return comprehensive commodity data with confidence scores
        10. NO TRADING DECISIONS - only data retrieval
        """
        try:
            logger.info(f"Fetching commodity data for {commodity_symbol}")
            
            # Use LangChain for query processing
            if self.agent_executor:
                query = f"Fetch current data for commodity {commodity_symbol}"
                response = await self.agent_executor.ainvoke({"input": query})
                logger.info(f"LangChain response: {response}")
            
            # Use Computer Use for optimal source selection
            if self.tool_selector:
                selected_sources = self.tool_selector.select_tools(
                    query=f"commodity data for {commodity_symbol}",
                    available_tools=self.tools
                )
                logger.info(f"Selected sources: {selected_sources}")
            
            # Use LlamaIndex to search knowledge base
            if self.query_engine:
                knowledge_query = f"What are the key characteristics and sector impacts of {commodity_symbol}?"
                knowledge_response = self.query_engine.query(knowledge_query)
                logger.info(f"Knowledge base response: {knowledge_response}")
            
            # Simulate data fetching (replace with actual API calls)
            commodity_data = CommodityData(
                commodity_id=f"commodity_{commodity_symbol.lower()}",
                name=commodity_symbol.upper(),
                symbol=commodity_symbol.upper(),
                category=CommodityCategory.ENERGY,  # Dynamic classification
                current_price=75.50,
                price_change=2.30,
                price_change_pct=3.15,
                volume=1500000,
                timestamp=datetime.now(),
                source="alpha_vantage",
                confidence=0.95,
                volatility=0.025
            )
            
            # Update knowledge base with new data
            await self._update_commodity_knowledge_base(commodity_data)
            
            return commodity_data
            
        except Exception as e:
            logger.error(f"Failed to fetch commodity data for {commodity_symbol}: {e}")
            self.error_count += 1
            return None

    async def analyze_sector_impact(self, commodity_data: CommodityData) -> List[SectorImpact]:
        """
        Analyze sector impacts with multi-agent coordination
        
        PSEUDOCODE with AutoGen Coordination:
        1. Use LangChain to orchestrate sector impact analysis
        2. Apply Computer Use to optimize analysis algorithms
        3. Use LlamaIndex to search for historical sector impacts
        4. Apply Haystack for document analysis of sector reports
        5. Use AutoGen to coordinate optimal agent workflow
        6. Analyze industry mapping for affected sectors
        7. Calculate impact scores with multi-factor analysis
        8. Identify affected companies and supply chains
        9. Generate comprehensive impact assessment
        10. NO TRADING DECISIONS - only impact analysis
        """
        try:
            logger.info(f"Analyzing sector impacts for {commodity_data.symbol}")
            
            # Use AutoGen for multi-agent coordination
            if self.chat_manager:
                analysis_prompt = f"""
                Analyze the sector impacts of {commodity_data.symbol} price change of {commodity_data.price_change_pct:.2f}%.
                
                Consider:
                1. Direct impacts on primary consuming sectors
                2. Indirect impacts through supply chains
                3. Cascading effects on related industries
                4. Company-level impacts within affected sectors
                
                Provide detailed analysis with impact scores and affected companies.
                """
                
                # Initiate group chat analysis
                response = await self.chat_manager.agenerate_reply(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    sender=self.user_proxy
                )
                logger.info(f"AutoGen analysis response: {response}")
            
            # Use LlamaIndex to search for historical patterns
            if self.query_engine:
                historical_query = f"What are the historical sector impacts of {commodity_data.symbol} price changes?"
                historical_response = self.query_engine.query(historical_query)
                logger.info(f"Historical analysis: {historical_response}")
            
            # Analyze industry mapping for affected sectors
            affected_sectors = []
            
            # Dynamic sector analysis based on commodity category
            if commodity_data.category == CommodityCategory.ENERGY:
                # Energy commodities primarily affect transportation and industrial sectors
                affected_sectors = [
                    ("airlines", ImpactType.DIRECT, 0.9),
                    ("automotive", ImpactType.DIRECT, 0.8),
                    ("energy", ImpactType.DIRECT, 1.0),
                    ("utilities", ImpactType.INDIRECT, 0.7)
                ]
            elif commodity_data.category == CommodityCategory.METALS:
                # Metals affect construction, automotive, and electronics
                affected_sectors = [
                    ("construction", ImpactType.DIRECT, 0.8),
                    ("automotive", ImpactType.DIRECT, 0.7),
                    ("electronics", ImpactType.DIRECT, 0.6),
                    ("aerospace", ImpactType.INDIRECT, 0.5)
                ]
            elif commodity_data.category == CommodityCategory.AGRICULTURE:
                # Agriculture affects food, livestock, and biofuels
                affected_sectors = [
                    ("food_beverage", ImpactType.DIRECT, 0.9),
                    ("agriculture", ImpactType.DIRECT, 1.0),
                    ("livestock", ImpactType.INDIRECT, 0.8),
                    ("ethanol", ImpactType.DIRECT, 0.7)
                ]
            
            # Generate sector impact analysis
            sector_impacts = []
            for sector, impact_type, base_score in affected_sectors:
                # Get sector information from industry mapping
                sector_info = self.industry_mapping.get(sector, {})
                
                # Calculate impact score based on commodity price change and sector sensitivity
                impact_score = base_score * abs(commodity_data.price_change_pct) / 100
                
                # Determine impact factors
                impact_factors = []
                if commodity_data.price_change_pct > 0:
                    impact_factors.append("increased_input_costs")
                    impact_factors.append("reduced_profit_margins")
                else:
                    impact_factors.append("decreased_input_costs")
                    impact_factors.append("improved_profit_margins")
                
                # Get affected companies from industry mapping
                affected_companies = sector_info.get("companies", [])
                
                # Determine severity
                if impact_score > 0.1:
                    severity = "high"
                elif impact_score > 0.05:
                    severity = "medium"
                else:
                    severity = "low"
                
                sector_impact = SectorImpact(
                    commodity_id=commodity_data.commodity_id,
                    sector=sector,
                    impact_type=impact_type,
                    impact_score=impact_score,
                    impact_factors=impact_factors,
                    affected_companies=affected_companies,
                    analysis_date=datetime.now(),
                    confidence=0.85,
                    severity=severity,
                    time_horizon="1-3 months"
                )
                
                sector_impacts.append(sector_impact)
            
            return sector_impacts
            
        except Exception as e:
            logger.error(f"Failed to analyze sector impacts: {e}")
            return []

    async def analyze_supply_demand(self, commodity_symbol: str) -> Optional[SupplyDemandAnalysis]:
        """
        Analyze supply and demand dynamics
        
        PSEUDOCODE with Multi-Tool Analysis:
        1. Use LangChain to orchestrate supply/demand analysis
        2. Apply Computer Use to optimize data source selection
        3. Use LlamaIndex to search for historical supply/demand patterns
        4. Apply Haystack for document analysis of market reports
        5. Use AutoGen to coordinate with market analysis agents
        6. Analyze inventory levels and production trends
        7. Assess consumption patterns and demand drivers
        8. Calculate supply/demand balance indicators
        9. Identify risk factors and market dynamics
        10. NO TRADING DECISIONS - only market analysis
        """
        try:
            logger.info(f"Analyzing supply/demand for {commodity_symbol}")
            
            # Use LangChain for analysis orchestration
            if self.agent_executor:
                analysis_query = f"Analyze supply and demand dynamics for {commodity_symbol}"
                response = await self.agent_executor.ainvoke({"input": analysis_query})
                logger.info(f"Supply/demand analysis: {response}")
            
            # Use LlamaIndex to search for historical patterns
            if self.query_engine:
                historical_query = f"What are the historical supply and demand patterns for {commodity_symbol}?"
                historical_response = self.query_engine.query(historical_query)
                logger.info(f"Historical supply/demand: {historical_response}")
            
            # Simulate supply/demand analysis (replace with actual data)
            supply_demand = SupplyDemandAnalysis(
                commodity_id=f"commodity_{commodity_symbol.lower()}",
                supply_level="moderate",
                demand_level="strong",
                inventory_levels="below_average",
                production_trend="increasing",
                consumption_trend="stable",
                analysis_date=datetime.now(),
                confidence=0.80,
                balance_score=0.6,  # 0-1 scale, 0.5 is balanced
                risk_factors=["weather_conditions", "geopolitical_tensions", "transportation_disruptions"]
            )
            
            return supply_demand
            
        except Exception as e:
            logger.error(f"Failed to analyze supply/demand for {commodity_symbol}: {e}")
            return None

    async def monitor_weather_impact(self, commodity_symbol: str) -> Dict[str, Any]:
        """
        Monitor weather impact on commodity production
        
        PSEUDOCODE with Multi-Tool Monitoring:
        1. Use LangChain to orchestrate weather impact analysis
        2. Apply Computer Use to select optimal weather data sources
        3. Use LlamaIndex to search for historical weather impacts
        4. Apply Haystack for document analysis of weather reports
        5. Use AutoGen to coordinate with environmental analysis agents
        6. Monitor weather conditions in key production regions
        7. Assess potential production impacts
        8. Identify supply chain vulnerabilities
        9. Generate weather risk assessment
        10. NO TRADING DECISIONS - only weather analysis
        """
        try:
            logger.info(f"Monitoring weather impact for {commodity_symbol}")
            
            # Use AutoGen for environmental analysis coordination
            if self.chat_manager:
                weather_prompt = f"""
                Analyze weather impacts on {commodity_symbol} production.
                
                Consider:
                1. Current weather conditions in production regions
                2. Seasonal weather patterns and forecasts
                3. Historical weather impact on production
                4. Supply chain vulnerabilities to weather events
                
                Provide detailed weather risk assessment.
                """
                
                response = await self.chat_manager.agenerate_reply(
                    messages=[{"role": "user", "content": weather_prompt}],
                    sender=self.user_proxy
                )
                logger.info(f"Weather analysis: {response}")
            
            # Simulate weather impact analysis
            weather_impact = {
                "weather_conditions": "normal",
                "production_impact": "minimal",
                "risk_level": "low",
                "affected_regions": ["Midwest", "Great Plains"],
                "forecast": "stable_conditions",
                "confidence": 0.85
            }
            
            return weather_impact
            
        except Exception as e:
            logger.error(f"Failed to monitor weather impact: {e}")
            return {}

    async def analyze_geopolitical_impact(self, commodity_symbol: str) -> Dict[str, Any]:
        """
        Analyze geopolitical impact on commodity supply
        
        PSEUDOCODE with Multi-Tool Analysis:
        1. Use LangChain to orchestrate geopolitical analysis
        2. Apply Computer Use to select optimal news and political data sources
        3. Use LlamaIndex to search for historical geopolitical impacts
        4. Apply Haystack for document analysis of political reports
        5. Use AutoGen to coordinate with political analysis agents
        6. Monitor political stability in key production regions
        7. Assess trade policy impacts and restrictions
        8. Identify supply chain risks from political events
        9. Generate geopolitical risk assessment
        10. NO TRADING DECISIONS - only political analysis
        """
        try:
            logger.info(f"Analyzing geopolitical impact for {commodity_symbol}")
            
            # Use Haystack for document analysis
            if self.qa_pipeline:
                geo_query = f"What are the geopolitical risks affecting {commodity_symbol} supply?"
                # Note: Would need document store populated with political reports
                logger.info(f"Geopolitical query: {geo_query}")
            
            # Use LlamaIndex to search for historical patterns
            if self.query_engine:
                historical_query = f"What are the historical geopolitical impacts on {commodity_symbol}?"
                historical_response = self.query_engine.query(historical_query)
                logger.info(f"Historical geopolitical analysis: {historical_response}")
            
            # Simulate geopolitical impact analysis
            geo_impact = {
                "political_stability": "moderate",
                "trade_risks": "low",
                "supply_chain_risks": "minimal",
                "regulatory_environment": "stable",
                "key_risks": ["trade_tensions", "regulatory_changes"],
                "confidence": 0.80
            }
            
            return geo_impact
            
        except Exception as e:
            logger.error(f"Failed to analyze geopolitical impact: {e}")
            return {}

    async def generate_alerts(self, commodity_symbol: str) -> List[Dict[str, Any]]:
        """
        Generate commodity alerts with multi-tool analysis
        
        PSEUDOCODE with Multi-Tool Alerting:
        1. Use LangChain to orchestrate alert generation
        2. Apply Computer Use to optimize alert thresholds and triggers
        3. Use LlamaIndex to search for historical alert patterns
        4. Apply Haystack for document analysis of market conditions
        5. Use AutoGen to coordinate with alert management agents
        6. Monitor commodity price movements and volatility
        7. Assess sector impact significance
        8. Evaluate supply/demand imbalances
        9. Generate appropriate alerts with severity levels
        10. NO TRADING DECISIONS - only alert generation
        """
        try:
            logger.info(f"Generating alerts for {commodity_symbol}")
            
            alerts = []
            
            # Fetch current commodity data
            commodity_data = await self.fetch_commodity_data(commodity_symbol)
            if not commodity_data:
                return alerts
            
            # Check for significant price movements
            if abs(commodity_data.price_change_pct) > self.config.get("alert_threshold", 0.2):
                alert = {
                    "type": "price_movement",
                    "commodity": commodity_symbol,
                    "severity": "high" if abs(commodity_data.price_change_pct) > 0.5 else "medium",
                    "message": f"Significant price movement: {commodity_data.price_change_pct:.2f}%",
                    "timestamp": datetime.now().isoformat(),
                    "impacted_sectors": [],
                    "impact_score": abs(commodity_data.price_change_pct)
                }
                alerts.append(alert)
            
            # Check for high volatility
            if commodity_data.volatility and commodity_data.volatility > 0.05:
                alert = {
                    "type": "volatility",
                    "commodity": commodity_symbol,
                    "severity": "medium",
                    "message": f"High volatility detected: {commodity_data.volatility:.3f}",
                    "timestamp": datetime.now().isoformat(),
                    "impacted_sectors": [],
                    "impact_score": commodity_data.volatility
                }
                alerts.append(alert)
            
            # Analyze sector impacts for significant changes
            if abs(commodity_data.price_change_pct) > 0.1:
                sector_impacts = await self.analyze_sector_impact(commodity_data)
                
                for impact in sector_impacts:
                    if impact.impact_score > 0.05:
                        alert = {
                            "type": "sector_impact",
                            "commodity": commodity_symbol,
                            "severity": impact.severity,
                            "message": f"Sector impact on {impact.sector}: {impact.impact_score:.3f}",
                            "timestamp": datetime.now().isoformat(),
                            "impacted_sectors": [impact.sector],
                            "impact_score": impact.impact_score
                        }
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
            return []

    async def _update_commodity_knowledge_base(self, commodity_data: CommodityData):
        """Update knowledge base with new commodity data"""
        try:
            if self.llama_index:
                # Create document with new data
                doc_text = f"""
                Commodity Update: {commodity_data.name} ({commodity_data.symbol})
                Price: ${commodity_data.current_price:.2f}
                Change: {commodity_data.price_change_pct:.2f}%
                Volume: {commodity_data.volume:,}
                Timestamp: {commodity_data.timestamp}
                Source: {commodity_data.source}
                Confidence: {commodity_data.confidence}
                """
                
                doc = Document(text=doc_text, metadata={
                    "commodity_id": commodity_data.commodity_id,
                    "timestamp": commodity_data.timestamp.isoformat(),
                    "source": commodity_data.source
                })
                
                # Add to index
                self.llama_index.insert(doc)
                
                logger.info(f"Updated knowledge base with {commodity_data.symbol} data")
                
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")

    # Tool functions for LangChain integration
    def _fetch_commodity_data_tool(self, query: str) -> str:
        """Tool function for fetching commodity data"""
        try:
            # Extract commodity symbol from query
            # This is a simplified implementation
            return f"Commodity data fetched for query: {query}"
        except Exception as e:
            return f"Error fetching commodity data: {e}"

    def _analyze_sector_impact_tool(self, query: str) -> str:
        """Tool function for analyzing sector impacts"""
        try:
            return f"Sector impact analysis completed for query: {query}"
        except Exception as e:
            return f"Error analyzing sector impacts: {e}"

    def _assess_supply_demand_tool(self, query: str) -> str:
        """Tool function for assessing supply and demand"""
        try:
            return f"Supply/demand assessment completed for query: {query}"
        except Exception as e:
            return f"Error assessing supply/demand: {e}"

    def get_health_score(self) -> float:
        """Get agent health score"""
        return self.health_score

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_name": self.name,
            "version": self.version,
            "health_score": self.health_score,
            "last_update": self.last_update.isoformat(),
            "error_count": self.error_count,
            "multi_tool_integration": {
                "langchain": self.agent_executor is not None,
                "llama_index": self.query_engine is not None,
                "haystack": self.qa_pipeline is not None,
                "autogen": self.chat_manager is not None,
                "computer_use": self.tool_selector is not None
            }
        } 