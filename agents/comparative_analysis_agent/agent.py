"""
Comparative Analysis Agent - Multi-Entity Analysis and Benchmarking
==================================================================

This agent performs comparative analysis across multiple entities, sectors,
and time periods to identify patterns, correlations, and relative performance.

CRITICAL SYSTEM POLICY: NO TRADING DECISIONS OR RECOMMENDATIONS
This agent only performs comparative analysis, benchmarking, and pattern
identification. No trading advice is provided.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics
from scipy import stats
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import uuid

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
SYSTEM POLICY: This agent is STRICTLY for comparative analysis and benchmarking.
NO TRADING DECISIONS should be made. All analysis is for research and
investigation purposes only.

AI REASONING: The agent should:
1. Perform comparative analysis across multiple entities
2. Identify patterns, correlations, and relative performance
3. Generate benchmarking reports and insights
4. NEVER make buy/sell recommendations
5. NEVER provide trading advice
6. NEVER execute trades
"""

@dataclass
class ComparisonRequest:
    """Represents a comparative analysis request"""
    request_id: str
    entities: List[str]  # Tickers, sectors, or other entities
    metrics: List[str]   # Metrics to compare
    time_period: str     # Time period for analysis
    comparison_type: str # 'peer', 'sector', 'historical', 'benchmark'
    analysis_depth: str  # 'basic', 'detailed', 'comprehensive'
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ComparisonResult:
    """Represents the result of a comparative analysis"""
    request_id: str
    entities: List[str]
    metrics: List[str]
    analysis_type: str
    results: Dict[str, Any]
    insights: List[str]
    confidence: float
    data_quality: float
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()

@dataclass
class BenchmarkData:
    """Represents benchmark data for comparison"""
    benchmark_id: str
    name: str
    category: str
    data: Dict[str, Any]
    last_updated: datetime
    confidence: float

class ComparativeAnalysisAgent:
    """
    Intelligent comparative analysis and benchmarking agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "ComparativeAnalysisAgent"
        self.version = "1.0.0"
        self.comparison_queue = []
        self.comparison_history = []
        self.benchmarks = {}
        
        # MCP Communication setup
        self.mcp_endpoint = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8000/mcp')
        self.message_queue = []
        
        # Error handling and recovery
        self.error_count = 0
        self.max_retries = 3
        self.health_score = 1.0
        
        # Analysis metrics
        self.comparisons_performed = 0
        self.insights_generated = 0
        self.benchmarks_updated = 0
        
    async def initialize(self):
        """Initialize the comparative analysis agent"""
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # AI REASONING: Initialize comparative analysis capabilities
        # PSEUDOCODE:
        # 1. Load comparative analysis models and algorithms
        # 2. Initialize benchmarking databases and reference data
        # 3. Set up statistical analysis and correlation engines
        # 4. Configure peer group and sector classification systems
        # 5. Initialize data validation and quality assessment
        # 6. Set up MCP communication for data requests
        # 7. Configure analysis depth and confidence scoring
        # 8. Initialize pattern recognition and anomaly detection
        # 9. NO TRADING DECISIONS - only comparative analysis
        
        try:
            # Initialize benchmarks
            await self._initialize_benchmarks()
            
            # Set up analysis engines
            await self._setup_analysis_engines()
            
            # Load reference data
            await self._load_reference_data()
            
        except Exception as e:
            logger.error(f"Error initializing comparative analysis: {e}")
            raise
            
        logger.info(f"{self.name} initialized successfully")
    
    async def _initialize_benchmarks(self):
        """
        Initialize benchmark data and reference points
        """
        # AI REASONING: Benchmark initialization
        # PSEUDOCODE:
        # 1. Load industry benchmarks and reference data
        # 2. Initialize sector-specific benchmarks
        # 3. Set up historical benchmark tracking
        # 4. Configure peer group definitions
        # 5. Validate benchmark data quality and relevance
        # 6. Set up benchmark update mechanisms
        # 7. NO TRADING DECISIONS - only benchmark setup
        
        # Initialize sector benchmarks
        sector_benchmarks = {
            'technology': {'pe_ratio': 25.0, 'revenue_growth': 0.15, 'profit_margin': 0.20},
            'healthcare': {'pe_ratio': 20.0, 'revenue_growth': 0.10, 'profit_margin': 0.15},
            'finance': {'pe_ratio': 15.0, 'revenue_growth': 0.08, 'profit_margin': 0.25},
            'energy': {'pe_ratio': 12.0, 'revenue_growth': 0.05, 'profit_margin': 0.10}
        }
        
        for sector, data in sector_benchmarks.items():
            benchmark = BenchmarkData(
                benchmark_id=str(uuid.uuid4()),
                name=f"{sector}_sector_benchmark",
                category="sector",
                data=data,
                last_updated=datetime.now(),
                confidence=0.8
            )
            self.benchmarks[benchmark.benchmark_id] = benchmark
        
        logger.info(f"Initialized {len(self.benchmarks)} benchmarks")
    
    async def _setup_analysis_engines(self):
        """
        Set up statistical analysis and correlation engines
        """
        # AI REASONING: Analysis engine setup
        # PSEUDOCODE:
        # 1. Initialize statistical analysis libraries
        # 2. Set up correlation and regression engines
        # 3. Configure pattern recognition algorithms
        # 4. Initialize anomaly detection systems
        # 5. Set up data normalization and scaling
        # 6. Configure confidence interval calculations
        # 7. NO TRADING DECISIONS - only engine setup
        
        logger.info("Analysis engines initialized")
    
    async def _load_reference_data(self):
        """
        Load reference data for comparative analysis
        """
        # AI REASONING: Reference data loading
        # PSEUDOCODE:
        # 1. Load historical performance data
        # 2. Initialize peer group classifications
        # 3. Load sector and industry classifications
        # 4. Set up market cap and size categories
        # 5. Load geographical and regional data
        # 6. Validate data quality and completeness
        # 7. NO TRADING DECISIONS - only data loading
        
        logger.info("Reference data loaded")
    
    async def perform_peer_comparison(self, target_entity: str, 
                                    peer_group: List[str] = None,
                                    metrics: List[str] = None) -> ComparisonResult:
        """
        Perform peer comparison analysis
        """
        # AI REASONING: Peer comparison analysis
        # PSEUDOCODE:
        # 1. Identify appropriate peer group for target entity
        # 2. Collect relevant metrics for all entities
        # 3. Normalize and standardize data for comparison
        # 4. Calculate relative performance and rankings
        # 5. Identify outliers and anomalies
        # 6. Generate insights and recommendations
        # 7. Calculate confidence and data quality scores
        # 8. NO TRADING DECISIONS - only peer analysis
        
        request_id = str(uuid.uuid4())
        
        # Identify peer group if not provided
        if not peer_group:
            peer_group = await self._identify_peer_group(target_entity)
        
        # Set default metrics if not provided
        if not metrics:
            metrics = ['pe_ratio', 'revenue_growth', 'profit_margin', 'debt_to_equity']
        
        # Collect data for all entities
        entity_data = {}
        for entity in [target_entity] + peer_group:
            data = await self._collect_entity_data(entity, metrics)
            if data:
                entity_data[entity] = data
        
        # Perform comparative analysis
        analysis_results = await self._analyze_peer_comparison(target_entity, entity_data, metrics)
        
        # Generate insights
        insights = await self._generate_peer_insights(target_entity, entity_data, analysis_results)
        
        # Calculate confidence and quality scores
        confidence = self._calculate_analysis_confidence(entity_data, analysis_results)
        data_quality = self._assess_data_quality(entity_data)
        
        result = ComparisonResult(
            request_id=request_id,
            entities=[target_entity] + peer_group,
            metrics=metrics,
            analysis_type='peer_comparison',
            results=analysis_results,
            insights=insights,
            confidence=confidence,
            data_quality=data_quality
        )
        
        self.comparisons_performed += 1
        self.insights_generated += len(insights)
        
        return result
    
    async def _identify_peer_group(self, target_entity: str) -> List[str]:
        """
        Identify appropriate peer group for target entity
        """
        # AI REASONING: Peer group identification
        # PSEUDOCODE:
        # 1. Analyze target entity characteristics (sector, size, geography)
        # 2. Search for entities with similar characteristics
        # 3. Apply size and market cap filters
        # 4. Consider geographical and regulatory factors
        # 5. Validate peer group relevance and quality
        # 6. Generate peer group recommendations
        # 7. NO TRADING DECISIONS - only peer identification
        
        # Mock peer group identification
        sector_peers = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'TSLA': ['F', 'GM', 'TM', 'NIO', 'XPEV'],
            'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS']
        }
        
        return sector_peers.get(target_entity, ['MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'])
    
    async def _collect_entity_data(self, entity: str, metrics: List[str]) -> Dict[str, Any]:
        """
        Collect data for a specific entity and metrics
        """
        # AI REASONING: Entity data collection
        # PSEUDOCODE:
        # 1. Query knowledge base for entity data
        # 2. Validate data completeness and quality
        # 3. Normalize data formats and units
        # 4. Handle missing data and outliers
        # 5. Calculate derived metrics if needed
        # 6. Validate data consistency and accuracy
        # 7. NO TRADING DECISIONS - only data collection
        
        # Mock data collection
        mock_data = {
            'AAPL': {'pe_ratio': 25.5, 'revenue_growth': 0.12, 'profit_margin': 0.22, 'debt_to_equity': 0.15},
            'MSFT': {'pe_ratio': 28.2, 'revenue_growth': 0.14, 'profit_margin': 0.35, 'debt_to_equity': 0.12},
            'GOOGL': {'pe_ratio': 24.8, 'revenue_growth': 0.18, 'profit_margin': 0.28, 'debt_to_equity': 0.08},
            'AMZN': {'pe_ratio': 45.2, 'revenue_growth': 0.22, 'profit_margin': 0.05, 'debt_to_equity': 0.25},
            'META': {'pe_ratio': 22.1, 'revenue_growth': 0.16, 'profit_margin': 0.32, 'debt_to_equity': 0.10},
            'NVDA': {'pe_ratio': 35.6, 'revenue_growth': 0.45, 'profit_margin': 0.40, 'debt_to_equity': 0.05}
        }
        
        return mock_data.get(entity, {})
    
    async def _analyze_peer_comparison(self, target_entity: str, 
                                     entity_data: Dict[str, Any], 
                                     metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze peer comparison data
        """
        # AI REASONING: Peer comparison analysis
        # PSEUDOCODE:
        # 1. Calculate statistical measures for each metric
        # 2. Determine target entity ranking and percentile
        # 3. Identify outliers and anomalies
        # 4. Calculate correlation coefficients
        # 5. Perform trend analysis and pattern recognition
        # 6. Generate comparative statistics and rankings
        # 7. Assess relative performance and positioning
        # 8. NO TRADING DECISIONS - only comparative analysis
        
        analysis_results = {
            'target_entity': target_entity,
            'metrics_analysis': {},
            'rankings': {},
            'percentiles': {},
            'outliers': [],
            'correlations': {},
            'summary_statistics': {}
        }
        
        for metric in metrics:
            # Collect metric values
            values = []
            entities = []
            for entity, data in entity_data.items():
                if metric in data:
                    values.append(data[metric])
                    entities.append(entity)
            
            if not values:
                continue
            
            # Calculate statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            
            # Find target entity value and ranking
            target_value = entity_data[target_entity].get(metric)
            if target_value is not None:
                target_rank = sum(1 for v in values if v > target_value) + 1
                target_percentile = (target_rank / len(values)) * 100
                
                analysis_results['rankings'][metric] = target_rank
                analysis_results['percentiles'][metric] = target_percentile
            
            # Store summary statistics
            analysis_results['summary_statistics'][metric] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            
            # Identify outliers
            outlier_threshold = 2 * std_val
            outliers = [entity for entity, val in zip(entities, values) 
                       if abs(val - mean_val) > outlier_threshold]
            if outliers:
                analysis_results['outliers'].extend(outliers)
        
        return analysis_results
    
    async def _generate_peer_insights(self, target_entity: str, 
                                    entity_data: Dict[str, Any], 
                                    analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate insights from peer comparison
        """
        # AI REASONING: Insight generation
        # PSEUDOCODE:
        # 1. Analyze relative performance patterns
        # 2. Identify strengths and weaknesses
        # 3. Detect anomalies and outliers
        # 4. Generate actionable insights
        # 5. Consider industry and sector context
        # 6. Assess competitive positioning
        # 7. NO TRADING DECISIONS - only insight generation
        
        insights = []
        
        # Analyze rankings and percentiles
        for metric, percentile in analysis_results.get('percentiles', {}).items():
            if percentile > 75:
                insights.append(f"{target_entity} ranks in top 25% for {metric}")
            elif percentile < 25:
                insights.append(f"{target_entity} ranks in bottom 25% for {metric}")
        
        # Analyze outliers
        if target_entity in analysis_results.get('outliers', []):
            insights.append(f"{target_entity} shows outlier behavior in some metrics")
        
        # Generate comparative insights
        target_data = entity_data.get(target_entity, {})
        if target_data:
            avg_pe = np.mean([data.get('pe_ratio', 0) for data in entity_data.values() if data.get('pe_ratio')])
            if target_data.get('pe_ratio', 0) > avg_pe * 1.2:
                insights.append(f"{target_entity} has higher P/E ratio than peer average")
            elif target_data.get('pe_ratio', 0) < avg_pe * 0.8:
                insights.append(f"{target_entity} has lower P/E ratio than peer average")
        
        return insights
    
    def _calculate_analysis_confidence(self, entity_data: Dict[str, Any], 
                                     analysis_results: Dict[str, Any]) -> float:
        """
        Calculate confidence in analysis results
        """
        # AI REASONING: Confidence calculation
        # PSEUDOCODE:
        # 1. Assess data completeness and quality
        # 2. Evaluate sample size and statistical significance
        # 3. Consider data freshness and relevance
        # 4. Factor in analysis methodology quality
        # 5. Generate confidence score (0-1)
        # 6. NO TRADING DECISIONS - only confidence assessment
        
        # Base confidence on data completeness
        total_entities = len(entity_data)
        if total_entities < 3:
            return 0.3
        elif total_entities < 5:
            return 0.6
        else:
            return 0.8
    
    def _assess_data_quality(self, entity_data: Dict[str, Any]) -> float:
        """
        Assess data quality for analysis
        """
        # AI REASONING: Data quality assessment
        # PSEUDOCODE:
        # 1. Check data completeness across entities
        # 2. Assess data consistency and accuracy
        # 3. Evaluate data freshness and timeliness
        # 4. Check for outliers and anomalies
        # 5. Generate quality score (0-1)
        # 6. NO TRADING DECISIONS - only quality assessment
        
        if not entity_data:
            return 0.0
        
        # Simple quality assessment based on data availability
        total_data_points = sum(len(data) for data in entity_data.values())
        expected_data_points = len(entity_data) * 4  # Assuming 4 metrics
        
        return min(1.0, total_data_points / expected_data_points)
    
    async def perform_sector_analysis(self, sector: str, 
                                    metrics: List[str] = None) -> ComparisonResult:
        """
        Perform sector-wide comparative analysis
        """
        # AI REASONING: Sector analysis
        # PSEUDOCODE:
        # 1. Identify all entities in the sector
        # 2. Collect sector-wide metrics and data
        # 3. Calculate sector averages and benchmarks
        # 4. Identify sector leaders and laggards
        # 5. Analyze sector trends and patterns
        # 6. Compare with other sectors
        # 7. Generate sector insights and recommendations
        # 8. NO TRADING DECISIONS - only sector analysis
        
        request_id = str(uuid.uuid4())
        
        # Set default metrics if not provided
        if not metrics:
            metrics = ['pe_ratio', 'revenue_growth', 'profit_margin', 'market_cap']
        
        # Collect sector data
        sector_entities = await self._get_sector_entities(sector)
        sector_data = {}
        
        for entity in sector_entities:
            data = await self._collect_entity_data(entity, metrics)
            if data:
                sector_data[entity] = data
        
        # Perform sector analysis
        analysis_results = await self._analyze_sector_data(sector, sector_data, metrics)
        
        # Generate sector insights
        insights = await self._generate_sector_insights(sector, sector_data, analysis_results)
        
        # Calculate confidence and quality
        confidence = self._calculate_analysis_confidence(sector_data, analysis_results)
        data_quality = self._assess_data_quality(sector_data)
        
        result = ComparisonResult(
            request_id=request_id,
            entities=sector_entities,
            metrics=metrics,
            analysis_type='sector_analysis',
            results=analysis_results,
            insights=insights,
            confidence=confidence,
            data_quality=data_quality
        )
        
        self.comparisons_performed += 1
        self.insights_generated += len(insights)
        
        return result
    
    async def _get_sector_entities(self, sector: str) -> List[str]:
        """
        Get entities in a specific sector
        """
        # AI REASONING: Sector entity identification
        # PSEUDOCODE:
        # 1. Query sector classification database
        # 2. Filter entities by sector classification
        # 3. Validate sector assignments and accuracy
        # 4. Apply size and liquidity filters
        # 5. Generate sector entity list
        # 6. NO TRADING DECISIONS - only entity identification
        
        sector_entities = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR'],
            'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL']
        }
        
        return sector_entities.get(sector, [])
    
    async def _analyze_sector_data(self, sector: str, sector_data: Dict[str, Any], 
                                 metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze sector-wide data
        """
        # AI REASONING: Sector data analysis
        # PSEUDOCODE:
        # 1. Calculate sector-wide statistics for each metric
        # 2. Identify sector leaders and laggards
        # 3. Analyze sector trends and patterns
        # 4. Compare with sector benchmarks
        # 5. Identify sector-specific insights
        # 6. Generate sector performance rankings
        # 7. NO TRADING DECISIONS - only sector analysis
        
        analysis_results = {
            'sector': sector,
            'sector_statistics': {},
            'leaders': {},
            'laggards': {},
            'benchmark_comparison': {},
            'trends': {}
        }
        
        for metric in metrics:
            values = []
            entities = []
            for entity, data in sector_data.items():
                if metric in data:
                    values.append(data[metric])
                    entities.append(entity)
            
            if not values:
                continue
            
            # Calculate sector statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            
            analysis_results['sector_statistics'][metric] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            
            # Identify leaders and laggards
            sorted_entities = sorted(zip(entities, values), key=lambda x: x[1], reverse=True)
            analysis_results['leaders'][metric] = sorted_entities[:3]
            analysis_results['laggards'][metric] = sorted_entities[-3:]
            
            # Compare with sector benchmark
            benchmark = self.benchmarks.get(f"{sector}_sector_benchmark")
            if benchmark and metric in benchmark.data:
                benchmark_value = benchmark.data[metric]
                analysis_results['benchmark_comparison'][metric] = {
                    'sector_average': mean_val,
                    'benchmark': benchmark_value,
                    'difference': mean_val - benchmark_value,
                    'percentage_diff': ((mean_val - benchmark_value) / benchmark_value) * 100
                }
        
        return analysis_results
    
    async def _generate_sector_insights(self, sector: str, sector_data: Dict[str, Any], 
                                      analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate sector-specific insights
        """
        # AI REASONING: Sector insight generation
        # PSEUDOCODE:
        # 1. Analyze sector performance patterns
        # 2. Identify sector strengths and weaknesses
        # 3. Compare with benchmarks and other sectors
        # 4. Generate sector-specific recommendations
        # 5. Assess sector trends and outlook
        # 6. NO TRADING DECISIONS - only insight generation
        
        insights = []
        
        # Analyze benchmark comparisons
        for metric, comparison in analysis_results.get('benchmark_comparison', {}).items():
            if comparison['percentage_diff'] > 10:
                insights.append(f"{sector} sector outperforms benchmark for {metric}")
            elif comparison['percentage_diff'] < -10:
                insights.append(f"{sector} sector underperforms benchmark for {metric}")
        
        # Analyze sector leaders
        for metric, leaders in analysis_results.get('leaders', {}).items():
            if leaders:
                top_entity = leaders[0][0]
                insights.append(f"{top_entity} leads {sector} sector in {metric}")
        
        # Generate sector summary
        total_entities = len(sector_data)
        insights.append(f"Analyzed {total_entities} entities in {sector} sector")
        
        return insights
    
    async def perform_historical_comparison(self, entity: str, 
                                          time_period: str = "1y",
                                          metrics: List[str] = None) -> ComparisonResult:
        """
        Perform historical comparison analysis
        """
        # AI REASONING: Historical comparison analysis
        # PSEUDOCODE:
        # 1. Collect historical data for specified time period
        # 2. Calculate historical trends and patterns
        # 3. Identify performance cycles and seasonality
        # 4. Compare with historical benchmarks
        # 5. Analyze performance consistency and volatility
        # 6. Generate historical insights and trends
        # 7. NO TRADING DECISIONS - only historical analysis
        
        request_id = str(uuid.uuid4())
        
        # Set default metrics if not provided
        if not metrics:
            metrics = ['revenue_growth', 'profit_margin', 'pe_ratio', 'return_on_equity']
        
        # Collect historical data
        historical_data = await self._collect_historical_data(entity, time_period, metrics)
        
        # Perform historical analysis
        analysis_results = await self._analyze_historical_data(entity, historical_data, metrics)
        
        # Generate historical insights
        insights = await self._generate_historical_insights(entity, historical_data, analysis_results)
        
        # Calculate confidence and quality
        confidence = self._calculate_analysis_confidence({'historical': historical_data}, analysis_results)
        data_quality = self._assess_data_quality({'historical': historical_data})
        
        result = ComparisonResult(
            request_id=request_id,
            entities=[entity],
            metrics=metrics,
            analysis_type='historical_comparison',
            results=analysis_results,
            insights=insights,
            confidence=confidence,
            data_quality=data_quality
        )
        
        self.comparisons_performed += 1
        self.insights_generated += len(insights)
        
        return result
    
    async def _collect_historical_data(self, entity: str, time_period: str, 
                                     metrics: List[str]) -> Dict[str, Any]:
        """
        Collect historical data for analysis
        """
        # AI REASONING: Historical data collection
        # PSEUDOCODE:
        # 1. Query historical database for entity data
        # 2. Filter data by time period and metrics
        # 3. Validate data completeness and quality
        # 4. Handle missing data and outliers
        # 5. Normalize data formats and units
        # 6. NO TRADING DECISIONS - only data collection
        
        # Mock historical data
        historical_data = {
            'revenue_growth': [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35],
            'profit_margin': [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
            'pe_ratio': [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
            'return_on_equity': [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
        }
        
        return {metric: historical_data.get(metric, []) for metric in metrics}
    
    async def _analyze_historical_data(self, entity: str, historical_data: Dict[str, Any], 
                                     metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze historical data for trends and patterns
        """
        # AI REASONING: Historical data analysis
        # PSEUDOCODE:
        # 1. Calculate trend lines and growth rates
        # 2. Identify seasonal patterns and cycles
        # 3. Calculate volatility and consistency measures
        # 4. Compare with historical benchmarks
        # 5. Identify performance inflection points
        # 6. Generate trend analysis and forecasts
        # 7. NO TRADING DECISIONS - only historical analysis
        
        analysis_results = {
            'entity': entity,
            'trends': {},
            'volatility': {},
            'growth_rates': {},
            'seasonality': {},
            'inflection_points': []
        }
        
        for metric, values in historical_data.items():
            if not values:
                continue
            
            # Calculate trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            analysis_results['trends'][metric] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value
            }
            
            # Calculate volatility
            analysis_results['volatility'][metric] = {
                'std': np.std(values),
                'coefficient_of_variation': np.std(values) / np.mean(values)
            }
            
            # Calculate growth rates
            if len(values) > 1:
                growth_rates = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
                analysis_results['growth_rates'][metric] = {
                    'average_growth': np.mean(growth_rates),
                    'growth_volatility': np.std(growth_rates)
                }
        
        return analysis_results
    
    async def _generate_historical_insights(self, entity: str, historical_data: Dict[str, Any], 
                                          analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate insights from historical analysis
        """
        # AI REASONING: Historical insight generation
        # PSEUDOCODE:
        # 1. Analyze trend strength and direction
        # 2. Identify performance patterns and cycles
        # 3. Assess consistency and stability
        # 4. Generate historical context and insights
        # 5. Compare with historical benchmarks
        # 6. NO TRADING DECISIONS - only insight generation
        
        insights = []
        
        # Analyze trends
        for metric, trend in analysis_results.get('trends', {}).items():
            if trend['r_squared'] > 0.7:
                if trend['slope'] > 0:
                    insights.append(f"Strong positive trend in {metric}")
                else:
                    insights.append(f"Strong negative trend in {metric}")
            elif trend['r_squared'] < 0.3:
                insights.append(f"High volatility in {metric} with weak trend")
        
        # Analyze volatility
        for metric, volatility in analysis_results.get('volatility', {}).items():
            if volatility['coefficient_of_variation'] > 0.5:
                insights.append(f"High volatility in {metric}")
            elif volatility['coefficient_of_variation'] < 0.1:
                insights.append(f"Low volatility in {metric}")
        
        return insights
    
    async def run(self):
        """
        Main execution loop for comparative analysis
        """
        logger.info(f"Starting {self.name} with comparative analysis")
        
        # PSEUDOCODE for main execution loop:
        # 1. Initialize comparative analysis capabilities
        # 2. Start continuous monitoring loop:
        #    - Monitor for comparison requests
        #    - Update benchmarks and reference data
        #    - Perform scheduled comparative analyses
        #    - Generate insights and recommendations
        #    - Update system health and metrics
        # 3. Monitor system performance and adjust frequency
        # 4. Handle errors and recovery
        # 5. NO TRADING DECISIONS - only comparative analysis
        
        while True:
            try:
                # Process comparison queue
                await self._process_comparison_queue()
                
                # Update benchmarks
                await self._update_benchmarks()
                
                # Update health metrics
                await self.update_health_metrics()
                
                # Sleep interval based on analysis requirements
                sleep_interval = self.calculate_sleep_interval()
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                await self.handle_error(e, "main_loop")
                await asyncio.sleep(60)
    
    async def _process_comparison_queue(self):
        """Process pending comparison requests"""
        # AI REASONING: Queue processing
        # PSEUDOCODE:
        # 1. Process comparison requests in priority order
        # 2. Execute appropriate analysis type
        # 3. Generate results and insights
        # 4. Update comparison history
        # 5. NO TRADING DECISIONS - only queue processing
        
        if self.comparison_queue:
            request = self.comparison_queue.pop(0)
            
            if request.comparison_type == 'peer':
                result = await self.perform_peer_comparison(
                    request.entities[0], request.entities[1:], request.metrics
                )
            elif request.comparison_type == 'sector':
                result = await self.perform_sector_analysis(
                    request.entities[0], request.metrics
                )
            elif request.comparison_type == 'historical':
                result = await self.perform_historical_comparison(
                    request.entities[0], request.time_period, request.metrics
                )
            
            self.comparison_history.append(result)
    
    async def _update_benchmarks(self):
        """Update benchmark data"""
        # AI REASONING: Benchmark updates
        # PSEUDOCODE:
        # 1. Collect latest market data
        # 2. Update sector and industry benchmarks
        # 3. Validate benchmark accuracy and relevance
        # 4. Update confidence scores
        # 5. NO TRADING DECISIONS - only benchmark updates
        
        for benchmark in self.benchmarks.values():
            benchmark.last_updated = datetime.now()
            self.benchmarks_updated += 1
    
    async def update_health_metrics(self):
        """Update agent health and performance metrics"""
        # AI REASONING: Health monitoring and optimization
        # PSEUDOCODE:
        # 1. Calculate analysis success rate and quality
        # 2. Monitor insight generation effectiveness
        # 3. Track benchmark accuracy and relevance
        # 4. Update health score based on performance
        # 5. Identify optimization opportunities
        
        self.health_score = min(1.0, self.comparisons_performed / max(len(self.comparison_history), 1))
        
        logger.info(f"Health metrics: {self.comparisons_performed} comparisons, {self.insights_generated} insights, health: {self.health_score:.2f}")
    
    def calculate_sleep_interval(self) -> int:
        """Calculate sleep interval based on analysis requirements"""
        # AI REASONING: Dynamic interval calculation
        # PSEUDOCODE:
        # 1. Assess current analysis queue and workload
        # 2. Consider benchmark update frequency
        # 3. Factor in analysis complexity and time requirements
        # 4. Adjust interval for optimal performance
        
        base_interval = 1800  # 30 minutes for analysis operations
        
        # Adjust based on queue size
        if len(self.comparison_queue) > 5:
            base_interval = 900  # 15 minutes if queue is large
        elif len(self.comparison_queue) < 2:
            base_interval = 3600  # 1 hour if queue is small
        
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
        "analysis_depth": "detailed",
        "confidence_threshold": 0.7,
        "max_entities_per_analysis": 20
    }
    
    agent = ComparativeAnalysisAgent(config)
    await agent.initialize()
    
    try:
        # Run the agent
        await agent.run()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 