"""
Comparative Analysis Agent - Multi-Source Data Comparison
=======================================================

This agent performs comprehensive comparative analysis across different
data sources, metrics, and time periods to identify patterns and insights.

CRITICAL SYSTEM POLICY: NO TRADING DECISIONS OR RECOMMENDATIONS
This agent only performs data comparison, analysis, and pattern recognition.
No trading advice is provided.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonMetric:
    """Represents a comparison metric"""
    name: str
    value: float
    source: str
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ComparisonResult:
    """Represents a comparison analysis result"""
    comparison_id: str
    metric_name: str
    sources: List[str]
    values: List[float]
    differences: List[float]
    correlation: float
    significance: float
    confidence: float
    insights: List[str]
    recommendations: List[str]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class ComparativeAnalysisAgent:
    """
    Comprehensive comparative analysis agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "ComparativeAnalysisAgent"
        self.version = "1.0.0"
        self.comparison_history = []
        self.analysis_cache = {}
        
    async def initialize(self):
        """Initialize the comparative analysis agent"""
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # AI REASONING: Initialize comparative analysis capabilities
        # PSEUDOCODE:
        # 1. Set up statistical analysis and comparison frameworks
        # 2. Initialize data normalization and standardization tools
        # 3. Configure correlation and significance testing methods
        # 4. Set up visualization and reporting capabilities
        # 5. Initialize pattern recognition and anomaly detection
        # 6. Configure multi-source data integration
        # 7. Set up comparative analysis caching and optimization
        # 8. Initialize MCP communication for analysis results
        
        logger.info(f"{self.name} initialized successfully")
    
    async def compare_metrics(self, metrics: List[ComparisonMetric], 
                            analysis_type: str = "comprehensive") -> ComparisonResult:
        """
        Perform comprehensive comparison of metrics
        """
        # AI REASONING: Multi-metric comparative analysis
        # PSEUDOCODE:
        # 1. Validate input metrics and data quality
        # 2. Normalize and standardize metric values
        # 3. Calculate statistical differences and correlations
        # 4. Perform significance testing and confidence analysis
        # 5. Identify patterns, trends, and anomalies
        # 6. Generate comparative insights and observations
        # 7. Create actionable recommendations
        # 8. Generate visualization and reporting data
        # 9. Update analysis cache and history
        # 10. Validate analysis accuracy and completeness
        
        try:
            # Validate input
            if len(metrics) < 2:
                raise ValueError("At least 2 metrics required for comparison")
            
            # Extract comparison data
            metric_name = metrics[0].name
            sources = [m.source for m in metrics]
            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]
            confidences = [m.confidence for m in metrics]
            
            # Perform statistical analysis
            differences = self._calculate_differences(values)
            correlation = self._calculate_correlation(values, timestamps)
            significance = self._calculate_significance(values)
            confidence = np.mean(confidences)
            
            # Generate insights
            insights = await self._generate_insights(metrics, differences, correlation, significance)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(metrics, insights)
            
            # Create comparison result
            comparison_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = ComparisonResult(
                comparison_id=comparison_id,
                metric_name=metric_name,
                sources=sources,
                values=values,
                differences=differences,
                correlation=correlation,
                significance=significance,
                confidence=confidence,
                insights=insights,
                recommendations=recommendations
            )
            
            # Update history
            self.comparison_history.append(result)
            
            logger.info(f"Comparison completed: {comparison_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in metric comparison: {e}")
            raise
    
    def _calculate_differences(self, values: List[float]) -> List[float]:
        """
        Calculate differences between metric values
        """
        # AI REASONING: Difference calculation and analysis
        # PSEUDOCODE:
        # 1. Calculate pairwise differences between all values
        # 2. Apply appropriate scaling and normalization
        # 3. Identify significant differences and outliers
        # 4. Calculate relative and absolute differences
        # 5. Generate difference distribution analysis
        
        differences = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff = values[j] - values[i]
                differences.append(diff)
        
        return differences
    
    def _calculate_correlation(self, values: List[float], timestamps: List[datetime]) -> float:
        """
        Calculate correlation between values and time
        """
        # AI REASONING: Temporal correlation analysis
        # PSEUDOCODE:
        # 1. Convert timestamps to numerical values
        # 2. Calculate Pearson correlation coefficient
        # 3. Assess correlation significance and strength
        # 4. Identify temporal patterns and trends
        # 5. Generate correlation confidence intervals
        
        if len(values) < 2:
            return 0.0
        
        # Convert timestamps to numerical values
        time_nums = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(values, time_nums)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_significance(self, values: List[float]) -> float:
        """
        Calculate statistical significance of differences
        """
        # AI REASONING: Statistical significance testing
        # PSEUDOCODE:
        # 1. Perform t-test or ANOVA for multiple groups
        # 2. Calculate p-values and confidence intervals
        # 3. Assess effect size and practical significance
        # 4. Apply multiple comparison corrections if needed
        # 5. Generate significance interpretation
        
        if len(values) < 2:
            return 1.0
        
        try:
            # Simple variance-based significance
            variance = np.var(values)
            mean = np.mean(values)
            
            if variance == 0:
                return 1.0
            
            # Coefficient of variation as significance measure
            cv = np.sqrt(variance) / abs(mean) if mean != 0 else 0
            significance = 1.0 / (1.0 + cv)
            
            return min(significance, 1.0)
        except:
            return 0.5
    
    async def _generate_insights(self, metrics: List[ComparisonMetric], 
                               differences: List[float], correlation: float, 
                               significance: float) -> List[str]:
        """
        Generate insights from comparison analysis
        """
        # AI REASONING: Insight generation and pattern recognition
        # PSEUDOCODE:
        # 1. Analyze value distributions and patterns
        # 2. Identify outliers and anomalies
        # 3. Assess temporal trends and seasonality
        # 4. Compare source reliability and consistency
        # 5. Identify data quality issues and gaps
        # 6. Generate contextual insights and observations
        # 7. Assess comparative performance and rankings
        # 8. Identify potential data source biases
        
        insights = []
        
        # Analyze value ranges
        values = [m.value for m in metrics]
        value_range = max(values) - min(values)
        value_mean = np.mean(values)
        
        if value_range > value_mean * 0.5:
            insights.append("High variability observed across data sources")
        else:
            insights.append("Consistent values across data sources")
        
        # Analyze differences
        if differences:
            max_diff = max(abs(d) for d in differences)
            if max_diff > value_mean * 0.3:
                insights.append("Significant differences detected between sources")
        
        # Analyze correlation
        if abs(correlation) > 0.7:
            insights.append("Strong temporal correlation detected")
        elif abs(correlation) > 0.3:
            insights.append("Moderate temporal correlation detected")
        else:
            insights.append("Weak temporal correlation detected")
        
        # Analyze significance
        if significance > 0.8:
            insights.append("High statistical significance in differences")
        elif significance > 0.5:
            insights.append("Moderate statistical significance in differences")
        else:
            insights.append("Low statistical significance in differences")
        
        # Analyze source confidence
        confidences = [m.confidence for m in metrics]
        avg_confidence = np.mean(confidences)
        if avg_confidence < 0.7:
            insights.append("Low confidence levels across data sources")
        
        return insights
    
    async def _generate_recommendations(self, metrics: List[ComparisonMetric], 
                                      insights: List[str]) -> List[str]:
        """
        Generate recommendations based on analysis
        """
        # AI REASONING: Recommendation generation
        # PSEUDOCODE:
        # 1. Analyze insights and identify actionable items
        # 2. Prioritize recommendations by impact and feasibility
        # 3. Generate data quality improvement suggestions
        # 4. Recommend additional analysis and investigation
        # 5. Suggest source validation and verification
        # 6. Generate monitoring and alerting recommendations
        # 7. Recommend process improvements and optimizations
        
        recommendations = []
        
        # Data quality recommendations
        if "High variability" in str(insights):
            recommendations.append("Investigate source-specific data collection methods")
            recommendations.append("Implement data validation and quality checks")
        
        if "Low confidence" in str(insights):
            recommendations.append("Improve data source reliability and accuracy")
            recommendations.append("Implement confidence scoring and weighting")
        
        if "Significant differences" in str(insights):
            recommendations.append("Perform detailed source comparison analysis")
            recommendations.append("Validate data processing and transformation steps")
        
        # General recommendations
        recommendations.append("Establish regular comparative analysis monitoring")
        recommendations.append("Implement automated anomaly detection")
        
        return recommendations
    
    async def compare_time_series(self, time_series_data: Dict[str, List[Tuple[datetime, float]]], 
                                analysis_window: timedelta = None) -> Dict[str, Any]:
        """
        Compare multiple time series data
        """
        # AI REASONING: Time series comparative analysis
        # PSEUDOCODE:
        # 1. Align time series data to common time grid
        # 2. Apply time window filtering and normalization
        # 3. Calculate temporal correlations and cross-correlations
        # 4. Perform trend analysis and seasonality detection
        # 5. Identify lag relationships and lead-lag patterns
        # 6. Generate time series similarity metrics
        # 7. Perform change point detection and analysis
        # 8. Generate forecasting and prediction insights
        
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "sources": list(time_series_data.keys()),
                "correlations": {},
                "trends": {},
                "anomalies": {},
                "insights": []
            }
            
            # Convert to pandas DataFrames
            dfs = {}
            for source, data in time_series_data.items():
                df = pd.DataFrame(data, columns=['timestamp', 'value'])
                df.set_index('timestamp', inplace=True)
                dfs[source] = df
            
            # Align time series
            aligned_data = self._align_time_series(dfs)
            
            # Calculate correlations
            correlation_matrix = aligned_data.corr()
            results["correlations"] = correlation_matrix.to_dict()
            
            # Analyze trends
            for source in time_series_data.keys():
                if source in aligned_data.columns:
                    trend = self._analyze_trend(aligned_data[source])
                    results["trends"][source] = trend
            
            # Detect anomalies
            for source in time_series_data.keys():
                if source in aligned_data.columns:
                    anomalies = self._detect_anomalies(aligned_data[source])
                    results["anomalies"][source] = anomalies
            
            # Generate insights
            results["insights"] = await self._generate_time_series_insights(
                aligned_data, correlation_matrix, results["trends"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in time series comparison: {e}")
            raise
    
    def _align_time_series(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple time series to common time grid
        """
        # AI REASONING: Time series alignment and synchronization
        # PSEUDOCODE:
        # 1. Identify common time range across all series
        # 2. Resample series to common frequency
        # 3. Handle missing values and gaps
        # 4. Apply interpolation and smoothing
        # 5. Validate alignment quality and consistency
        
        if not dfs:
            return pd.DataFrame()
        
        # Find common time range
        all_timestamps = set()
        for df in dfs.values():
            all_timestamps.update(df.index)
        
        common_timestamps = sorted(all_timestamps)
        
        # Create aligned DataFrame
        aligned_data = pd.DataFrame(index=common_timestamps)
        
        for source, df in dfs.items():
            aligned_data[source] = df.reindex(common_timestamps)
        
        # Forward fill missing values
        aligned_data.fillna(method='ffill', inplace=True)
        
        return aligned_data
    
    def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze trend in time series
        """
        # AI REASONING: Trend analysis and characterization
        # PSEUDOCODE:
        # 1. Calculate linear trend using regression
        # 2. Assess trend significance and strength
        # 3. Identify trend direction and magnitude
        # 4. Calculate trend confidence intervals
        # 5. Generate trend interpretation
        
        if len(series) < 2:
            return {"trend": "insufficient_data", "slope": 0, "significance": 0}
        
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return {"trend": "insufficient_data", "slope": 0, "significance": 0}
        
        # Calculate trend
        x = np.arange(len(clean_series))
        y = clean_series.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if abs(slope) < std_err:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        return {
            "trend": trend_direction,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "significance": 1 - p_value
        }
    
    def _detect_anomalies(self, series: pd.Series) -> List[Dict[str, Any]]:
        """
        Detect anomalies in time series
        """
        # AI REASONING: Anomaly detection and analysis
        # PSEUDOCODE:
        # 1. Calculate statistical measures (mean, std, percentiles)
        # 2. Identify outliers using z-score and IQR methods
        # 3. Detect change points and structural breaks
        # 4. Assess anomaly severity and significance
        # 5. Generate anomaly classification and interpretation
        
        anomalies = []
        
        if len(series) < 10:
            return anomalies
        
        # Remove NaN values
        clean_series = series.dropna()
        
        # Calculate statistics
        mean_val = clean_series.mean()
        std_val = clean_series.std()
        
        # Z-score based anomaly detection
        z_scores = np.abs((clean_series - mean_val) / std_val)
        anomaly_indices = z_scores > 2.5  # 2.5 standard deviations
        
        for idx in clean_series[anomaly_indices].index:
            anomaly = {
                "timestamp": idx.isoformat(),
                "value": clean_series[idx],
                "z_score": z_scores[idx],
                "severity": "high" if z_scores[idx] > 3 else "medium"
            }
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _generate_time_series_insights(self, aligned_data: pd.DataFrame, 
                                           correlation_matrix: pd.DataFrame,
                                           trends: Dict[str, Any]) -> List[str]:
        """
        Generate insights from time series comparison
        """
        # AI REASONING: Time series insight generation
        # PSEUDOCODE:
        # 1. Analyze correlation patterns and relationships
        # 2. Compare trend directions and magnitudes
        # 3. Identify synchronization and lag patterns
        # 4. Assess data quality and consistency
        # 5. Generate comparative performance insights
        # 6. Identify potential causal relationships
        
        insights = []
        
        # Correlation insights
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_correlations.append((correlation_matrix.columns[i], 
                                           correlation_matrix.columns[j], corr))
        
        if high_correlations:
            insights.append(f"Strong correlations detected between {len(high_correlations)} source pairs")
        
        # Trend insights
        trend_directions = [trend["trend"] for trend in trends.values()]
        if len(set(trend_directions)) == 1:
            insights.append(f"Consistent {trend_directions[0]} trend across all sources")
        else:
            insights.append("Mixed trend directions across sources")
        
        # Data quality insights
        missing_data = aligned_data.isnull().sum()
        if missing_data.sum() > 0:
            insights.append(f"Missing data detected in {missing_data.sum()} observations")
        
        return insights
    
    async def perform_benchmark_analysis(self, metrics: List[ComparisonMetric], 
                                       benchmark_value: float) -> Dict[str, Any]:
        """
        Perform benchmark comparison analysis
        """
        # AI REASONING: Benchmark analysis and performance assessment
        # PSEUDOCODE:
        # 1. Calculate performance relative to benchmark
        # 2. Assess benchmark achievement and gaps
        # 3. Identify over/under performance patterns
        # 4. Calculate performance rankings and percentiles
        # 5. Generate improvement recommendations
        # 6. Assess benchmark relevance and validity
        # 7. Generate performance trend analysis
        
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "benchmark_value": benchmark_value,
                "performance_analysis": {},
                "rankings": [],
                "insights": [],
                "recommendations": []
            }
            
            # Calculate performance metrics
            values = [m.value for m in metrics]
            sources = [m.source for m in metrics]
            
            for i, (value, source) in enumerate(zip(values, sources)):
                performance = {
                    "source": source,
                    "value": value,
                    "difference": value - benchmark_value,
                    "percentage_diff": ((value - benchmark_value) / benchmark_value) * 100,
                    "achievement": value >= benchmark_value,
                    "rank": 0
                }
                results["performance_analysis"][source] = performance
            
            # Calculate rankings
            sorted_sources = sorted(values, reverse=True)
            for source, value in zip(sources, values):
                rank = sorted_sources.index(value) + 1
                results["performance_analysis"][source]["rank"] = rank
                results["rankings"].append({
                    "source": source,
                    "rank": rank,
                    "value": value
                })
            
            # Generate insights
            above_benchmark = sum(1 for v in values if v >= benchmark_value)
            results["insights"].append(f"{above_benchmark}/{len(values)} sources meet benchmark")
            
            best_performer = min(results["rankings"], key=lambda x: x["rank"])
            results["insights"].append(f"Best performer: {best_performer['source']} (rank {best_performer['rank']})")
            
            # Generate recommendations
            if above_benchmark < len(values) / 2:
                results["recommendations"].append("Focus on improving underperforming sources")
            
            results["recommendations"].append("Establish regular benchmark monitoring")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in benchmark analysis: {e}")
            raise
    
    async def generate_comparison_report(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report
        """
        # AI REASONING: Report generation and synthesis
        # PSEUDOCODE:
        # 1. Aggregate comparison results and statistics
        # 2. Identify patterns and trends across comparisons
        # 3. Generate executive summary and key findings
        # 4. Create detailed analysis sections
        # 5. Generate visualizations and charts
        # 6. Provide actionable recommendations
        # 7. Include methodology and data quality assessment
        
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_comparisons": len(comparison_results),
                "summary": {},
                "detailed_analysis": {},
                "recommendations": [],
                "methodology": {}
            }
            
            # Generate summary statistics
            correlations = [r.correlation for r in comparison_results]
            significances = [r.significance for r in comparison_results]
            confidences = [r.confidence for r in comparison_results]
            
            report["summary"] = {
                "avg_correlation": np.mean(correlations),
                "avg_significance": np.mean(significances),
                "avg_confidence": np.mean(confidences),
                "high_correlation_count": sum(1 for c in correlations if abs(c) > 0.7),
                "high_significance_count": sum(1 for s in significances if s > 0.8)
            }
            
            # Aggregate insights
            all_insights = []
            for result in comparison_results:
                all_insights.extend(result.insights)
            
            # Count insight frequencies
            insight_counts = {}
            for insight in all_insights:
                insight_counts[insight] = insight_counts.get(insight, 0) + 1
            
            report["detailed_analysis"]["common_insights"] = insight_counts
            
            # Aggregate recommendations
            all_recommendations = []
            for result in comparison_results:
                all_recommendations.extend(result.recommendations)
            
            report["recommendations"] = list(set(all_recommendations))
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} cleanup completed")

# Example usage
async def main():
    config = {
        "analysis_cache_size": 1000,
        "confidence_threshold": 0.7,
        "correlation_threshold": 0.5
    }
    
    agent = ComparativeAnalysisAgent(config)
    await agent.initialize()
    
    try:
        # Create sample metrics for comparison
        metrics = [
            ComparisonMetric("revenue", 1000000, "source_a", datetime.now(), 0.9),
            ComparisonMetric("revenue", 950000, "source_b", datetime.now(), 0.8),
            ComparisonMetric("revenue", 1050000, "source_c", datetime.now(), 0.85)
        ]
        
        # Perform comparison
        result = await agent.compare_metrics(metrics)
        print(f"Comparison completed: {result.comparison_id}")
        print(f"Correlation: {result.correlation:.3f}")
        print(f"Significance: {result.significance:.3f}")
        print(f"Insights: {len(result.insights)}")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 