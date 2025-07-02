# Short Interest Agent

## Overview

The Short Interest Agent is an intelligent system designed to monitor and analyze short interest data, detect potential short squeezes, and provide insights into market sentiment and risk factors.

## CRITICAL SYSTEM POLICY

**NO TRADING DECISIONS OR RECOMMENDATIONS**

This agent only performs:
- Short interest data collection and analysis
- Short squeeze probability assessment
- Market sentiment analysis
- Risk factor identification
- Alert generation and monitoring

**NO trading advice, recommendations, or decisions are provided.**

## AI Reasoning Capabilities

### Comprehensive Short Interest Monitoring
- **Multi-Source Data Collection**: Fetches short interest data from multiple sources
- **Real-Time Updates**: Monitors short interest changes with intelligent update frequency
- **Data Validation**: Validates data quality and cross-references sources
- **Trend Analysis**: Identifies short interest trends and pattern changes
- **Historical Analysis**: Tracks historical short interest patterns

### Short Squeeze Analysis
- **Squeeze Probability Calculation**: Calculates probability of short squeezes
- **Risk Factor Assessment**: Identifies factors contributing to squeeze potential
- **Catalyst Detection**: Monitors potential catalysts for squeezes
- **Volume Analysis**: Analyzes volume patterns in relation to short interest
- **Price Impact Modeling**: Models potential price impacts of squeezes

### Market Sentiment Analysis
- **Sentiment Scoring**: Calculates market sentiment based on short interest
- **Contrarian Analysis**: Identifies contrarian opportunities
- **Crowd Psychology**: Analyzes crowd behavior patterns
- **Fear/Greed Indicators**: Tracks fear and greed in short interest data

### Risk Management
- **Risk Assessment**: Assesses risk levels for different stocks
- **Alert Generation**: Generates alerts for significant short interest changes
- **Threshold Monitoring**: Monitors various risk thresholds
- **Correlation Analysis**: Analyzes correlations with other market factors

## Key Features

### Short Interest Data
- **Short Interest Ratio**: Calculates short interest as percentage of float
- **Days to Cover**: Calculates days needed to cover short positions
- **Change Analysis**: Tracks changes in short interest over time
- **Borrow Fee Data**: Monitors cost to borrow shares

### Short Squeeze Detection
- **Squeeze Probability**: Calculates probability of short squeezes
- **Risk Factors**: Identifies factors contributing to squeeze potential
- **Catalysts**: Monitors potential catalysts for squeezes
- **Volume Analysis**: Analyzes volume patterns

### Alert System
- **High Short Interest Alerts**: Alerts for stocks with high short interest
- **Squeeze Risk Alerts**: Alerts for potential short squeezes
- **Borrow Fee Alerts**: Alerts for high borrow fees
- **Volume Spike Alerts**: Alerts for unusual volume activity

## Configuration

```python
config = {
    "short_interest_update_interval": 3600,  # 1 hour
    "alert_threshold": 0.2,
    "max_tickers_per_cycle": 100,
    "squeeze_threshold": 0.5,
    "high_ratio_threshold": 0.3
}
```

## Usage Examples

### Short Interest Data
```python
# Get short interest data
short_interest = await agent.fetch_short_interest_data("GME")
print(f"Short Interest: {short_interest.short_interest:,}")
print(f"Short Interest Ratio: {short_interest.short_interest_ratio:.2%}")
print(f"Days to Cover: {short_interest.days_to_cover:.1f}")
```

### Short Squeeze Analysis
```python
# Analyze short squeeze potential
squeeze_analysis = await agent.analyze_short_squeeze_potential("GME")
print(f"Squeeze Probability: {squeeze_analysis.squeeze_probability:.2%}")
print(f"Risk Factors: {squeeze_analysis.risk_factors}")
print(f"Potential Catalysts: {squeeze_analysis.potential_catalysts}")
```

### High Short Interest Stocks
```python
# Get stocks with high short interest
high_ratio_stocks = await agent.get_high_short_interest_ratio_stocks(0.3)
for stock in high_ratio_stocks:
    print(f"{stock.ticker}: {stock.short_interest_ratio:.2%}")
```

## Integration

### MCP Communication
- **Short Interest Alerts**: Sends alerts to orchestrator and other agents
- **Squeeze Notifications**: Notifies agents of potential squeezes
- **Data Sharing**: Shares short interest data with relevant agents
- **Risk Coordination**: Coordinates with risk management agents

### Agent Coordination
- **Event Impact Agent**: Triggers for significant short interest events
- **Options Flow Agent**: Coordinates for options activity analysis
- **Market News Agent**: Shares short interest-related news
- **Social Media Agent**: Monitors social media sentiment

### Knowledge Base Integration
- **Short Interest Storage**: Stores short interest data and analysis
- **Squeeze Tracking**: Tracks historical squeeze events
- **Pattern Analysis**: Analyzes short interest patterns
- **Alert History**: Maintains alert history and effectiveness

## Error Handling

### Robust Data Collection
- **Source Failures**: Handles API failures with fallback sources
- **Data Validation**: Validates data quality and consistency
- **Rate Limiting**: Respects API rate limits and quotas
- **Timeout Handling**: Handles network timeouts gracefully

### Health Monitoring
- **Data Collection Health**: Monitors short interest data collection success
- **Analysis Health**: Tracks analysis quality and accuracy
- **Alert Health**: Monitors alert generation and delivery
- **Performance Metrics**: Tracks system performance and optimization

## Security Considerations

### Data Privacy
- **API Key Security**: Securely manages data source API keys
- **Data Encryption**: Encrypts sensitive short interest data
- **Access Control**: Implements access controls for data
- **Audit Logging**: Maintains audit logs for all operations

### Compliance
- **No Trading Policy**: Strictly enforces no trading decisions policy
- **Data Protection**: Ensures compliance with data protection regulations
- **Information Security**: Implements information security best practices
- **Audit Compliance**: Maintains compliance with audit requirements

## Development Workflow

### Adding New Data Sources
1. **Source Integration**: Integrate new short interest data sources
2. **Data Validation**: Implement data validation for new sources
3. **Testing**: Test data collection and analysis
4. **Documentation**: Update documentation for new sources

### Customizing Analysis
1. **Algorithm Tuning**: Customize squeeze probability algorithms
2. **Threshold Adjustment**: Adjust alert thresholds
3. **Risk Factor Weighting**: Update risk factor weights
4. **Alert Customization**: Customize alert generation

## Monitoring and Analytics

### Short Interest Metrics
- **Data Collection Success Rate**: Success rate of data collection
- **Analysis Accuracy**: Accuracy of squeeze analysis
- **Alert Effectiveness**: Effectiveness of generated alerts
- **Response Time**: Time to detect and analyze changes

### Performance Monitoring
- **Data Collection Speed**: Speed of short interest data collection
- **Analysis Efficiency**: Efficiency of squeeze analysis
- **Alert Generation**: Speed and accuracy of alert generation
- **System Throughput**: Overall system throughput and performance

## Future Enhancements

### Advanced AI Capabilities
- **Predictive Analysis**: Predict short squeeze probabilities
- **Pattern Recognition**: Advanced pattern recognition in short interest data
- **Machine Learning**: ML-based squeeze prediction and analysis
- **Natural Language Processing**: NLP for news sentiment analysis

### Enhanced Integration
- **Real-Time Streaming**: Real-time short interest data streaming
- **Advanced Analytics**: More sophisticated analytics and modeling
- **Predictive Coordination**: Predict optimal agent coordination
- **Automated Reporting**: Automated short interest reporting 