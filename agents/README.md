# Agents

This folder contains all agent microservices. Each agent runs as a separate Docker service and is responsible for a specific data ingestion, analysis, or reasoning task.

## Agent List
- equity_research_agent: Ingests and parses equity research reports
- sec_filings_agent: Parses SEC filings and extracts financial statements
- market_news_agent: Scrapes and analyzes market news
- insider_trading_agent: Monitors insider trading activity
- social_media_nlp_agent: Analyzes social media for sentiment and claims
- fundamental_pricing_agent: Calculates intrinsic value and pricing models
- kpi_tracker_agent: Tracks KPIs by ticker/industry
- event_impact_agent: Analyzes event-driven price impacts
- data_tagging_agent: Tags and indexes all data/events
- revenue_geography_agent: Maps company sales by region
- macro_calendar_agent: Tags macro calendar events
- options_flow_agent: Detects options flow and volatility events
- ml_model_testing_agent: Tests and deploys ML models

Each agent subfolder will contain a README and placeholder code for future implementation. 