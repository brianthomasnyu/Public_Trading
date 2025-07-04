// ========================================
// FRONTEND CONFIGURATION - UPDATED FOR NEW BACKEND
// ========================================

export const config = {
  // API Configuration
  API_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',
  
  // Environment Settings
  ENV: process.env.REACT_APP_ENV || 'development',
  DEBUG: process.env.REACT_APP_DEBUG === 'true',
  
  // Mock data toggle - set to false when backend is fully ready
  USE_MOCK_DATA: process.env.REACT_APP_USE_MOCK_DATA === 'true',
  
  // Backend API endpoints (matching orchestrator routes)
  API_ENDPOINTS: {
    HEALTH: '/health',
    TIMELINE: '/timeline',
    AGENTS_STATUS: '/agents/status',
    QUERY: '/query',
    MCP: '/mcp',
    // Future endpoints to implement:
    TRADING_STATS: '/trading/stats',
    PNL_DATA: '/trading/pnl',
    POSITIONS: '/trading/positions',
    SYSTEM_STATUS: '/system/status',
  },

  // Polling intervals (in milliseconds)
  REFRESH_INTERVALS: {
    TIMELINE: parseInt(process.env.REACT_APP_TIMELINE_REFRESH_INTERVAL || '30000'),
    AGENT_STATUS: parseInt(process.env.REACT_APP_AGENT_STATUS_REFRESH_INTERVAL || '30000'),
    TRADING_STATS: parseInt(process.env.REACT_APP_TRADING_STATS_REFRESH_INTERVAL || '30000'),
    PNL_DATA: 60000,      // 1 minute
    POSITIONS: 15000,     // 15 seconds
    SYSTEM_STATUS: 10000, // 10 seconds
  },

  // Agent categories (matching backend agent structure)
  AGENT_CATEGORIES: {
    MARKET_DATA: 'market_data',
    NEWS: 'news',
    ANALYSIS: 'analysis',
    COMPLIANCE: 'compliance',
    SOCIAL: 'social',
  },

  // Event types (matching backend database structure)
  EVENT_TYPES: {
    NEWS: 'news',
    SEC_FILING: 'sec_filing',
    EARNINGS: 'earnings',
    OPTIONS_FLOW: 'options_flow',
    SENTIMENT: 'sentiment',
    KPI: 'kpi',
    INSIDER_TRADING: 'insider_trading',
    FUNDAMENTAL_ANALYSIS: 'fundamental_analysis',
  },

  // Known agents from the backend infrastructure
  KNOWN_AGENTS: [
    'equity_research_agent',
    'sec_filings_agent',
    'market_news_agent',
    'insider_trading_agent',
    'social_media_nlp_agent',
    'fundamental_pricing_agent',
    'kpi_tracker_agent',
    'event_impact_agent',
    'data_tagging_agent',
    'revenue_geography_agent',
    'macro_calendar_agent',
    'options_flow_agent',
    'ml_model_testing_agent',
  ],

  // Animation and UI settings
  ANIMATIONS: {
    ENABLED: true,
    DURATION_FAST: 150,
    DURATION_NORMAL: 300,
    DURATION_SLOW: 500,
  },

  // Error retry configuration
  ERROR_HANDLING: {
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
    TIMEOUT: 10000,
  },
};