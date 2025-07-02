// ========================================
// MOCK DATA SERVICE - TO BE REPLACED
// ========================================
// This entire file contains mock/placeholder data
// and will be replaced with real API calls in the future

export interface TradingStats {
  totalPnL: number;
  totalPnLPercentage: number;
  winRate: number;
  winRateChange: number;
  averageRR: number;
  averageRRChange: number;
  avgHoldTime: string;
  avgHoldTimeChange?: number;
}

export interface TradingPosition {
  symbol: string;
  side: 'Long' | 'Short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

export interface ChartDataPoint {
  time: string;
  value: number;
}

export interface SystemStatus {
  systemOnline: boolean;
  activeAgents: number;
  marketStatus: 'Open' | 'Closed' | 'Pre-Market' | 'After-Hours';
}

// ========================================
// ENHANCED AGENT DATA STRUCTURES
// ========================================

export interface AgentDataFeed {
  id: string;
  source: string;
  symbol?: string;
  lastUpdate: string;
  dataType: string;
  value: string | number;
  trend?: 'up' | 'down' | 'neutral';
  confidence?: number;
}

export interface DetailedAgentStatus {
  agent_name: string;
  display_name: string;
  status: 'active' | 'idle' | 'error' | 'stopped';
  last_run: string;
  next_run?: string;
  description: string;
  category: 'market_data' | 'news' | 'analysis' | 'compliance' | 'social';
  data_feeds: AgentDataFeed[];
  metrics: {
    uptime: number;
    success_rate: number;
    avg_processing_time: number;
    data_points_processed: number;
  };
  icon: string;
  color: string;
}

// ========================================
// MOCK DATA GENERATORS - TO BE REPLACED
// ========================================

export const getMockTradingStats = (): TradingStats => {
  // TODO: Replace with real API call to get actual trading statistics
  return {
    totalPnL: 125.13,
    totalPnLPercentage: 11.5,
    winRate: 79,
    winRateChange: 5.2,
    averageRR: 2.97,
    averageRRChange: 8.3,
    avgHoldTime: "0d 23h 22m",
    avgHoldTimeChange: 0,
  };
};

export const getMockPnLData = (): ChartDataPoint[] => {
  // TODO: Replace with real API call to get historical P&L data
  return Array.from({ length: 30 }, (_, i) => ({
    time: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toLocaleDateString(),
    value: Math.random() * 5000 + 2000 + i * 100
  }));
};

export const getMockTradingPositions = (): TradingPosition[] => {
  // TODO: Replace with real API call to get current trading positions
  return [
    { 
      symbol: 'AAPL', 
      side: 'Long', 
      quantity: 100, 
      entryPrice: 150.25, 
      currentPrice: 155.30, 
      pnl: 505, 
      pnlPercent: 3.36 
    },
    { 
      symbol: 'TSLA', 
      side: 'Short', 
      quantity: 50, 
      entryPrice: 240.80, 
      currentPrice: 235.20, 
      pnl: 280, 
      pnlPercent: 2.33 
    },
    { 
      symbol: 'NVDA', 
      side: 'Long', 
      quantity: 75, 
      entryPrice: 420.15, 
      currentPrice: 415.80, 
      pnl: -326.25, 
      pnlPercent: -1.04 
    },
    { 
      symbol: 'MSFT', 
      side: 'Long', 
      quantity: 80, 
      entryPrice: 330.50, 
      currentPrice: 335.75, 
      pnl: 420, 
      pnlPercent: 1.59 
    },
  ];
};

export const getMockSystemStatus = (): SystemStatus => {
  // TODO: Replace with real API call to get system status
  return {
    systemOnline: true,
    activeAgents: 13,
    marketStatus: 'Open',
  };
};

export const getMockDetailedAgentStatus = (): DetailedAgentStatus[] => {
  // TODO: Replace with real API call to get detailed agent status and data feeds
  return [
    {
      agent_name: 'market_news_agent',
      display_name: 'Market News Agent',
      status: 'active',
      last_run: '2 minutes ago',
      next_run: '3 minutes',
      description: 'Monitors financial news and market sentiment',
      category: 'news',
      data_feeds: [
        { 
          id: '1', 
          source: 'Reuters', 
          symbol: 'AAPL', 
          lastUpdate: '1m ago', 
          dataType: 'news', 
          value: 'Positive earnings outlook', 
          trend: 'up',
          confidence: 87
        },
        { 
          id: '2', 
          source: 'Bloomberg', 
          symbol: 'TSLA', 
          lastUpdate: '3m ago', 
          dataType: 'news', 
          value: 'Production milestone reached', 
          trend: 'up',
          confidence: 92
        }
      ],
      metrics: {
        uptime: 99.8,
        success_rate: 96.4,
        avg_processing_time: 1.2,
        data_points_processed: 1247
      },
      icon: '📰',
      color: '#0088FF'
    },
    {
      agent_name: 'sec_filings_agent',
      display_name: 'SEC Filings Agent',
      status: 'active',
      last_run: '5 minutes ago',
      next_run: '10 minutes',
      description: 'Tracks SEC filings and regulatory updates',
      category: 'compliance',
      data_feeds: [
        { 
          id: '3', 
          source: 'SEC EDGAR', 
          symbol: 'MSFT', 
          lastUpdate: '15m ago', 
          dataType: '10-Q', 
          value: 'Filed quarterly report', 
          trend: 'neutral',
          confidence: 100
        }
      ],
      metrics: {
        uptime: 99.9,
        success_rate: 98.7,
        avg_processing_time: 3.4,
        data_points_processed: 342
      },
      icon: '📋',
      color: '#00D4AA'
    },
    {
      agent_name: 'options_flow_agent',
      display_name: 'Options Flow Agent',
      status: 'active',
      last_run: '30 seconds ago',
      next_run: '1 minute',
      description: 'Analyzes unusual options activity and flow',
      category: 'market_data',
      data_feeds: [
        { 
          id: '4', 
          source: 'CBOE', 
          symbol: 'SPY', 
          lastUpdate: '30s ago', 
          dataType: 'options_flow', 
          value: 'Large call volume', 
          trend: 'up',
          confidence: 78
        },
        { 
          id: '5', 
          source: 'Options Chain', 
          symbol: 'QQQ', 
          lastUpdate: '1m ago', 
          dataType: 'put_call_ratio', 
          value: 0.67, 
          trend: 'down',
          confidence: 85
        }
      ],
      metrics: {
        uptime: 98.5,
        success_rate: 94.2,
        avg_processing_time: 0.8,
        data_points_processed: 5678
      },
      icon: '📊',
      color: '#FFA726'
    },
    {
      agent_name: 'social_media_nlp_agent',
      display_name: 'Social Sentiment Agent',
      status: 'active',
      last_run: '1 minute ago',
      next_run: '2 minutes',
      description: 'Analyzes social media sentiment and trends',
      category: 'social',
      data_feeds: [
        { 
          id: '6', 
          source: 'Twitter API', 
          symbol: 'TSLA', 
          lastUpdate: '45s ago', 
          dataType: 'sentiment', 
          value: 'Bullish: 72%', 
          trend: 'up',
          confidence: 81
        },
        { 
          id: '7', 
          source: 'Reddit WSB', 
          symbol: 'GME', 
          lastUpdate: '2m ago', 
          dataType: 'mentions', 
          value: '2.3k mentions', 
          trend: 'up',
          confidence: 76
        }
      ],
      metrics: {
        uptime: 97.8,
        success_rate: 91.3,
        avg_processing_time: 2.1,
        data_points_processed: 8945
      },
      icon: '💬',
      color: '#9C27B0'
    },
    {
      agent_name: 'fundamental_pricing_agent',
      display_name: 'Fundamental Analysis Agent',
      status: 'idle',
      last_run: '1 hour ago',
      next_run: '2 hours',
      description: 'Calculates fair value using fundamental metrics',
      category: 'analysis',
      data_feeds: [
        { 
          id: '8', 
          source: 'Financial Statements', 
          symbol: 'AAPL', 
          lastUpdate: '1h ago', 
          dataType: 'fair_value', 
          value: '$162.45', 
          trend: 'up',
          confidence: 88
        }
      ],
      metrics: {
        uptime: 99.2,
        success_rate: 97.1,
        avg_processing_time: 15.3,
        data_points_processed: 156
      },
      icon: '🔍',
      color: '#FF4757'
    },
    {
      agent_name: 'insider_trading_agent',
      display_name: 'Insider Trading Monitor',
      status: 'error',
      last_run: '3 hours ago',
      next_run: 'Retrying...',
      description: 'Monitors insider trading activity and Form 4 filings',
      category: 'compliance',
      data_feeds: [],
      metrics: {
        uptime: 85.3,
        success_rate: 92.8,
        avg_processing_time: 4.7,
        data_points_processed: 89
      },
      icon: '⚠️',
      color: '#FF4757'
    }
  ];
};