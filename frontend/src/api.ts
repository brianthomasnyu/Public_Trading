// ========================================
// API SERVICE - UPDATED FOR NEW BACKEND
// ========================================

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ========================================
// UPDATED TYPES TO MATCH BACKEND DATABASE
// ========================================

export interface TimelineEvent {
  id: number;
  event_time: string;  // ISO timestamp from backend
  source_agent: string;
  event_type: string;
  ticker?: string;
  data: any;  // JSONB data from backend
  tags: string[];
  summary?: string;  // Derived from data or added by frontend
  created_at?: string;
}

export interface AgentStatus {
  id?: number;
  agent_name: string;
  last_run?: string;  // ISO timestamp from backend
  status: string;
  details?: any;  // JSONB details from backend
}

// ========================================
// UPDATED API FUNCTIONS FOR NEW BACKEND
// ========================================

// Fetch timeline events from the orchestrator
export async function fetchTimeline(): Promise<TimelineEvent[]> {
  try {
    const res = await fetch(`${API_URL}/timeline`);
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    const data = await res.json();
    
    // Transform backend data to include summary for frontend display
    const eventsWithSummary = data.events.map((event: TimelineEvent) => ({
      ...event,
      summary: event.summary || generateEventSummary(event)
    }));
    
    return eventsWithSummary;
  } catch (e) {
    console.error('Error fetching timeline:', e);
    // MOCK DATA FALLBACK - TO BE REMOVED WHEN BACKEND IS FULLY READY
    console.warn('Using mock timeline data - backend may not be ready');
    return [
      { 
        id: 1, 
        event_time: new Date().toISOString(), 
        source_agent: 'market_news_agent', 
        event_type: 'news', 
        ticker: 'AAPL',
        data: { headline: 'Apple reports strong Q4 earnings', sentiment: 'positive' },
        tags: ['earnings', 'positive'], 
        summary: 'Apple reports strong Q4 earnings beating estimates',
        created_at: new Date().toISOString()
      },
      { 
        id: 2, 
        event_time: new Date(Date.now() - 3600000).toISOString(), 
        source_agent: 'sec_filings_agent', 
        event_type: 'sec_filing', 
        ticker: 'TSLA',
        data: { filing_type: '10-K', document_url: 'https://sec.gov/...' },
        tags: ['10-K', 'annual'], 
        summary: 'Tesla files annual 10-K report',
        created_at: new Date(Date.now() - 3600000).toISOString()
      },
    ];
  }
}

// Fetch agent status from the orchestrator
export async function fetchAgentStatus(): Promise<AgentStatus[]> {
  try {
    const res = await fetch(`${API_URL}/agents/status`);
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    const data = await res.json();
    return data.agents;
  } catch (e) {
    console.error('Error fetching agent status:', e);
    // MOCK DATA FALLBACK - TO BE REMOVED WHEN BACKEND IS FULLY READY
    console.warn('Using mock agent status data - backend may not be ready');
    return [
      { agent_name: 'market_news_agent', status: 'online', last_run: new Date(Date.now() - 120000).toISOString() },
      { agent_name: 'sec_filings_agent', status: 'online', last_run: new Date(Date.now() - 300000).toISOString() },
      { agent_name: 'fundamental_pricing_agent', status: 'offline', last_run: new Date(Date.now() - 3600000).toISOString() },
      { agent_name: 'options_flow_agent', status: 'online', last_run: new Date(Date.now() - 30000).toISOString() },
      { agent_name: 'social_media_nlp_agent', status: 'online', last_run: new Date(Date.now() - 60000).toISOString() },
      { agent_name: 'insider_trading_agent', status: 'offline', last_run: new Date(Date.now() - 10800000).toISOString() },
    ];
  }
}

// Send user query to the orchestrator
export async function sendQuery(query: string, ticker?: string, event?: string): Promise<any> {
  try {
    const res = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ticker, event })
    });
    
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    return await res.json();
  } catch (e) {
    console.error('Error sending query:', e);
    // MOCK RESPONSE FALLBACK - TO BE REMOVED WHEN BACKEND IS FULLY READY
    console.warn('Using mock query response - backend may not be ready');
    return { 
      message: `Processed query: "${query}". Backend integration in progress.`,
      query,
      ticker,
      event,
      timestamp: new Date().toISOString()
    };
  }
}

// ========================================
// NEW API FUNCTIONS FOR ENHANCED FEATURES
// ========================================

// Send MCP message (for future agent communication)
export async function sendMCPMessage(sender: string, recipient: string, content: any, context?: any): Promise<any> {
  try {
    const res = await fetch(`${API_URL}/mcp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sender, recipient, content, context })
    });
    
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    return await res.json();
  } catch (e) {
    console.error('Error sending MCP message:', e);
    return { error: 'Failed to send MCP message' };
  }
}

// ========================================
// UTILITY FUNCTIONS
// ========================================

// Generate a summary from event data for display
function generateEventSummary(event: TimelineEvent): string {
  switch (event.event_type) {
    case 'news':
      return event.data?.headline || `News update for ${event.ticker || 'market'}`;
    case 'sec_filing':
      return `${event.data?.filing_type || 'SEC'} filing for ${event.ticker}`;
    case 'earnings':
      return `Earnings report for ${event.ticker}`;
    case 'options_flow':
      return `Unusual options activity detected for ${event.ticker}`;
    case 'sentiment':
      return `Social sentiment update for ${event.ticker}`;
    default:
      return `${event.event_type} event for ${event.ticker || 'market'}`;
  }
}

// Format timestamp for display
export function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins} minutes ago`;
  if (diffHours < 24) return `${diffHours} hours ago`;
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString();
}

// ========================================
// HEALTH CHECK FUNCTION
// ========================================

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch (e) {
    console.error('Backend health check failed:', e);
    return false;
  }
} 