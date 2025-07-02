const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Types
export interface TimelineEvent {
  id: number;
  event_time: string;
  source_agent: string;
  event_type: string;
  ticker?: string;
  tags: string[];
  summary: string;
}

export interface AgentStatus {
  agent_name: string;
  status: string;
  last_run: string;
}

// Fetch timeline events
export async function fetchTimeline(): Promise<TimelineEvent[]> {
  try {
    const res = await fetch(`${API_URL}/timeline`);
    const data = await res.json();
    return data.events;
  } catch (e) {
    // Return mock data on error
    return [
      { id: 0, event_time: '', source_agent: '', event_type: '', ticker: '', tags: [], summary: 'Error fetching timeline.' }
    ];
  }
}

// Fetch agent status
export async function fetchAgentStatus(): Promise<AgentStatus[]> {
  try {
    const res = await fetch(`${API_URL}/agents/status`);
    const data = await res.json();
    return data.agents;
  } catch (e) {
    // Return mock data on error
    return [
      { agent_name: 'unknown', status: 'error', last_run: '' }
    ];
  }
}

// Send user query
export async function sendQuery(query: string, ticker?: string, event?: string): Promise<any> {
  try {
    const res = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ticker, event })
    });
    return await res.json();
  } catch (e) {
    return { message: 'Error sending query.' };
  }
}

// Future: Add more API utilities for event details, document upload, etc. 