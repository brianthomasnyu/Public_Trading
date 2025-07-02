-- Table for all ingested events/data
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_time TIMESTAMP NOT NULL,
    source_agent VARCHAR(64) NOT NULL, -- Which agent ingested this
    event_type VARCHAR(64) NOT NULL,   -- e.g., 'sec_filing', 'news', 'kpi', etc.
    ticker VARCHAR(16),                -- Stock ticker if applicable
    data JSONB NOT NULL,               -- Raw or parsed data
    tags TEXT[],                       -- Tags for search/query
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for agent metadata (status, last run, etc.)
CREATE TABLE IF NOT EXISTS agent_status (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(64) UNIQUE NOT NULL,
    last_run TIMESTAMP,
    status VARCHAR(32),
    details JSONB
);

-- Table for timeline index (for fast event timeline queries)
CREATE INDEX IF NOT EXISTS idx_events_event_time ON events(event_time);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON events(ticker);
CREATE INDEX IF NOT EXISTS idx_events_tags ON events USING GIN(tags); 