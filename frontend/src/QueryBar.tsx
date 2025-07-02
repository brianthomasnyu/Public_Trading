import React, { useState } from 'react';
import { sendQuery } from './api';

// QueryBar: Input for user queries (natural language, ticker, event)
// Future: Add state, input handling, and API call to backend

const QueryBar: React.FC = () => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<string | null>(null);

  const handleSubmit = async () => {
    setLoading(true);
    setResponse(null);
    const res = await sendQuery(input);
    setResponse(res.message || 'No response');
    setLoading(false);
  };

  return (
    <div>
      <input
        type="text"
        placeholder="Enter your query or ticker..."
        value={input}
        onChange={e => setInput(e.target.value)}
        disabled={loading}
      />
      <button onClick={handleSubmit} disabled={loading || !input}>
        {loading ? 'Submitting...' : 'Submit'}
      </button>
      {response && <div>Response: {response}</div>}
    </div>
  );
};

export default QueryBar; 