import React, { useEffect, useState } from 'react';
import { fetchAgentStatus, AgentStatus } from './api';

// AgentStatusPanel: Shows agent health and activity
// Future: Fetch agent status from backend, render status indicators

const AgentStatusPanel: React.FC = () => {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAgentStatus().then(data => {
      setAgents(data);
      setLoading(false);
    });
  }, []);

  if (loading) return <div>Loading agent status...</div>;

  return (
    <div>
      <h2>Agent Status</h2>
      <ul>
        {agents.map(agent => (
          <li key={agent.agent_name}>
            <strong>{agent.agent_name}</strong>: {agent.status} (last run: {agent.last_run})
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AgentStatusPanel; 