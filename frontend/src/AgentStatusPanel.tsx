import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Activity, AlertCircle, Clock, CheckCircle, XCircle } from 'lucide-react';
import { fetchAgentStatus, AgentStatus, formatTimestamp, checkBackendHealth } from './api';
import { theme } from './theme';

// ========================================
// ENHANCED AGENT STATUS PANEL
// ========================================
// Updated to work with new backend infrastructure

const Container = styled(motion.div)`
  background: ${theme.colors.cardBackground};
  backdrop-filter: blur(10px);
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.lg};
  transition: all ${theme.animations.normal};

  &:hover {
    border-color: ${theme.colors.primary};
    box-shadow: ${theme.shadows.glow};
  }
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${theme.spacing.lg};
`;

const Title = styled.h2`
  font-size: ${theme.typography.fontSizes.lg};
  font-weight: ${theme.typography.fontWeights.semibold};
  color: ${theme.colors.text};
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
`;

const BackendStatus = styled.div<{ online: boolean }>`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  border-radius: ${theme.borderRadius.md};
  background: ${props => props.online ? `${theme.colors.success}22` : `${theme.colors.danger}22`};
  color: ${props => props.online ? theme.colors.success : theme.colors.danger};
  font-size: ${theme.typography.fontSizes.sm};
`;

const AgentList = styled.div`
  display: grid;
  gap: ${theme.spacing.md};
`;

const AgentItem = styled(motion.div)<{ status: string }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${theme.spacing.md};
  background: ${theme.colors.surface};
  border-radius: ${theme.borderRadius.md};
  border-left: 4px solid ${props => 
    props.status === 'online' ? theme.colors.success :
    props.status === 'offline' ? theme.colors.danger :
    theme.colors.warning
  };
  transition: all ${theme.animations.fast};

  &:hover {
    background: ${theme.colors.surfaceLight};
    transform: translateX(4px);
  }
`;

const AgentInfo = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.md};
`;

const AgentIcon = styled.div<{ status: string }>`
  color: ${props => 
    props.status === 'online' ? theme.colors.success :
    props.status === 'offline' ? theme.colors.danger :
    theme.colors.warning
  };
`;

const AgentDetails = styled.div``;

const AgentName = styled.div`
  font-weight: ${theme.typography.fontWeights.semibold};
  color: ${theme.colors.text};
  font-size: ${theme.typography.fontSizes.base};
`;

const AgentMeta = styled.div`
  font-size: ${theme.typography.fontSizes.sm};
  color: ${theme.colors.textSecondary};
  margin-top: ${theme.spacing.xs};
`;

const StatusBadge = styled.div<{ status: string }>`
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.typography.fontSizes.xs};
  font-weight: ${theme.typography.fontWeights.medium};
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: ${props => 
    props.status === 'online' ? `${theme.colors.success}22` :
    props.status === 'offline' ? `${theme.colors.danger}22` :
    `${theme.colors.warning}22`
  };
  color: ${props => 
    props.status === 'online' ? theme.colors.success :
    props.status === 'offline' ? theme.colors.danger :
    theme.colors.warning
  };
`;

const LoadingMessage = styled.div`
  text-align: center;
  color: ${theme.colors.textSecondary};
  padding: ${theme.spacing.xl};
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${theme.spacing.sm};
`;

const ErrorMessage = styled.div`
  text-align: center;
  color: ${theme.colors.danger};
  padding: ${theme.spacing.xl};
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${theme.spacing.sm};
`;

const getStatusIcon = (status: string) => {
  switch (status.toLowerCase()) {
    case 'online':
      return <CheckCircle size={20} />;
    case 'offline':
      return <XCircle size={20} />;
    default:
      return <AlertCircle size={20} />;
  }
};

const formatAgentName = (agentName: string): string => {
  return agentName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
};

const AgentStatusPanel: React.FC = () => {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Check backend health first
        const isBackendHealthy = await checkBackendHealth();
        setBackendOnline(isBackendHealthy);
        
        // Fetch agent status
        const agentData = await fetchAgentStatus();
        setAgents(agentData);
      } catch (err) {
        console.error('Error fetching agent status:', err);
        setError('Failed to fetch agent status');
        setBackendOnline(false);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Set up polling for real-time updates
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Container
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Header>
          <Title>
            <Activity size={20} />
            Agent Status
          </Title>
        </Header>
        <LoadingMessage>
          <Clock size={16} className="animate-pulse" />
          Loading agent status...
        </LoadingMessage>
      </Container>
    );
  }

  if (error && !backendOnline) {
    return (
      <Container
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Header>
          <Title>
            <Activity size={20} />
            Agent Status
          </Title>
          <BackendStatus online={false}>
            <XCircle size={16} />
            Backend Offline
          </BackendStatus>
        </Header>
        <ErrorMessage>
          <AlertCircle size={16} />
          {error}
        </ErrorMessage>
      </Container>
    );
  }

  return (
    <Container
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Header>
        <Title>
          <Activity size={20} />
          Agent Status
        </Title>
        <BackendStatus online={backendOnline}>
          {backendOnline ? <CheckCircle size={16} /> : <XCircle size={16} />}
          Backend {backendOnline ? 'Online' : 'Offline'}
        </BackendStatus>
      </Header>

      <AgentList>
        {agents.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            color: theme.colors.textMuted, 
            padding: theme.spacing.lg 
          }}>
            No agents found
          </div>
        ) : (
          agents.map((agent, index) => (
            <AgentItem
              key={agent.agent_name}
              status={agent.status}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <AgentInfo>
                <AgentIcon status={agent.status}>
                  {getStatusIcon(agent.status)}
                </AgentIcon>
                <AgentDetails>
                  <AgentName>{formatAgentName(agent.agent_name)}</AgentName>
                  <AgentMeta>
                    {agent.last_run ? (
                      <>Last run: {formatTimestamp(agent.last_run)}</>
                    ) : (
                      'Never run'
                    )}
                    {agent.details && Object.keys(agent.details).length > 0 && (
                      <div style={{ marginTop: '4px', fontSize: '12px' }}>
                        {JSON.stringify(agent.details).length > 50 
                          ? `${JSON.stringify(agent.details).substring(0, 50)}...`
                          : JSON.stringify(agent.details)
                        }
                      </div>
                    )}
                  </AgentMeta>
                </AgentDetails>
              </AgentInfo>
              <StatusBadge status={agent.status}>
                {agent.status}
              </StatusBadge>
            </AgentItem>
          ))
        )}
      </AgentList>
    </Container>
  );
};

export default AgentStatusPanel; 