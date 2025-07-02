import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Filter, Search, Activity, AlertCircle, Pause, Play } from 'lucide-react';
import { theme } from '../theme';
import { AgentCard } from './AgentCard';
import { DetailedAgentStatus } from '../services/mockDataService';

interface AgentMonitorPanelProps {
  agents: DetailedAgentStatus[];
  loading?: boolean;
}

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
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${theme.spacing.lg};
  flex-wrap: wrap;
  gap: ${theme.spacing.md};
`;

const Title = styled.h2`
  font-size: ${theme.typography.fontSizes.xl};
  font-weight: ${theme.typography.fontWeights.bold};
  color: ${theme.colors.text};
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
`;

const Controls = styled.div`
  display: flex;
  gap: ${theme.spacing.md};
  align-items: center;
  flex-wrap: wrap;
`;

const SearchInput = styled.input`
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.md};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  color: ${theme.colors.text};
  font-size: ${theme.typography.fontSizes.sm};
  min-width: 200px;

  &::placeholder {
    color: ${theme.colors.textMuted};
  }

  &:focus {
    outline: none;
    border-color: ${theme.colors.primary};
    box-shadow: 0 0 0 2px ${theme.colors.primary}33;
  }
`;

const FilterButton = styled(motion.button)<{ active: boolean }>`
  background: ${props => props.active ? theme.colors.primary : theme.colors.surface};
  border: 1px solid ${props => props.active ? theme.colors.primary : theme.colors.border};
  border-radius: ${theme.borderRadius.md};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  color: ${props => props.active ? theme.colors.text : theme.colors.textSecondary};
  font-size: ${theme.typography.fontSizes.sm};
  cursor: pointer;
  transition: all ${theme.animations.fast};

  &:hover {
    background: ${props => props.active ? theme.colors.primaryDark : theme.colors.surfaceLight};
    border-color: ${theme.colors.primary};
  }
`;

const StatusSummary = styled.div`
  display: flex;
  gap: ${theme.spacing.lg};
  margin-bottom: ${theme.spacing.lg};
  flex-wrap: wrap;
`;

const StatusItem = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  background: ${theme.colors.surface};
  border-radius: ${theme.borderRadius.md};
  border: 1px solid ${theme.colors.border};
`;

const StatusIcon = styled.div<{ status: string }>`
  color: ${props => 
    props.status === 'active' ? theme.colors.success :
    props.status === 'error' ? theme.colors.danger :
    props.status === 'idle' ? theme.colors.warning :
    theme.colors.textMuted
  };
`;

const StatusCount = styled.span`
  font-weight: ${theme.typography.fontWeights.semibold};
  color: ${theme.colors.text};
`;

const StatusLabel = styled.span`
  color: ${theme.colors.textSecondary};
  font-size: ${theme.typography.fontSizes.sm};
`;

const AgentsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: ${theme.spacing.lg};

  @media (max-width: ${theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
  }
`;

const LoadingMessage = styled.div`
  text-align: center;
  color: ${theme.colors.textSecondary};
  padding: ${theme.spacing.xl};
  font-size: ${theme.typography.fontSizes.lg};
`;

const NoAgentsMessage = styled.div`
  text-align: center;
  color: ${theme.colors.textMuted};
  padding: ${theme.spacing.xl};
  font-size: ${theme.typography.fontSizes.base};
`;

const categories = [
  { id: 'all', label: 'All Agents' },
  { id: 'market_data', label: 'Market Data' },
  { id: 'news', label: 'News' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'compliance', label: 'Compliance' },
  { id: 'social', label: 'Social' },
];

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'active':
      return <Activity size={16} />;
    case 'error':
      return <AlertCircle size={16} />;
    case 'idle':
      return <Pause size={16} />;
    default:
      return <Play size={16} />;
  }
};

export const AgentMonitorPanel: React.FC<AgentMonitorPanelProps> = ({ 
  agents, 
  loading = false 
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  // Filter agents based on search and category
  const filteredAgents = agents.filter(agent => {
    const matchesSearch = agent.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || agent.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  // Calculate status counts
  const statusCounts = agents.reduce((acc, agent) => {
    acc[agent.status] = (acc[agent.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  if (loading) {
    return (
      <Container
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <LoadingMessage>Loading agent status...</LoadingMessage>
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
          <Activity size={24} />
          Agent Monitor
        </Title>
        <Controls>
          <SearchInput
            type="text"
            placeholder="Search agents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <Filter size={16} color={theme.colors.textSecondary} />
          {categories.map(category => (
            <FilterButton
              key={category.id}
              active={selectedCategory === category.id}
              onClick={() => setSelectedCategory(category.id)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {category.label}
            </FilterButton>
          ))}
        </Controls>
      </Header>

      <StatusSummary>
        <StatusItem>
          <StatusIcon status="active">
            {getStatusIcon('active')}
          </StatusIcon>
          <StatusCount>{statusCounts.active || 0}</StatusCount>
          <StatusLabel>Active</StatusLabel>
        </StatusItem>
        <StatusItem>
          <StatusIcon status="idle">
            {getStatusIcon('idle')}
          </StatusIcon>
          <StatusCount>{statusCounts.idle || 0}</StatusCount>
          <StatusLabel>Idle</StatusLabel>
        </StatusItem>
        <StatusItem>
          <StatusIcon status="error">
            {getStatusIcon('error')}
          </StatusIcon>
          <StatusCount>{statusCounts.error || 0}</StatusCount>
          <StatusLabel>Error</StatusLabel>
        </StatusItem>
        <StatusItem>
          <StatusIcon status="stopped">
            {getStatusIcon('stopped')}
          </StatusIcon>
          <StatusCount>{statusCounts.stopped || 0}</StatusCount>
          <StatusLabel>Stopped</StatusLabel>
        </StatusItem>
      </StatusSummary>

      <AgentsGrid>
        <AnimatePresence mode="popLayout">
          {filteredAgents.length === 0 ? (
            <NoAgentsMessage>
              {searchTerm || selectedCategory !== 'all' 
                ? 'No agents match your filters' 
                : 'No agents available'
              }
            </NoAgentsMessage>
          ) : (
            filteredAgents.map((agent, index) => (
              <AgentCard 
                key={agent.agent_name} 
                agent={agent} 
                delay={index * 0.1}
              />
            ))
          )}
        </AnimatePresence>
      </AgentsGrid>
    </Container>
  );
};