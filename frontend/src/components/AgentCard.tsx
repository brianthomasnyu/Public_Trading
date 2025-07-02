import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, Activity, AlertCircle, Pause, Play } from 'lucide-react';
import { theme } from '../theme';
import { DetailedAgentStatus, AgentDataFeed } from '../services/mockDataService';

interface AgentCardProps {
  agent: DetailedAgentStatus;
  delay?: number;
}

const Card = styled(motion.div)<{ status: string; agentColor: string }>`
  background: ${theme.colors.cardBackground};
  backdrop-filter: blur(10px);
  border: 1px solid ${props => 
    props.status === 'active' ? props.agentColor :
    props.status === 'error' ? theme.colors.danger :
    theme.colors.border
  };
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.lg};
  transition: all ${theme.animations.normal};
  position: relative;
  overflow: hidden;

  &:hover {
    background: ${theme.colors.cardBackgroundHover};
    border-color: ${props => props.agentColor};
    box-shadow: 0 0 20px ${props => props.agentColor}33;
    transform: translateY(-2px);
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: ${props => props.agentColor};
    opacity: ${props => props.status === 'active' ? 1 : 0.3};
  }
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${theme.spacing.md};
`;

const AgentInfo = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.md};
`;

const AgentIcon = styled.div<{ agentColor: string }>`
  font-size: 24px;
  width: 48px;
  height: 48px;
  background: ${props => props.agentColor}22;
  border: 2px solid ${props => props.agentColor};
  border-radius: ${theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const AgentDetails = styled.div`
  flex: 1;
`;

const AgentName = styled.h3`
  font-size: ${theme.typography.fontSizes.lg};
  font-weight: ${theme.typography.fontWeights.semibold};
  color: ${theme.colors.text};
  margin: 0;
`;

const AgentDescription = styled.p`
  font-size: ${theme.typography.fontSizes.sm};
  color: ${theme.colors.textSecondary};
  margin: ${theme.spacing.xs} 0 0 0;
`;

const StatusIndicator = styled.div<{ status: string; agentColor: string }>`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  border-radius: ${theme.borderRadius.md};
  background: ${props => 
    props.status === 'active' ? `${props.agentColor}22` :
    props.status === 'error' ? `${theme.colors.danger}22` :
    props.status === 'idle' ? `${theme.colors.warning}22` :
    `${theme.colors.textMuted}22`
  };
  color: ${props => 
    props.status === 'active' ? props.agentColor :
    props.status === 'error' ? theme.colors.danger :
    props.status === 'idle' ? theme.colors.warning :
    theme.colors.textMuted
  };
  font-size: ${theme.typography.fontSizes.sm};
  font-weight: ${theme.typography.fontWeights.medium};
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${theme.spacing.md};
  margin: ${theme.spacing.md} 0;
`;

const MetricItem = styled.div`
  text-align: center;
`;

const MetricValue = styled.div`
  font-size: ${theme.typography.fontSizes.lg};
  font-weight: ${theme.typography.fontWeights.bold};
  color: ${theme.colors.text};
`;

const MetricLabel = styled.div`
  font-size: ${theme.typography.fontSizes.xs};
  color: ${theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: ${theme.spacing.xs};
`;

const ExpandButton = styled(motion.button)`
  background: none;
  border: none;
  color: ${theme.colors.textSecondary};
  cursor: pointer;
  padding: ${theme.spacing.xs};
  border-radius: ${theme.borderRadius.md};
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  font-size: ${theme.typography.fontSizes.sm};

  &:hover {
    background: ${theme.colors.surfaceLight};
    color: ${theme.colors.text};
  }
`;

const DataFeedsSection = styled(motion.div)`
  border-top: 1px solid ${theme.colors.border};
  padding-top: ${theme.spacing.md};
  margin-top: ${theme.spacing.md};
`;

const DataFeedItem = styled(motion.div)`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${theme.spacing.sm};
  margin: ${theme.spacing.xs} 0;
  background: ${theme.colors.surfaceLight};
  border-radius: ${theme.borderRadius.md};
  border-left: 3px solid ${theme.colors.primary};
`;

const DataFeedInfo = styled.div`
  flex: 1;
`;

const DataFeedSource = styled.div`
  font-size: ${theme.typography.fontSizes.sm};
  font-weight: ${theme.typography.fontWeights.medium};
  color: ${theme.colors.text};
`;

const DataFeedValue = styled.div`
  font-size: ${theme.typography.fontSizes.xs};
  color: ${theme.colors.textSecondary};
  margin-top: ${theme.spacing.xs};
`;

const DataFeedMeta = styled.div`
  text-align: right;
  font-size: ${theme.typography.fontSizes.xs};
  color: ${theme.colors.textMuted};
`;

const ConfidenceBar = styled.div<{ confidence: number; agentColor: string }>`
  width: 40px;
  height: 4px;
  background: ${theme.colors.surface};
  border-radius: 2px;
  overflow: hidden;
  margin-top: ${theme.spacing.xs};

  &::after {
    content: '';
    display: block;
    width: ${props => props.confidence}%;
    height: 100%;
    background: ${props => props.agentColor};
    transition: width ${theme.animations.normal};
  }
`;

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

export const AgentCard: React.FC<AgentCardProps> = ({ agent, delay = 0 }) => {
  const [expanded, setExpanded] = useState(false);

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.5,
        delay: delay
      }
    }
  };

  return (
    <Card
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover={{ scale: 1.02 }}
      status={agent.status}
      agentColor={agent.color}
    >
      <Header>
        <AgentInfo>
          <AgentIcon agentColor={agent.color}>
            {agent.icon}
          </AgentIcon>
          <AgentDetails>
            <AgentName>{agent.display_name}</AgentName>
            <AgentDescription>{agent.description}</AgentDescription>
          </AgentDetails>
        </AgentInfo>
        <StatusIndicator status={agent.status} agentColor={agent.color}>
          {getStatusIcon(agent.status)}
          <span>{agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
        </StatusIndicator>
      </Header>

      <MetricsGrid>
        <MetricItem>
          <MetricValue>{agent.metrics.uptime.toFixed(1)}%</MetricValue>
          <MetricLabel>Uptime</MetricLabel>
        </MetricItem>
        <MetricItem>
          <MetricValue>{agent.metrics.success_rate.toFixed(1)}%</MetricValue>
          <MetricLabel>Success Rate</MetricLabel>
        </MetricItem>
        <MetricItem>
          <MetricValue>{agent.metrics.avg_processing_time.toFixed(1)}s</MetricValue>
          <MetricLabel>Avg Time</MetricLabel>
        </MetricItem>
        <MetricItem>
          <MetricValue>{agent.metrics.data_points_processed.toLocaleString()}</MetricValue>
          <MetricLabel>Data Points</MetricLabel>
        </MetricItem>
      </MetricsGrid>

      <ExpandButton
        onClick={() => setExpanded(!expanded)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        <span>{expanded ? 'Hide' : 'Show'} Data Feeds ({agent.data_feeds.length})</span>
      </ExpandButton>

      <AnimatePresence>
        {expanded && (
          <DataFeedsSection
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            {agent.data_feeds.length === 0 ? (
              <div style={{ 
                textAlign: 'center', 
                color: theme.colors.textMuted, 
                padding: theme.spacing.md 
              }}>
                No active data feeds
              </div>
            ) : (
              agent.data_feeds.map((feed, index) => (
                <DataFeedItem
                  key={feed.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <DataFeedInfo>
                    <DataFeedSource>
                      {feed.source} {feed.symbol && `(${feed.symbol})`}
                    </DataFeedSource>
                    <DataFeedValue>
                      {feed.dataType}: {feed.value}
                    </DataFeedValue>
                  </DataFeedInfo>
                  <DataFeedMeta>
                    <div>{feed.lastUpdate}</div>
                    {feed.confidence && (
                      <>
                        <div>Confidence: {feed.confidence}%</div>
                        <ConfidenceBar 
                          confidence={feed.confidence} 
                          agentColor={agent.color}
                        />
                      </>
                    )}
                  </DataFeedMeta>
                </DataFeedItem>
              ))
            )}
          </DataFeedsSection>
        )}
      </AnimatePresence>
    </Card>
  );
};