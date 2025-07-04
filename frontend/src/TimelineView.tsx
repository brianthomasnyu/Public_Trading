import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Clock, FileText, TrendingUp, AlertCircle, Tag } from 'lucide-react';
import { fetchTimeline, TimelineEvent, formatTimestamp } from './api';
import { theme } from './theme';

// ========================================
// ENHANCED TIMELINE VIEW
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

const EventCount = styled.div`
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  background: ${theme.colors.primary}22;
  color: ${theme.colors.primary};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.typography.fontSizes.sm};
  font-weight: ${theme.typography.fontWeights.medium};
`;

const EventList = styled.div`
  display: grid;
  gap: ${theme.spacing.md};
  max-height: 400px;
  overflow-y: auto;
`;

const EventItem = styled(motion.div)<{ eventType: string }>`
  display: flex;
  gap: ${theme.spacing.md};
  padding: ${theme.spacing.md};
  background: ${theme.colors.surface};
  border-radius: ${theme.borderRadius.md};
  border-left: 4px solid ${props => 
    props.eventType === 'news' ? theme.colors.primary :
    props.eventType === 'sec_filing' ? theme.colors.success :
    props.eventType === 'earnings' ? theme.colors.warning :
    props.eventType === 'options_flow' ? '#9C27B0' :
    theme.colors.textMuted
  };
  transition: all ${theme.animations.fast};

  &:hover {
    background: ${theme.colors.surfaceLight};
    transform: translateX(4px);
  }
`;

const EventIcon = styled.div<{ eventType: string }>`
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: ${theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  background: ${props => 
    props.eventType === 'news' ? `${theme.colors.primary}22` :
    props.eventType === 'sec_filing' ? `${theme.colors.success}22` :
    props.eventType === 'earnings' ? `${theme.colors.warning}22` :
    props.eventType === 'options_flow' ? '#9C27B022' :
    `${theme.colors.textMuted}22`
  };
  color: ${props => 
    props.eventType === 'news' ? theme.colors.primary :
    props.eventType === 'sec_filing' ? theme.colors.success :
    props.eventType === 'earnings' ? theme.colors.warning :
    props.eventType === 'options_flow' ? '#9C27B0' :
    theme.colors.textMuted
  };
`;

const EventContent = styled.div`
  flex: 1;
  min-width: 0;
`;

const EventHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${theme.spacing.xs};
`;

const EventTitle = styled.div`
  font-weight: ${theme.typography.fontWeights.semibold};
  color: ${theme.colors.text};
  font-size: ${theme.typography.fontSizes.base};
`;

const EventTime = styled.div`
  font-size: ${theme.typography.fontSizes.sm};
  color: ${theme.colors.textSecondary};
`;

const EventSummary = styled.div`
  color: ${theme.colors.textSecondary};
  font-size: ${theme.typography.fontSizes.sm};
  margin-bottom: ${theme.spacing.sm};
  line-height: 1.4;
`;

const EventMeta = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  flex-wrap: wrap;
`;

const AgentBadge = styled.div`
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  background: ${theme.colors.backgroundLight};
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.typography.fontSizes.xs};
  color: ${theme.colors.textSecondary};
`;

const TickerBadge = styled.div`
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  background: ${theme.colors.primary}22;
  border: 1px solid ${theme.colors.primary};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.typography.fontSizes.xs};
  color: ${theme.colors.primary};
  font-weight: ${theme.typography.fontWeights.medium};
`;

const TagList = styled.div`
  display: flex;
  gap: ${theme.spacing.xs};
  flex-wrap: wrap;
`;

const TagItem = styled.div`
  padding: 2px ${theme.spacing.xs};
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.typography.fontSizes.xs};
  color: ${theme.colors.textMuted};
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

const getEventIcon = (eventType: string) => {
  switch (eventType) {
    case 'news':
      return <FileText size={20} />;
    case 'sec_filing':
      return <FileText size={20} />;
    case 'earnings':
      return <TrendingUp size={20} />;
    case 'options_flow':
      return <TrendingUp size={20} />;
    default:
      return <AlertCircle size={20} />;
  }
};

const TimelineView: React.FC = () => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const eventData = await fetchTimeline();
        setEvents(eventData);
      } catch (err) {
        console.error('Error fetching timeline:', err);
        setError('Failed to fetch timeline events');
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
            <Clock size={20} />
            Event Timeline
          </Title>
        </Header>
        <LoadingMessage>
          <Clock size={16} className="animate-pulse" />
          Loading timeline...
        </LoadingMessage>
      </Container>
    );
  }

  if (error) {
    return (
      <Container
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Header>
          <Title>
            <Clock size={20} />
            Event Timeline
          </Title>
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
          <Clock size={20} />
          Event Timeline
        </Title>
        <EventCount>{events.length} events</EventCount>
      </Header>

      <EventList>
        {events.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            color: theme.colors.textMuted, 
            padding: theme.spacing.lg 
          }}>
            No timeline events found
          </div>
        ) : (
          events.map((event, index) => (
            <EventItem
              key={event.id}
              eventType={event.event_type}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <EventIcon eventType={event.event_type}>
                {getEventIcon(event.event_type)}
              </EventIcon>
              
              <EventContent>
                <EventHeader>
                  <EventTitle>
                    {event.event_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </EventTitle>
                  <EventTime>
                    {formatTimestamp(event.event_time)}
                  </EventTime>
                </EventHeader>
                
                {event.summary && (
                  <EventSummary>{event.summary}</EventSummary>
                )}
                
                <EventMeta>
                  <AgentBadge>{event.source_agent.replace(/_/g, ' ')}</AgentBadge>
                  
                  {event.ticker && (
                    <TickerBadge>{event.ticker}</TickerBadge>
                  )}
                  
                  {event.tags && event.tags.length > 0 && (
                    <TagList>
                      <Tag size={12} color={theme.colors.textMuted} />
                      {event.tags.slice(0, 3).map((tag, tagIndex) => (
                        <TagItem key={tagIndex}>{tag}</TagItem>
                      ))}
                      {event.tags.length > 3 && (
                        <TagItem>+{event.tags.length - 3} more</TagItem>
                      )}
                    </TagList>
                  )}
                </EventMeta>
              </EventContent>
            </EventItem>
          ))
        )}
      </EventList>
    </Container>
  );
};

export default TimelineView; 