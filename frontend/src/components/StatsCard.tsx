import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import CountUp from 'react-countup';
import { theme } from '../theme';

interface StatsCardProps {
  title: string;
  value: number | string;
  prefix?: string;
  suffix?: string;
  trend?: 'up' | 'down' | 'neutral';
  percentage?: number;
  icon?: React.ReactNode;
  animated?: boolean;
  delay?: number;
}

const Card = styled(motion.div)`
  background: ${theme.colors.cardBackground};
  backdrop-filter: blur(10px);
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.lg};
  transition: all ${theme.animations.normal};
  position: relative;
  overflow: hidden;

  &:hover {
    background: ${theme.colors.cardBackgroundHover};
    border-color: ${theme.colors.primary};
    box-shadow: ${theme.shadows.glow};
    transform: translateY(-2px);
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: ${theme.colors.gradientBlue};
  }
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${theme.spacing.md};
`;

const Title = styled.h3`
  font-size: ${theme.typography.fontSizes.sm};
  font-weight: ${theme.typography.fontWeights.medium};
  color: ${theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const IconWrapper = styled.div`
  color: ${theme.colors.primary};
  opacity: 0.7;
`;

const ValueWrapper = styled.div`
  display: flex;
  align-items: baseline;
  gap: ${theme.spacing.xs};
  margin-bottom: ${theme.spacing.sm};
`;

const Value = styled.div`
  font-size: ${theme.typography.fontSizes.xxl};
  font-weight: ${theme.typography.fontWeights.bold};
  color: ${theme.colors.text};
  line-height: 1;
`;

const TrendWrapper = styled.div<{ trend: 'up' | 'down' | 'neutral' }>`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  font-size: ${theme.typography.fontSizes.sm};
  font-weight: ${theme.typography.fontWeights.medium};
  color: ${props => 
    props.trend === 'up' ? theme.colors.success :
    props.trend === 'down' ? theme.colors.danger :
    theme.colors.textSecondary
  };
`;

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  prefix = '',
  suffix = '',
  trend = 'neutral',
  percentage,
  icon,
  animated = true,
  delay = 0
}) => {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return '↗';
      case 'down':
        return '↘';
      default:
        return '→';
    }
  };

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
      initial={animated ? "hidden" : "visible"}
      animate="visible"
      whileHover={{ scale: 1.02 }}
    >
      <Header>
        <Title>{title}</Title>
        {icon && <IconWrapper>{icon}</IconWrapper>}
      </Header>
      
      <ValueWrapper>
        <Value>
          {prefix}
          {typeof value === 'number' && animated ? (
            <CountUp
              end={value}
              duration={1.5}
              delay={delay}
              preserveValue
              decimals={value % 1 !== 0 ? 2 : 0}
            />
          ) : (
            value
          )}
          {suffix}
        </Value>
      </ValueWrapper>

      {percentage !== undefined && (
        <TrendWrapper trend={trend}>
          <span>{getTrendIcon()}</span>
          <span>
            {percentage > 0 ? '+' : ''}{percentage.toFixed(2)}%
          </span>
        </TrendWrapper>
      )}
    </Card>
  );
};