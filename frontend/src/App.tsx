import React from 'react';
import styled, { ThemeProvider } from 'styled-components';
import { GlobalStyles } from './GlobalStyles';
import { theme } from './theme';
import { Header } from './components/Header';
import { StatsCard } from './components/StatsCard';
import { Chart } from './components/Chart';
import { TradingPositions } from './components/TradingPositions';
import { AgentMonitorPanel } from './components/AgentMonitorPanel';
import QueryBar from './QueryBar';
import TimelineView from './TimelineView';
import { TrendingUp, DollarSign, Percent, Clock } from 'lucide-react';
import { 
  useTradingStats, 
  usePnLData, 
  useTradingPositions, 
  useDetailedAgentStatus 
} from './hooks/useTradingData';

const AppContainer = styled.div`
  min-height: 100vh;
  background: ${theme.colors.background};
`;

const MainContent = styled.main`
  max-width: 1400px;
  margin: 0 auto;
  padding: ${theme.spacing.xl};
  display: grid;
  gap: ${theme.spacing.xl};
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${theme.spacing.lg};
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: ${theme.spacing.lg};

  @media (max-width: ${theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
  }
`;

const ComponentsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${theme.spacing.lg};

  @media (max-width: ${theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
  }
`;

const FullWidthSection = styled.div`
  width: 100%;
`;

const LoadingCard = styled.div`
  background: ${theme.colors.cardBackground};
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${theme.colors.textSecondary};
`;

const App: React.FC = () => {
  // ========================================
  // USING MOCK DATA HOOKS - TO BE REPLACED
  // ========================================
  const { stats: tradingStats, loading: statsLoading } = useTradingStats();
  const { data: pnlData, loading: pnlLoading } = usePnLData();
  const { positions, loading: positionsLoading } = useTradingPositions();
  const { agents, loading: agentsLoading } = useDetailedAgentStatus();

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles />
      <AppContainer>
        <Header />
        <MainContent>
          {/* Trading Statistics */}
          <StatsGrid>
            {statsLoading ? (
              <LoadingCard>Loading stats...</LoadingCard>
            ) : tradingStats ? (
              <>
                <StatsCard
                  title="Total P&L"
                  value={tradingStats.totalPnL}
                  prefix="$"
                  trend={tradingStats.totalPnLPercentage > 0 ? "up" : "down"}
                  percentage={tradingStats.totalPnLPercentage}
                  icon={<DollarSign size={20} />}
                  delay={0}
                />
                <StatsCard
                  title="Win Rate"
                  value={tradingStats.winRate}
                  suffix="%"
                  trend={tradingStats.winRateChange > 0 ? "up" : "down"}
                  percentage={tradingStats.winRateChange}
                  icon={<Percent size={20} />}
                  delay={0.1}
                />
                <StatsCard
                  title="Average RR"
                  value={tradingStats.averageRR}
                  trend={tradingStats.averageRRChange > 0 ? "up" : "down"}
                  percentage={tradingStats.averageRRChange}
                  icon={<TrendingUp size={20} />}
                  delay={0.2}
                />
                <StatsCard
                  title="Avg Hold Time"
                  value={tradingStats.avgHoldTime}
                  trend={tradingStats.avgHoldTimeChange !== undefined ? 
                    (tradingStats.avgHoldTimeChange > 0 ? "up" : 
                     tradingStats.avgHoldTimeChange < 0 ? "down" : "neutral") : "neutral"}
                  percentage={tradingStats.avgHoldTimeChange}
                  icon={<Clock size={20} />}
                  delay={0.3}
                />
              </>
            ) : (
              <LoadingCard>Failed to load stats</LoadingCard>
            )}
          </StatsGrid>

          {/* Charts and Positions */}
          <ChartsGrid>
            {pnlLoading ? (
              <LoadingCard>Loading P&L chart...</LoadingCard>
            ) : (
              <Chart
                data={pnlData}
                title="P&L Chart"
                color={theme.colors.success}
                type="area"
              />
            )}
            
            {positionsLoading ? (
              <LoadingCard>Loading positions...</LoadingCard>
            ) : (
              <TradingPositions positions={positions} />
            )}
          </ChartsGrid>

          {/* Query Bar */}
          <QueryBar />

          {/* Agent Monitor Panel - Full Width */}
          <FullWidthSection>
            <AgentMonitorPanel 
              agents={agents} 
              loading={agentsLoading} 
            />
          </FullWidthSection>

          {/* Timeline and Legacy Agent Status */}
          <ComponentsGrid>
            <TimelineView />
          </ComponentsGrid>
        </MainContent>
      </AppContainer>
    </ThemeProvider>
  );
};

export default App; 