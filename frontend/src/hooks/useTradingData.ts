import { useState, useEffect } from 'react';
import { 
  TradingStats, 
  TradingPosition, 
  ChartDataPoint, 
  SystemStatus,
  DetailedAgentStatus,
  getMockTradingStats,
  getMockPnLData,
  getMockTradingPositions,
  getMockSystemStatus,
  getMockDetailedAgentStatus
} from '../services/mockDataService';

// ========================================
// DATA HOOKS - USING MOCK DATA FOR NOW
// ========================================
// These hooks will be updated to use real API calls in the future

export const useTradingStats = () => {
  const [stats, setStats] = useState<TradingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Replace with real API call
    const fetchStats = async () => {
      try {
        setLoading(true);
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        const mockStats = getMockTradingStats();
        setStats(mockStats);
      } catch (err) {
        setError('Failed to fetch trading stats');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    
    // TODO: Set up real-time updates when using real data
    // const interval = setInterval(fetchStats, 30000); // Update every 30 seconds
    // return () => clearInterval(interval);
  }, []);

  return { stats, loading, error };
};

export const usePnLData = () => {
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Replace with real API call
    const fetchPnLData = async () => {
      try {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 300));
        const mockData = getMockPnLData();
        setData(mockData);
      } catch (err) {
        setError('Failed to fetch P&L data');
      } finally {
        setLoading(false);
      }
    };

    fetchPnLData();
  }, []);

  return { data, loading, error };
};

export const useTradingPositions = () => {
  const [positions, setPositions] = useState<TradingPosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Replace with real API call
    const fetchPositions = async () => {
      try {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 400));
        const mockPositions = getMockTradingPositions();
        setPositions(mockPositions);
      } catch (err) {
        setError('Failed to fetch trading positions');
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();
  }, []);

  return { positions, loading, error };
};

export const useSystemStatus = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Replace with real API call
    const fetchStatus = async () => {
      try {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 200));
        const mockStatus = getMockSystemStatus();
        setStatus(mockStatus);
      } catch (err) {
        setError('Failed to fetch system status');
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, []);

  return { status, loading, error };
};

// ========================================
// NEW HOOK FOR DETAILED AGENT STATUS
// ========================================

export const useDetailedAgentStatus = () => {
  const [agents, setAgents] = useState<DetailedAgentStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Replace with real API call to get detailed agent status
    const fetchDetailedAgentStatus = async () => {
      try {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 600));
        const mockAgents = getMockDetailedAgentStatus();
        setAgents(mockAgents);
      } catch (err) {
        setError('Failed to fetch detailed agent status');
      } finally {
        setLoading(false);
      }
    };

    fetchDetailedAgentStatus();

    // TODO: Set up real-time updates for agent status
    // const interval = setInterval(fetchDetailedAgentStatus, 10000); // Update every 10 seconds
    // return () => clearInterval(interval);
  }, []);

  return { agents, loading, error };
};