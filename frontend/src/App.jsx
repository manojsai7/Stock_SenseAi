import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import StockHeader from './components/StockHeader';
import StockChart from './components/StockChart';
import ForecastPanel from './components/ForecastPanel';
import BenchmarkMatrix from './components/BenchmarkMatrix';
import ArchitectureModal from './components/ArchitectureModal';
import {
  getStockInfo,
  getStockQuote,
  getStockHistory,
  getPrediction,
  getModelMetrics
} from './services/api';

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('TCS.NS');
  const [stockInfo, setStockInfo] = useState(null);
  const [stockQuote, setStockQuote] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const [period, setPeriod] = useState('1y');
  const [prediction, setPrediction] = useState(null);
  const [metricsData, setMetricsData] = useState(null);

  // Granular loading & error states
  const [loadingInfo, setLoadingInfo] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [chartError, setChartError] = useState(null);
  const [forecastError, setForecastError] = useState(null);

  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);

  // Controlled live quote polling (every 20s)
  useEffect(() => {
    let isMounted = true;

    async function fetchQuote() {
      try {
        const q = await getStockQuote(selectedSymbol);
        if (isMounted) {
          setStockQuote(q);
        }
      } catch (err) {
        console.warn('Quote polling notice:', err.message);
      }
    }

    fetchQuote();
    const intervalId = setInterval(fetchQuote, 20000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, [selectedSymbol]);

  // Load stock profile & fundamentals
  const loadStockInfo = useCallback(async (symbol) => {
    setLoadingInfo(true);
    try {
      const info = await getStockInfo(symbol);
      setStockInfo(info);
    } catch (err) {
      console.warn('Info load notice:', err.message);
    } finally {
      setLoadingInfo(false);
    }
  }, []);

  // Load historical price data
  const loadStockHistory = useCallback(async (symbol, p) => {
    setLoadingHistory(true);
    setChartError(null);
    try {
      const res = await getStockHistory(symbol, p);
      setHistoryData(res.data || []);
    } catch (err) {
      console.error('History load failed:', err.message);
      setChartError(err.message || 'Failed to load historical price series');
      setHistoryData([]);
    } finally {
      setLoadingHistory(false);
    }
  }, []);

  // Load benchmark metrics
  const loadModelMetrics = useCallback(async (symbol) => {
    try {
      const metrics = await getModelMetrics(symbol);
      setMetricsData(metrics);
    } catch (err) {
      console.warn('Metrics load notice:', err.message);
      setMetricsData(null);
    }
  }, []);

  // Load or run prediction
  const runPrediction = useCallback(async (symbol) => {
    setPredicting(true);
    setForecastError(null);
    try {
      const pred = await getPrediction(symbol);
      setPrediction(pred);
    } catch (err) {
      console.warn('Prediction notice:', err.message);
      setForecastError(err.message || 'Prediction unavailable');
      setPrediction(null);
    } finally {
      setPredicting(false);
    }
  }, []);

  // Orchestrate data fetching on symbol change
  useEffect(() => {
    let isCancelled = false;

    // Reset localized states to prevent stale flashes
    setPrediction(null);
    setForecastError(null);
    setChartError(null);

    loadStockInfo(selectedSymbol);
    loadStockHistory(selectedSymbol, period);
    loadModelMetrics(selectedSymbol);

    // Auto-fetch latest prediction on stock change
    (async () => {
      try {
        const pred = await getPrediction(selectedSymbol);
        if (!isCancelled) {
          setPrediction(pred);
        }
      } catch (err) {
        if (!isCancelled) {
          console.warn('Auto-prediction notice:', err.message);
          setForecastError(err.message);
        }
      }
    })();

    return () => {
      isCancelled = true;
    };
  }, [selectedSymbol, loadStockInfo, loadModelMetrics]);

  // Handle period change for historical chart
  const handlePeriodChange = (newPeriod) => {
    setPeriod(newPeriod);
    loadStockHistory(selectedSymbol, newPeriod);
  };

  // Stock selection handler
  const handleSelectSymbol = (newSymbol) => {
    if (newSymbol !== selectedSymbol) {
      setSelectedSymbol(newSymbol);
    }
  };

  const currencySymbol = '₹';

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'var(--canvas-bg)',
      color: 'var(--text-primary)'
    }}>
      {/* Navbar with Watchlist & Search */}
      <Navbar
        currentSymbol={selectedSymbol}
        onSelectSymbol={handleSelectSymbol}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
        marketStatus={stockQuote?.market_status}
      />

      {/* Main Financial Dashboard */}
      <main style={{
        maxWidth: '1360px',
        width: '100%',
        margin: '0 auto',
        padding: '24px 20px',
        flex: 1
      }}>
        {/* Stock Overview & Quote Header */}
        <StockHeader
          info={stockInfo}
          quote={stockQuote}
          loading={loadingInfo && !stockQuote}
        />

        {/* Historical Candlestick / Line Chart */}
        <StockChart
          historyData={historyData}
          period={period}
          onPeriodChange={handlePeriodChange}
          loading={loadingHistory}
          error={chartError}
          onRetry={() => loadStockHistory(selectedSymbol, period)}
          currency={currencySymbol}
        />

        {/* Next-Trading-Day Forecast */}
        <ForecastPanel
          prediction={prediction}
          onRunPrediction={() => runPrediction(selectedSymbol)}
          predicting={predicting}
          error={forecastError}
          onRetry={() => runPrediction(selectedSymbol)}
          symbol={selectedSymbol}
          currency={currencySymbol}
        />

        {/* Benchmark Evaluation Matrix */}
        <BenchmarkMatrix
          metricsData={metricsData}
          symbol={selectedSymbol}
          currency={currencySymbol}
        />
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        backgroundColor: '#FFFFFF',
        padding: '20px 24px',
        fontSize: '0.75rem',
        color: 'var(--text-secondary)'
      }}>
        <div style={{
          maxWidth: '1360px',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <div>
            <strong style={{ color: 'var(--text-primary)' }}>StockSense AI</strong> — Academic Deep Learning Time-Series Research © 2026.
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
            <span>Architecture: CNN-LSTM Hybrid</span>
            <span>•</span>
            <span>Feed: Yahoo Finance (NSE Equities)</span>
            <span>•</span>
            <span>Backend: FastAPI Monolith</span>
          </div>
        </div>
      </footer>

      {/* Architecture & Viva Defense Modal */}
      <ArchitectureModal
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />
    </div>
  );
}
