import React, { useState, useEffect } from 'react';
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
  getModelMetrics,
  searchStocks
} from './services/api';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('TCS.NS');
  const [stockInfo, setStockInfo] = useState(null);
  const [stockQuote, setStockQuote] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const [period, setPeriod] = useState('1y');
  const [prediction, setPrediction] = useState(null);
  const [metricsData, setMetricsData] = useState(null);

  const [loadingInfo, setLoadingInfo] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

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
        console.warn('Quote polling notice:', err);
      }
    }

    fetchQuote();
    const intervalId = setInterval(fetchQuote, 20000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, [selectedSymbol]);


  // Load stock info & history whenever symbol or period changes
  useEffect(() => {
    let isCancelled = false;

    async function loadData() {
      setErrorMessage(null);
      setLoadingInfo(true);
      setLoadingHistory(true);

      try {
        const info = await getStockInfo(selectedSymbol);
        if (!isCancelled) {
          setStockInfo(info);
          setLoadingInfo(false);
        }
      } catch (err) {
        if (!isCancelled) {
          console.error('Info error:', err);
          setErrorMessage(`Unable to fetch profile for ${selectedSymbol}.`);
          setLoadingInfo(false);
        }
      }

      try {
        const historyRes = await getStockHistory(selectedSymbol, period);
        if (!isCancelled) {
          setHistoryData(historyRes.data || []);
          setLoadingHistory(false);
        }
      } catch (err) {
        if (!isCancelled) {
          console.error('History error:', err);
          setLoadingHistory(false);
        }
      }

      // Fetch model metrics for the symbol
      try {
        const metrics = await getModelMetrics(selectedSymbol);
        if (!isCancelled) {
          setMetricsData(metrics);
        }
      } catch (err) {
        console.warn('Metrics notice:', err);
      }

      // Auto-fetch latest prediction
      try {
        const pred = await getPrediction(selectedSymbol);
        if (!isCancelled) {
          setPrediction(pred);
        }
      } catch (err) {
        console.warn('Prediction auto-load notice:', err);
      }
    }

    loadData();

    return () => {
      isCancelled = true;
    };
  }, [selectedSymbol, period]);

  const handleRunPrediction = async () => {
    setPredicting(true);
    setErrorMessage(null);
    try {
      const res = await getPrediction(selectedSymbol);
      setPrediction(res);
    } catch (err) {
      console.error('Prediction failed:', err);
      setErrorMessage(`Prediction failed for ${selectedSymbol}: ${err.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const handleSelectSymbol = (sym) => {
    if (sym !== selectedSymbol) {
      setSelectedSymbol(sym);
      setPrediction(null);
    }
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    const query = searchQuery.trim().toUpperCase();
    if (query) {
      const formatted = query.includes('.') ? query : (['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'].includes(query) ? query : `${query}.NS`);
      handleSelectSymbol(formatted);
      setSearchQuery('');
    }
  };

  const currencySymbol = stockInfo?.currency === 'INR' ? '₹' : '$';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        currentSymbol={selectedSymbol}
        onSelectSymbol={handleSelectSymbol}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        onSearchSubmit={handleSearchSubmit}
      />

      <main style={{ maxWidth: '1440px', width: '100%', margin: '0 auto', padding: '24px', flex: 1 }}>
        {/* Error notification banner if any */}
        {errorMessage && (
          <div style={{
            background: 'rgba(244, 63, 94, 0.15)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '10px',
            padding: '12px 18px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: '#FB7185',
            fontSize: '0.85rem',
          }}>
            <AlertTriangle size={16} />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Stock Overview Header */}
        <StockHeader info={stockInfo} quote={stockQuote} loading={loadingInfo && !stockQuote} />

        {/* Historical Candlestick / Line Chart */}
        <StockChart
          historyData={historyData}
          period={period}
          onPeriodChange={setPeriod}
          loading={loadingHistory}
          currency={currencySymbol}
        />

        {/* CNN-LSTM AI Forecast Panel */}
        <ForecastPanel
          prediction={prediction}
          onRunPrediction={handleRunPrediction}
          predicting={predicting}
          symbol={selectedSymbol}
          currency={currencySymbol}
        />

        {/* Experimental Benchmark Comparison Matrix */}
        <BenchmarkMatrix
          metricsData={metricsData}
          symbol={selectedSymbol}
          currency={currencySymbol}
        />
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(7, 9, 14, 0.95)',
        padding: '24px',
        textAlign: 'center',
        color: 'var(--text-muted)',
        fontSize: '0.8rem',
      }}>
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <strong>StockSense AI</strong> — Academic Final Year Deep Learning Implementation &copy; 2026.
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.75rem' }}>
            <span>Architecture: CNN-LSTM Hybrid</span>
            <span>•</span>
            <span>Data Feed: Yahoo Finance (NSE/BSE)</span>
            <span>•</span>
            <span>Backend: FastAPI RESTful Monolith</span>
          </div>
        </div>
      </footer>

      {/* Architecture / Viva Modal */}
      <ArchitectureModal
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />
    </div>
  );
}
