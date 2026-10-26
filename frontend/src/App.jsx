import React, { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { animate, createScope } from 'animejs';
import Navbar from './components/Navbar';
import StockHeader from './components/StockHeader';
import ForecastPanel from './components/ForecastPanel';
const StockChart = lazy(() => import('./components/StockChart'));
const BenchmarkMatrix = lazy(() => import('./components/BenchmarkMatrix'));
const ArchitectureModal = lazy(() => import('./components/ArchitectureModal'));
const FeatureExplainabilityPanel = lazy(() => import('./components/FeatureExplainabilityPanel'));
import {
  getStockInfo,
  getStockQuote,
  getStockHistory,
  getPrediction,
  getModelMetrics,
} from './services/api';

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
  const [chartError, setChartError] = useState(null);
  const [forecastError, setForecastError] = useState(null);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);

  const selectedSymbolRef = useRef(selectedSymbol);
  selectedSymbolRef.current = selectedSymbol;
  const requestRef = useRef({ info: 0, history: 0, metrics: 0, prediction: 0 });
  const heroRef = useRef(null);

  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return undefined;
    const scope = createScope({ root: heroRef }).add(() => {
      animate('.stock-eyebrow', {
        opacity: [0.82, 1],
        translateY: [3, 0],
        duration: 380,
        ease: 'out(3)',
      });
    });
    return () => scope.revert();
  }, [selectedSymbol]);

  const loadStockInfo = useCallback(async (symbol) => {
    const request = ++requestRef.current.info;
    setLoadingInfo(true);
    try {
      const info = await getStockInfo(symbol);
      if (requestRef.current.info === request && selectedSymbolRef.current === symbol) setStockInfo(info);
    } catch (error) {
      if (requestRef.current.info === request && selectedSymbolRef.current === symbol) setStockInfo(null);
      console.warn('Company profile unavailable:', error.message);
    } finally {
      if (requestRef.current.info === request && selectedSymbolRef.current === symbol) setLoadingInfo(false);
    }
  }, []);

  const loadStockHistory = useCallback(async (symbol, selectedPeriod) => {
    const request = ++requestRef.current.history;
    setLoadingHistory(true);
    setChartError(null);
    try {
      const response = await getStockHistory(symbol, selectedPeriod);
      if (requestRef.current.history === request && selectedSymbolRef.current === symbol) setHistoryData(response.data || []);
    } catch (error) {
      if (requestRef.current.history === request && selectedSymbolRef.current === symbol) {
        setChartError(error.message || 'Failed to load historical price series');
        setHistoryData([]);
      }
    } finally {
      if (requestRef.current.history === request && selectedSymbolRef.current === symbol) setLoadingHistory(false);
    }
  }, []);

  const loadModelMetrics = useCallback(async (symbol) => {
    const request = ++requestRef.current.metrics;
    try {
      const metrics = await getModelMetrics(symbol);
      if (requestRef.current.metrics === request && selectedSymbolRef.current === symbol) setMetricsData(metrics);
    } catch (error) {
      if (requestRef.current.metrics === request && selectedSymbolRef.current === symbol) setMetricsData(null);
      console.warn('Model evaluation unavailable:', error.message);
    }
  }, []);

  const runPrediction = useCallback(async (symbol) => {
    const request = ++requestRef.current.prediction;
    setPredicting(true);
    setForecastError(null);
    try {
      // Keep the live endpoint and pipeline query stable.
      const result = await getPrediction(symbol, 'stationary-return-v1');
      if (requestRef.current.prediction === request && selectedSymbolRef.current === symbol) {
        setPrediction(result);
        return true;
      }
      return false;
    } catch (error) {
      if (requestRef.current.prediction === request && selectedSymbolRef.current === symbol) {
        setForecastError(error.message || 'Prediction unavailable');
        setPrediction(null);
      }
      return false;
    } finally {
      if (requestRef.current.prediction === request && selectedSymbolRef.current === symbol) setPredicting(false);
    }
  }, []);

  useEffect(() => {
    setStockInfo(null);
    setStockQuote(null);
    setMetricsData(null);
    setPrediction(null);
    setForecastError(null);
    setLoadingInfo(true);
    loadStockInfo(selectedSymbol);
    loadModelMetrics(selectedSymbol);
    runPrediction(selectedSymbol);
    return () => {
      requestRef.current.info += 1;
      requestRef.current.metrics += 1;
      requestRef.current.prediction += 1;
    };
  }, [selectedSymbol, loadStockInfo, loadModelMetrics, runPrediction]);

  useEffect(() => {
    setHistoryData([]);
    setChartError(null);
    setLoadingHistory(true);
    loadStockHistory(selectedSymbol, period);
    return () => { requestRef.current.history += 1; };
  }, [selectedSymbol, period, loadStockHistory]);

  useEffect(() => {
    let active = true;
    setStockQuote(null);
    const fetchQuote = async () => {
      try {
        const quote = await getStockQuote(selectedSymbol);
        if (active && selectedSymbolRef.current === selectedSymbol) setStockQuote(quote);
      } catch (error) {
        if (active) console.warn('Quote update unavailable:', error.message);
      }
    };
    fetchQuote();
    const intervalId = setInterval(fetchQuote, 20_000);
    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, [selectedSymbol]);

  const handleSelectSymbol = (symbol) => {
    if (symbol !== selectedSymbol) {
      setStockInfo(null);
      setStockQuote(null);
      setHistoryData([]);
      setMetricsData(null);
      setPrediction(null);
      setChartError(null);
      setForecastError(null);
      setLoadingInfo(true);
      setLoadingHistory(true);
      setSelectedSymbol(symbol);
    }
  };

  const handlePeriodChange = (nextPeriod) => {
    if (nextPeriod === period) return;
    setHistoryData([]);
    setChartError(null);
    setLoadingHistory(true);
    setPeriod(nextPeriod);
  };

  return (
    <div className="app-shell" id="top">
      <Navbar
        currentSymbol={selectedSymbol}
        onSelectSymbol={handleSelectSymbol}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
        marketStatus={stockQuote?.market_status || 'UNKNOWN'}
      />

      <main className="dashboard-main">
        <motion.div
          ref={heroRef}
          className="dashboard-section"
          initial={{ opacity: 1, y: 7 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.34, ease: 'easeOut' }}
          key={`hero-${selectedSymbol}`}
        >
          <StockHeader info={stockInfo} quote={stockQuote} loading={loadingInfo && !stockQuote} />
        </motion.div>

        <motion.div className="dashboard-section" initial={{ opacity: 1, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.04 }}>
          <Suspense fallback={<section className="fintech-card chart-card chart-state" aria-busy="true" role="status">Preparing OHLC chart…</section>}>
            <StockChart
              historyData={historyData}
              period={period}
              onPeriodChange={handlePeriodChange}
              loading={loadingHistory}
              error={chartError}
              onRetry={() => loadStockHistory(selectedSymbol, period)}
            />
          </Suspense>
        </motion.div>

        <motion.div className="dashboard-section" initial={{ opacity: 1, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.08 }}>
          <ForecastPanel
            prediction={prediction}
            onRunPrediction={() => runPrediction(selectedSymbol)}
            predicting={predicting}
            error={forecastError}
            onRetry={() => runPrediction(selectedSymbol)}
            symbol={selectedSymbol}
          />
        </motion.div>

        <motion.div className="dashboard-section" initial={{ opacity: 1, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
          <Suspense fallback={<section className="fintech-card explainability-card" role="status" aria-busy="true">Preparing feature sensitivity analysis…</section>}>
            <FeatureExplainabilityPanel symbol={selectedSymbol} />
          </Suspense>
        </motion.div>

        <motion.div className="dashboard-section" initial={{ opacity: 1, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.12 }}>
          <Suspense fallback={metricsData ? <div className="benchmark-loading" role="status">Loading historical evaluation…</div> : null}>
            <BenchmarkMatrix metricsData={metricsData} symbol={selectedSymbol} />
          </Suspense>
        </motion.div>
      </main>

      <footer className="site-footer">
        <div><strong>StockSense AI</strong><span>Academic market research · {new Date().getFullYear()}</span></div>
        <div><span>FastAPI research service</span><span>Yahoo Finance market feed</span><span>CNN–LSTM models</span></div>
      </footer>

      <Suspense fallback={null}>
        <ArchitectureModal isOpen={isArchitectureOpen} onClose={() => setIsArchitectureOpen(false)} />
      </Suspense>
    </div>
  );
}
