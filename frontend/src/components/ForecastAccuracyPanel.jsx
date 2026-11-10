import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import {
  ShieldCheck,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  RotateCw,
  Info,
  Calendar,
  CheckCircle2,
  XCircle,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp,
  BarChart2,
  HelpCircle,
  Lock,
} from 'lucide-react';
import { getForecastAccuracy } from '../services/api';
import './ForecastAccuracyPanel.css';

export default function ForecastAccuracyPanel({ symbol = 'RELIANCE.NS' }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('RESOLVED'); // 'RESOLVED', 'PENDING', 'ALL'
  const [limitFilter, setLimitFilter] = useState(null); // null (all), 20, 60
  const [activeTabModel, setActiveTabModel] = useState('GRU V2 Challenger');
  const [showGuide, setShowGuide] = useState(false);
  const [showHistorical, setShowHistorical] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getForecastAccuracy(symbol, {
        limit: limitFilter,
        includeHistorical: true,
      });
      setData(res);

      // Auto-select first available model family for the summary tab
      if (res?.per_family_metrics) {
        const families = Object.keys(res.per_family_metrics);
        if (families.includes('GRU V2 Challenger')) {
          setActiveTabModel('GRU V2 Challenger');
        } else if (families.length > 0) {
          setActiveTabModel(families[0]);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to load forecast accuracy track record');
    } finally {
      setLoading(false);
    }
  }, [symbol, limitFilter]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Unsupported Symbol State
  if (data && !data.supported) {
    return (
      <section className="fintech-card forecast-accuracy-card" aria-labelledby="accuracy-heading">
        <div className="accuracy-header">
          <div>
            <span className="accuracy-eyebrow">
              <ShieldCheck size={14} />
              PROSPECTIVE TRACK RECORD · SHADOW EVALUATION
            </span>
            <h2 id="accuracy-heading">Forecast Accuracy & Track Record</h2>
            <p className="accuracy-subtitle">See how closely our models have predicted actual stock prices.</p>
          </div>
        </div>
        <div className="accuracy-unsupported-box">
          <Lock size={32} className="accuracy-unsupported-icon" />
          <h3>Prospective Tracking Locked to RELIANCE.NS</h3>
          <p>
            Prospective forward evaluation under protocols EXP-05 and EXP-06 is currently restricted
            exclusively to <strong>RELIANCE.NS</strong>.
          </p>
          <span className="accuracy-unsupported-note">
            Constituent '{symbol}' is not enrolled in the live prospective shadow registry. Switch to RELIANCE.NS to inspect genuine predictive outcomes.
          </span>
        </div>
      </section>
    );
  }

  // Filter records for table
  const filteredRecords = useMemo(() => {
    if (!data) return [];
    let list = [];
    if (statusFilter === 'RESOLVED') {
      list = [...(data.resolved_records || [])];
    } else if (statusFilter === 'PENDING') {
      list = [...(data.pending_records || [])];
    } else {
      list = [...(data.resolved_records || []), ...(data.pending_records || [])];
    }

    if (selectedModel !== 'ALL') {
      list = list.filter(
        (r) =>
          r.model_family === selectedModel ||
          r.model_name === selectedModel ||
          r.model_name.toLowerCase().includes(selectedModel.toLowerCase())
      );
    }

    // Sort descending by target_date
    return list.sort((a, b) => (b.target_date || '').localeCompare(a.target_date || ''));
  }, [data, statusFilter, selectedModel]);

  const activeMetrics = useMemo(() => {
    if (!data?.per_family_metrics) return null;
    return data.per_family_metrics[activeTabModel] || Object.values(data.per_family_metrics)[0] || null;
  }, [data, activeTabModel]);

  const assessment = data?.sample_size_assessment || {
    tier: 'Awaiting Data',
    note: 'Loading prospective tracking logs...',
  };

  const chartData = data?.chart_data || [];
  const pairedData = data?.common_date_paired_metrics || {};
  const historicalBenchmarks = data?.historical_backtest_benchmarks || {};

  return (
    <section className="fintech-card forecast-accuracy-card" aria-labelledby="accuracy-heading">
      {/* Header */}
      <div className="accuracy-header">
        <div>
          <span className="accuracy-eyebrow">
            <ShieldCheck size={14} />
            VERIFIED PROSPECTIVE OUTCOMES · AUDITED SHADOW EVALUATION
          </span>
          <h2 id="accuracy-heading">Forecast Accuracy & Track Record</h2>
          <p className="accuracy-subtitle">See how closely our models have predicted actual stock prices.</p>
        </div>

        <div className="accuracy-header-controls">
          <div className="accuracy-limit-toggle" role="group" aria-label="Session Filter">
            <button
              type="button"
              className={`accuracy-btn-pill ${limitFilter === null ? 'active' : ''}`}
              onClick={() => setLimitFilter(null)}
            >
              All Sessions
            </button>
            <button
              type="button"
              className={`accuracy-btn-pill ${limitFilter === 20 ? 'active' : ''}`}
              onClick={() => setLimitFilter(20)}
            >
              Last 20
            </button>
            <button
              type="button"
              className={`accuracy-btn-pill ${limitFilter === 60 ? 'active' : ''}`}
              onClick={() => setLimitFilter(60)}
            >
              Last 60
            </button>
          </div>

          <button
            type="button"
            className="accuracy-refresh-btn"
            onClick={fetchData}
            disabled={loading}
            aria-label="Refresh forecast accuracy metrics"
          >
            <RotateCw size={14} className={loading ? 'accuracy-spinning' : ''} />
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="accuracy-banner accuracy-banner-error" role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>Unable to Load Track Record:</strong> {error}
          </div>
          <button type="button" className="accuracy-retry-btn" onClick={fetchData}>
            Retry
          </button>
        </div>
      )}

      {/* Prominent Sample Size & Reliability Warning Banner */}
      <div className="accuracy-sample-banner">
        <div className="accuracy-sample-badge">
          <Clock size={16} />
          <span>{assessment.tier.toUpperCase()}</span>
        </div>
        <div className="accuracy-sample-content">
          <div className="accuracy-sample-title">
            <strong>Resolved Prospective Sessions: {data?.total_available_resolved_sessions || 0}</strong>
            <span className="accuracy-sample-sep">·</span>
            <span>Target Gate: 60–100 Sessions</span>
            <span className="accuracy-sample-sep">·</span>
            <span className="accuracy-sample-pending">
              {data?.total_pending_forecasts || 0} Forward Predictions Pending
            </span>
          </div>
          <p className="accuracy-sample-note">{assessment.note}</p>
        </div>
      </div>

      {/* Model Family Tabs */}
      <div className="accuracy-model-tabs" role="tablist">
        {data?.per_family_metrics &&
          Object.keys(data.per_family_metrics).map((fam) => {
            const m = data.per_family_metrics[fam];
            const isActive = activeTabModel === fam;
            return (
              <button
                key={fam}
                type="button"
                role="tab"
                aria-selected={isActive}
                className={`accuracy-model-tab ${isActive ? 'active' : ''}`}
                onClick={() => setActiveTabModel(fam)}
                style={{ '--tab-accent': m.color || '#0ea5e9' }}
              >
                <span className="accuracy-tab-dot" style={{ backgroundColor: m.color || '#0ea5e9' }} />
                <span className="accuracy-tab-name">{fam}</span>
                <span className="accuracy-tab-count">N={m.sample_size}</span>
              </button>
            );
          })}
      </div>

      {/* Overall Model Performance Scorecards for Selected Model */}
      {activeMetrics && (
        <div className="accuracy-scorecard-grid">
          <div className="accuracy-metric-card primary-card">
            <span className="metric-label">Mean Absolute Error (MAE)</span>
            <div className="metric-val-row">
              <span className="metric-val">
                {activeMetrics.price_mae !== null ? `₹${activeMetrics.price_mae.toFixed(2)}` : '—'}
              </span>
              <span className="metric-unit">per share</span>
            </div>
            <span className="metric-desc">Average absolute error in rupees against settlement</span>
          </div>

          <div className="accuracy-metric-card">
            <span className="metric-label">Directional Accuracy (Hit Rate)</span>
            <div className="metric-val-row">
              <span
                className={`metric-val ${
                  activeMetrics.directional_accuracy_pct >= 50 ? 'accent-green' : 'accent-amber'
                }`}
              >
                {activeMetrics.directional_accuracy_pct !== null
                  ? `${activeMetrics.directional_accuracy_pct.toFixed(1)}%`
                  : '—'}
              </span>
            </div>
            <span className="metric-desc">
              Correct sign match: sign(r̂) == sign(r)
            </span>
          </div>

          <div className="accuracy-metric-card">
            <span className="metric-label">Mean Abs Percentage Error (MAPE)</span>
            <div className="metric-val-row">
              <span className="metric-val">
                {activeMetrics.price_mape !== null ? `${activeMetrics.price_mape.toFixed(2)}%` : '—'}
              </span>
            </div>
            <span className="metric-desc">Average percentage divergence from price</span>
          </div>

          <div className="accuracy-metric-card">
            <span className="metric-label">Root Mean Squared Error (RMSE)</span>
            <div className="metric-val-row">
              <span className="metric-val">
                {activeMetrics.price_rmse !== null ? `₹${activeMetrics.price_rmse.toFixed(2)}` : '—'}
              </span>
            </div>
            <span className="metric-desc">Penalizes larger outliers more heavily</span>
          </div>

          <div className="accuracy-metric-card">
            <span className="metric-label">Matthews Correlation (MCC)</span>
            <div className="metric-val-row">
              <span
                className={`metric-val ${
                  activeMetrics.mcc !== null && activeMetrics.mcc > 0
                    ? 'accent-green'
                    : activeMetrics.mcc !== null && activeMetrics.mcc < 0
                    ? 'accent-red'
                    : ''
                }`}
              >
                {activeMetrics.mcc !== null
                  ? (activeMetrics.mcc > 0 ? `+${activeMetrics.mcc.toFixed(4)}` : activeMetrics.mcc.toFixed(4))
                  : 'N/A'}
              </span>
            </div>
            <span className="metric-desc">
              {activeMetrics.mcc !== null
                ? 'Balanced binary directional quality [-1 to +1]'
                : 'N/A — insufficient class variation in sample'}
            </span>
          </div>

          <div className="accuracy-metric-card">
            <span className="metric-label">Return Prediction Error</span>
            <div className="metric-val-row">
              <span className="metric-val">
                {activeMetrics.return_mae_bps !== null ? `${activeMetrics.return_mae_bps} bps` : '—'}
              </span>
            </div>
            <span className="metric-desc">Basis points difference on daily return</span>
          </div>
        </div>
      )}

      {/* Directional Asymmetry & Recall Decomposition */}
      {activeMetrics && activeMetrics.sample_size > 0 && (
        <div className="accuracy-recall-card">
          <div className="recall-header">
            <div>
              <h4>Directional Recall Decomposition ({activeTabModel})</h4>
              <p>Transparent breakdown revealing model behavior on bullish vs bearish days.</p>
            </div>
            <span className="recall-balanced-badge">
              Balanced Accuracy: {activeMetrics.balanced_accuracy_pct !== null ? `${activeMetrics.balanced_accuracy_pct}%` : 'N/A (Single-Class)'}
            </span>
          </div>

          <div className="recall-bars-grid">
            <div className="recall-bar-item">
              <div className="recall-bar-label">
                <span>UP Recall (Sensitivity to Market Advances)</span>
                <strong>{activeMetrics.up_recall_pct !== null ? `${activeMetrics.up_recall_pct}%` : 'N/A'}</strong>
              </div>
              <div className="recall-progress-track">
                <div
                  className="recall-progress-fill up-fill"
                  style={{ width: `${Math.min(100, activeMetrics.up_recall_pct || 0)}%` }}
                />
              </div>
              <span className="recall-sub">
                {activeMetrics.confusion.tp + activeMetrics.confusion.fn > 0
                  ? `Correctly anticipated ${activeMetrics.confusion.tp} of ${activeMetrics.confusion.tp + activeMetrics.confusion.fn} upward sessions`
                  : 'N/A — 0 upward sessions observed in sample'}
              </span>
            </div>

            <div className="recall-bar-item">
              <div className="recall-bar-label">
                <span>DOWN Recall (Sensitivity to Market Pullbacks)</span>
                <strong>{activeMetrics.down_recall_pct !== null ? `${activeMetrics.down_recall_pct}%` : 'N/A'}</strong>
              </div>
              <div className="recall-progress-track">
                <div
                  className="recall-progress-fill down-fill"
                  style={{ width: `${Math.min(100, activeMetrics.down_recall_pct || 0)}%` }}
                />
              </div>
              <span className="recall-sub">
                {activeMetrics.confusion.tn + activeMetrics.confusion.fp > 0
                  ? `Correctly anticipated ${activeMetrics.confusion.tn} of ${activeMetrics.confusion.tn + activeMetrics.confusion.fp} downward/flat sessions`
                  : 'N/A — 0 downward sessions observed in sample (all actual sessions were UP)'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Common-Date Paired Comparison Matrix */}
      <div className="accuracy-paired-section">
        <div className="paired-section-header">
          <div>
            <h3>Fair Head-to-Head Comparison (Common Resolved Sessions)</h3>
            <p>
              Calculated exclusively on sessions where all comparison models generated predictions,
              evaluated against identical settlement prices.
            </p>
          </div>
          <span className="paired-sessions-badge">
            <Calendar size={13} />
            {pairedData.common_dates_count || 0} Identical Session(s) Evaluated
          </span>
        </div>

        <div className="accuracy-table-wrapper">
          <table className="accuracy-table paired-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Role</th>
                <th>Price MAE (₹) ↓</th>
                <th>Price RMSE (₹) ↓</th>
                <th>Price MAPE (%) ↓</th>
                <th>Direction Acc (%) ↑</th>
                <th>Directional MCC ↑</th>
                <th>Evaluated Sessions</th>
              </tr>
            </thead>
            <tbody>
              {pairedData.metrics_per_model &&
                Object.keys(pairedData.metrics_per_model).map((modelName) => {
                  const m = pairedData.metrics_per_model[modelName];
                  const isGRU = modelName.includes('GRU');
                  const isProd = modelName.includes('CNN') || modelName.includes('Production');
                  const role = isGRU
                    ? 'Challenger (Shadow)'
                    : isProd
                    ? 'Production V1'
                    : 'Reference Baseline';

                  return (
                    <tr key={modelName} className={isGRU ? 'highlight-row' : ''}>
                      <td>
                        <div className="table-model-cell">
                          <span
                            className="table-model-dot"
                            style={{ backgroundColor: m.color || '#0ea5e9' }}
                          />
                          <strong>{modelName}</strong>
                        </div>
                      </td>
                      <td>
                        <span className={`table-role-badge ${isGRU ? 'challenger' : isProd ? 'production' : 'baseline'}`}>
                          {role}
                        </span>
                      </td>
                      <td>
                        <strong>{m.price_mae !== null ? `₹${m.price_mae.toFixed(2)}` : '—'}</strong>
                      </td>
                      <td>{m.price_rmse !== null ? `₹${m.price_rmse.toFixed(2)}` : '—'}</td>
                      <td>{m.price_mape !== null ? `${m.price_mape.toFixed(2)}%` : '—'}</td>
                      <td>
                        <span className={m.directional_accuracy_pct >= 50 ? 'accent-green' : 'accent-amber'}>
                          {m.directional_accuracy_pct !== null ? `${m.directional_accuracy_pct.toFixed(1)}%` : '—'}
                        </span>
                      </td>
                      <td>
                        {m.mcc !== null ? (m.mcc > 0 ? `+${m.mcc.toFixed(4)}` : `${m.mcc.toFixed(4)}`) : 'N/A*'}
                      </td>
                      <td>
                        <span className="table-sample-tag">{m.sample_size} sessions</span>
                      </td>
                    </tr>
                  );
                })}
              {(!pairedData.metrics_per_model || Object.keys(pairedData.metrics_per_model).length === 0) && (
                <tr>
                  <td colSpan="8" className="empty-table-cell">
                    No common resolved sessions available yet for paired comparison.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <p className="paired-table-footnote">
            * <strong>N/A Note:</strong> MCC is undefined (zero denominator) because all evaluated common sessions experienced upward price movements (100% positive class). At least one downward session is required to compute empirical correlation.
          </p>
        </div>
      </div>

      {/* Predicted vs Actual Price Chart */}
      <div className="accuracy-chart-card">
        <div className="chart-card-header">
          <div>
            <h3>Predicted vs. Actual Closing Prices (Timeline)</h3>
            <p>Chronological plot of realized market closes alongside verified pre-session model forecasts.</p>
          </div>
          <div className="chart-legend-pills">
            <span className="legend-pill" style={{ color: '#10b981' }}>
              ● Actual Market Close
            </span>
            <span className="legend-pill" style={{ color: '#0ea5e9' }}>
              -- GRU V2 Predicted
            </span>
            <span className="legend-pill" style={{ color: '#8b5cf6' }}>
              -- Production V1 Predicted
            </span>
            <span className="legend-pill" style={{ color: '#f59e0b' }}>
              ·· Naive Baseline
            </span>
          </div>
        </div>

        {chartData.length > 0 ? (
          <div className="accuracy-chart-container" style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis dataKey="target_date" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <YAxis
                  domain={['auto', 'auto']}
                  stroke="#94a3b8"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(val) => `₹${val}`}
                />
                <Tooltip content={<CustomAccuracyTooltip />} />
                <Line
                  type="monotone"
                  dataKey="actual_close"
                  name="Actual Market Close"
                  stroke="#10b981"
                  strokeWidth={2.8}
                  dot={{ r: 5, fill: '#10b981', strokeWidth: 1.5, stroke: '#0f172a' }}
                  activeDot={{ r: 7 }}
                />
                <Line
                  type="monotone"
                  dataKey="v2_gru_pred"
                  name="GRU V2 Predicted"
                  stroke="#0ea5e9"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ r: 4, fill: '#0ea5e9' }}
                />
                <Line
                  type="monotone"
                  dataKey="v1_prod_pred"
                  name="Production V1 Predicted"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 4, fill: '#8b5cf6' }}
                />
                <Line
                  type="monotone"
                  dataKey="naive_pred"
                  name="Naive Baseline"
                  stroke="#f59e0b"
                  strokeWidth={1.6}
                  strokeDasharray="2 2"
                  dot={{ r: 3, fill: '#f59e0b' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="chart-empty-placeholder">
            <Clock size={36} />
            <p>Awaiting additional resolved trading sessions to draw multi-day trajectory curves.</p>
          </div>
        )}
      </div>

      {/* Forecast History Table */}
      <div className="accuracy-history-section">
        <div className="history-header">
          <div>
            <h3>Forecast History & Realized Outcomes</h3>
            <p>Immutable record of forecasts logged before trading and scored against official market closes.</p>
          </div>

          <div className="history-filters">
            {/* Status Filter */}
            <div className="filter-group">
              <span className="filter-label">Status:</span>
              <button
                type="button"
                className={`filter-btn ${statusFilter === 'RESOLVED' ? 'active' : ''}`}
                onClick={() => setStatusFilter('RESOLVED')}
              >
                Resolved Only
              </button>
              <button
                type="button"
                className={`filter-btn ${statusFilter === 'PENDING' ? 'active' : ''}`}
                onClick={() => setStatusFilter('PENDING')}
              >
                Pending
              </button>
              <button
                type="button"
                className={`filter-btn ${statusFilter === 'ALL' ? 'active' : ''}`}
                onClick={() => setStatusFilter('ALL')}
              >
                All
              </button>
            </div>

            {/* Model Filter */}
            <div className="filter-group">
              <span className="filter-label">Model:</span>
              <select
                className="filter-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="ALL">All Models</option>
                <option value="GRU V2 Challenger">GRU V2 Challenger</option>
                <option value="Production CNN-LSTM V1">Production CNN-LSTM V1</option>
                <option value="Naive Persistence Baseline">Naive Baseline</option>
              </select>
            </div>
          </div>
        </div>

        <div className="accuracy-table-wrapper">
          <table className="accuracy-table history-table">
            <thead>
              <tr>
                <th>Target Date</th>
                <th>Model</th>
                <th>Input Prev Close</th>
                <th>Predicted Close</th>
                <th>Actual Close</th>
                <th>Abs Error (₹)</th>
                <th>Abs Error (%)</th>
                <th>Pred Dir</th>
                <th>Actual Dir</th>
                <th>Direction Result</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map((r, i) => {
                const isResolved = r.status === 'RESOLVED';
                const isCorrect = r.direction_correct === true;

                return (
                  <tr key={r.forecast_id || `${r.target_date}-${r.model_name}-${i}`}>
                    <td>
                      <span className="table-date">{r.target_date}</span>
                    </td>
                    <td>
                      <div className="table-model-cell">
                        <span
                          className="table-model-dot"
                          style={{ backgroundColor: r.model_color || '#94a3b8' }}
                        />
                        <span>{r.model_name}</span>
                      </div>
                    </td>
                    <td>₹{r.previous_close !== null ? r.previous_close.toFixed(2) : '—'}</td>
                    <td>
                      <strong>₹{r.predicted_close !== null ? r.predicted_close.toFixed(2) : '—'}</strong>
                    </td>
                    <td>
                      {isResolved ? (
                        <span className="actual-price-highlight">₹{r.actual_close?.toFixed(2)}</span>
                      ) : (
                        <span className="pending-badge">Pending</span>
                      )}
                    </td>
                    <td>
                      {isResolved ? (
                        <span className="error-val">₹{r.price_error_abs?.toFixed(2)}</span>
                      ) : (
                        <span className="pending-text">—</span>
                      )}
                    </td>
                    <td>
                      {isResolved ? (
                        <span>{r.price_error_pct !== null ? `${r.price_error_pct.toFixed(2)}%` : '—'}</span>
                      ) : (
                        <span className="pending-text">—</span>
                      )}
                    </td>
                    <td>
                      <span className={`dir-tag ${r.predicted_direction?.toLowerCase()}`}>
                        {r.predicted_direction}
                      </span>
                    </td>
                    <td>
                      {isResolved ? (
                        <span className={`dir-tag ${r.actual_direction?.toLowerCase()}`}>
                          {r.actual_direction}
                        </span>
                      ) : (
                        <span className="pending-text">—</span>
                      )}
                    </td>
                    <td>
                      {isResolved ? (
                        isCorrect ? (
                          <span className="outcome-tag outcome-correct">
                            <CheckCircle2 size={13} />
                            CORRECT
                          </span>
                        ) : (
                          <span className="outcome-tag outcome-incorrect">
                            <XCircle size={13} />
                            INCORRECT
                          </span>
                        )
                      ) : (
                        <span className="outcome-tag outcome-pending">
                          <Clock size={13} />
                          AWAITING
                        </span>
                      )}
                    </td>
                    <td>
                      <span className={`status-pill ${isResolved ? 'status-resolved' : 'status-pending'}`}>
                        {r.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
              {filteredRecords.length === 0 && (
                <tr>
                  <td colSpan="11" className="empty-table-cell">
                    No forecast records match the selected filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Accuracy Interpretation & Educational Guide (Accordion) */}
      <div className="accuracy-guide-card">
        <button
          type="button"
          className="guide-toggle-btn"
          onClick={() => setShowGuide(!showGuide)}
        >
          <div className="guide-toggle-title">
            <HelpCircle size={17} />
            <span>Understanding Metric Definitions & Financial Reality</span>
          </div>
          {showGuide ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showGuide && (
          <div className="guide-content">
            <div className="guide-grid">
              <div className="guide-item">
                <strong>Mean Absolute Error (MAE):</strong>
                <p>
                  Average rupee difference between predicted and actual prices. Lower is generally better,
                  but an edge of a few paise over naive persistence does not cover trading costs.
                </p>
              </div>
              <div className="guide-item">
                <strong>Root Mean Squared Error (RMSE):</strong>
                <p>
                  Squares prediction errors before averaging. Heavily penalizes large forecasting misses or
                  black-swan gap days.
                </p>
              </div>
              <div className="guide-item">
                <strong>Mean Absolute Percentage Error (MAPE):</strong>
                <p>
                  Scale-free percentage error: |Predicted - Actual| / Actual. Allows comparing accuracy
                  across stocks with different share price levels.
                </p>
              </div>
              <div className="guide-item">
                <strong>Directional Accuracy (Hit Rate):</strong>
                <p>
                  Percentage of days the predicted sign (+/-) matched market direction. In strong bull markets,
                  naive positive predictions can artificially inflate hit rate without true forecasting alpha.
                </p>
              </div>
              <div className="guide-item">
                <strong>Matthews Correlation Coefficient (MCC):</strong>
                <p>
                  Balanced measure assessing true binary directional skill across all four confusion matrix
                  quadrants (TP, TN, FP, FN). 0 means random guessing; +1 is perfect.
                </p>
              </div>
              <div className="guide-item">
                <strong>UP vs DOWN Recall:</strong>
                <p>
                  Measures whether the model is symmetrically capable or biased toward positive momentum.
                  Models trained during multi-year bull runs often suffer lower DOWN recall.
                </p>
              </div>
            </div>

            <div className="guide-warning">
              <Info size={16} />
              <span>
                <strong>Academic Disclaimer:</strong> StockSense AI is an empirical financial research prototype.
                Lower price error on historical or prospective samples does not guarantee trading returns,
                and models should never be used as standalone automated trading systems.
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Historical Offline Research Benchmarks (Clearly Decoupled) */}
      <div className="accuracy-historical-section">
        <button
          type="button"
          className="historical-toggle-btn"
          onClick={() => setShowHistorical(!showHistorical)}
        >
          <div className="historical-toggle-title">
            <Layers size={17} />
            <span>View Historical Development Benchmarks (EXP-06 3-Fold Cross-Validation)</span>
          </div>
          <span className="historical-disclaimer-pill">OFFLINE RESEARCH ONLY</span>
          {showHistorical ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showHistorical && (
          <div className="historical-content">
            <div className="historical-warning">
              <AlertTriangle size={16} />
              <span>
                <strong>Methodological Boundary:</strong> These benchmarks reflect retrospective 3-fold
                cross-validation on 1,233 audited historical trading sessions (2021–2026).
                Out-of-sample test splits: Fold 1 (246 test sessions), Fold 2 (185 test sessions), Fold 3 (123 test sessions / 100 sequence windows).
                They are provided for architectural comparison and must <strong>NEVER</strong> be conflated with prospective live outcomes.
              </span>
            </div>

            <div className="historical-split-cards">
              <div className="historical-split-badge">
                <span>Audited Dataset: <strong>1,233 rows</strong></span>
              </div>
              <div className="historical-split-badge">
                <span>Fold 1 Test: <strong>246 sessions</strong></span>
              </div>
              <div className="historical-split-badge">
                <span>Fold 2 Test: <strong>185 sessions</strong></span>
              </div>
              <div className="historical-split-badge">
                <span>Fold 3 Test: <strong>123 sessions</strong> (100 windows)</span>
              </div>
            </div>

            <div className="accuracy-table-wrapper">
              <table className="accuracy-table historical-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model Architecture</th>
                    <th>Role</th>
                    <th>Parameters</th>
                    <th>Cross-Fold Mean MAE (₹)</th>
                    <th>Fold 3 Test MAE (₹)</th>
                    <th>Directional Acc (%)</th>
                    <th>MCC</th>
                    <th>UP Recall</th>
                    <th>DOWN Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {(historicalBenchmarks.cross_fold_leaderboard || []).map((row) => (
                    <tr key={row.model} className={row.rank === 1 ? 'highlight-row' : ''}>
                      <td><strong>#{row.rank}</strong></td>
                      <td><strong>{row.model}</strong></td>
                      <td><span className="table-sample-tag">{row.role}</span></td>
                      <td>{row.parameters.toLocaleString()}</td>
                      <td><strong>₹{row.cross_fold_mae.toFixed(2)}</strong></td>
                      <td>₹{row.fold3_mae.toFixed(2)}</td>
                      <td>{row.fold3_directional_acc_pct}%</td>
                      <td>+{row.fold3_mcc.toFixed(4)}</td>
                      <td>{row.up_recall_pct}%</td>
                      <td>{row.down_recall_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function CustomAccuracyTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const pData = payload[0]?.payload || {};

  return (
    <div className="accuracy-chart-tooltip">
      <div className="tooltip-header">
        <span className="tooltip-date">Target Date: {label}</span>
        <span className={`tooltip-dir-tag ${pData.actual_direction?.toLowerCase()}`}>
          Market: {pData.actual_direction}
        </span>
      </div>

      <div className="tooltip-body">
        <div className="tooltip-row actual-row">
          <span>Actual Settlement:</span>
          <strong>₹{pData.actual_close !== null ? pData.actual_close?.toFixed(2) : '—'}</strong>
        </div>

        {pData.v2_gru_pred !== undefined && (
          <div className="tooltip-row">
            <span style={{ color: '#0ea5e9' }}>GRU V2 Forecast:</span>
            <strong>
              ₹{pData.v2_gru_pred?.toFixed(2)} (Err: ₹{pData.v2_gru_error?.toFixed(2)})
            </strong>
          </div>
        )}

        {pData.v1_prod_pred !== undefined && (
          <div className="tooltip-row">
            <span style={{ color: '#8b5cf6' }}>Production V1 Forecast:</span>
            <strong>
              ₹{pData.v1_prod_pred?.toFixed(2)} (Err: ₹{pData.v1_prod_error?.toFixed(2)})
            </strong>
          </div>
        )}

        {pData.naive_pred !== undefined && (
          <div className="tooltip-row">
            <span style={{ color: '#f59e0b' }}>Naive Baseline Forecast:</span>
            <strong>
              ₹{pData.naive_pred?.toFixed(2)} (Err: ₹{pData.naive_error?.toFixed(2)})
            </strong>
          </div>
        )}
      </div>
    </div>
  );
}

