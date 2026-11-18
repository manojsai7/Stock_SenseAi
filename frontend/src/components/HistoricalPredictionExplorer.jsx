import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import {
  Layers,
  Database,
  BarChart2,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Info,
  CheckCircle2,
  XCircle,
  RotateCw,
  Search,
  Filter,
  Award,
  ShieldCheck,
  ShieldAlert,
} from 'lucide-react';
import { getHistoricalPredictions } from '../services/api';
import HelpTooltip from './common/HelpTooltip';
import './HistoricalPredictionExplorer.css';

export default function HistoricalPredictionExplorer({ symbol = 'RELIANCE.NS' }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [selectedModel, setSelectedModel] = useState('GRU V2 Challenger');
  const [selectedFold, setSelectedFold] = useState('ALL');
  const [selectedExperiment, setSelectedExperiment] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [showMatrix, setShowMatrix] = useState(false);

  const fetchHistoricalData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getHistoricalPredictions(symbol, {
        modelName: selectedModel,
        fold: selectedFold !== 'ALL' ? selectedFold : undefined,
        experiment: selectedExperiment !== 'ALL' ? selectedExperiment : undefined,
        limit: 300,
      });
      setData(res);

      // If currently selected model not in available models, sync to first available
      if (res?.available_models && res.available_models.length > 0) {
        if (!res.available_models.includes(selectedModel)) {
          setSelectedModel(res.available_models[0]);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to load historical walk-forward prediction archives');
    } finally {
      setLoading(false);
    }
  }, [symbol, selectedModel, selectedFold, selectedExperiment]);

  useEffect(() => {
    fetchHistoricalData();
  }, [fetchHistoricalData]);

  const metrics = data?.metrics || {};
  const capabilityScore = data?.selected_model_capability_score || {};
  const coverageMatrix = data?.model_coverage_matrix || [];
  const chartData = data?.chart_data || [];
  const residualHistogram = data?.residual_histogram || [];

  // Filtered table records
  const filteredRecords = useMemo(() => {
    if (!data?.records) return [];
    if (!searchQuery.trim()) return data.records;
    const q = searchQuery.toLowerCase().trim();
    return data.records.filter(
      (r) =>
        (r.target_date || '').includes(q) ||
        (r.model_name || '').toLowerCase().includes(q) ||
        (r.fold || '').toLowerCase().includes(q)
    );
  }, [data, searchQuery]);

  return (
    <div className="historical-explorer-root">
      {/* Historical Offline Header Banner */}
      <div className="historical-header-banner">
        <div className="historical-tag-group">
          <span className="historical-offline-badge">
            <Database size={13} />
            HISTORICAL OFFLINE DEVELOPMENT EVALUATION
          </span>
          <span className="historical-equity-pill">{symbol}</span>
        </div>
        <h3>Archived Walk-Forward Predictions Explorer</h3>
        <p className="historical-header-desc">
          Inspect authentic, non-interpolated test-fold predictions from development experiments
          (EXP-05 & EXP-06). These records originate from historical out-of-sample folds and must
          <strong> NEVER</strong> be conflated with genuine pre-session prospective forward tracking.
        </p>
      </div>

      {/* Model Predicting Capability Scorecard */}
      {capabilityScore && capabilityScore.score !== undefined && capabilityScore.score !== null && (
        <div className="capability-scorecard-card">
          <div className="capability-top-row">
            <div className="capability-badge-cluster">
              <Award size={20} className="capability-award-icon" style={{ color: capabilityScore.color }} />
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span className="capability-score-kicker">EXPLORATORY MODEL CAPABILITY INDEX</span>
                  <HelpTooltip term="capability_score" />
                </div>
                <h4>
                  {selectedModel} · {symbol}
                </h4>
              </div>
            </div>
            <div className="capability-big-score" style={{ color: capabilityScore.color }}>
              <span className="score-num">{capabilityScore.score}</span>
              <span className="score-denom">/ 100</span>
              <span
                className="score-tier-badge"
                style={{ backgroundColor: `${capabilityScore.color}20`, color: capabilityScore.color }}
              >
                {capabilityScore.badge}
              </span>
            </div>
          </div>

          <p className="capability-summary-text">{capabilityScore.summary}</p>
          <div className="capability-disclaimer-sub">
            <Info size={13} />
            <span>
              <strong>Scientific Framing:</strong> Exploratory multi-factor comparative index (0–100) vs naive persistence on identical historical slices. Not a scientifically guaranteed probability of trading success.
            </span>
          </div>

          <div className="capability-components-grid">
            <div className="capability-sub-metric">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="sub-metric-label">Directional Skill</span>
                <HelpTooltip term="win_rate" />
              </div>
              <div className="sub-metric-bar-bg">
                <div
                  className="sub-metric-bar-fill"
                  style={{
                    width: `${capabilityScore.components?.directional_skill_score || 0}%`,
                    backgroundColor: '#0ea5e9',
                  }}
                />
              </div>
              <span className="sub-metric-value">
                {capabilityScore.components?.directional_skill_score || 0} / 100
                <small> (Hit: {capabilityScore.raw_inputs?.directional_accuracy_pct}%)</small>
              </span>
            </div>

            <div className="capability-sub-metric">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="sub-metric-label">Matthews Correlation (MCC)</span>
                <HelpTooltip term="mcc" />
              </div>
              <div className="sub-metric-bar-bg">
                <div
                  className="sub-metric-bar-fill"
                  style={{
                    width: `${capabilityScore.components?.mcc_correlation_score || 0}%`,
                    backgroundColor: '#8b5cf6',
                  }}
                />
              </div>
              <span className="sub-metric-value">
                {capabilityScore.components?.mcc_correlation_score || 0} / 100
                <small> (MCC: {capabilityScore.raw_inputs?.mcc > 0 ? `+${capabilityScore.raw_inputs.mcc}` : capabilityScore.raw_inputs?.mcc})</small>
              </span>
            </div>

            <div className="capability-sub-metric">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="sub-metric-label">Price Edge vs. Naive</span>
                <HelpTooltip term="naive_baseline" />
              </div>
              <div className="sub-metric-bar-bg">
                <div
                  className="sub-metric-bar-fill"
                  style={{
                    width: `${capabilityScore.components?.price_precision_vs_naive_score || 0}%`,
                    backgroundColor: '#10b981',
                  }}
                />
              </div>
              <span className="sub-metric-value">
                {capabilityScore.components?.price_precision_vs_naive_score || 0} / 100
                <small> (MAE: ₹{capabilityScore.raw_inputs?.model_mae} vs ₹{capabilityScore.raw_inputs?.naive_mae})</small>
              </span>
            </div>

            <div className="capability-sub-metric">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="sub-metric-label">Regime & Recall Balance</span>
                <HelpTooltip term="up_recall" />
              </div>
              <div className="sub-metric-bar-bg">
                <div
                  className="sub-metric-bar-fill"
                  style={{
                    width: `${capabilityScore.components?.regime_balance_score || 0}%`,
                    backgroundColor: '#f59e0b',
                  }}
                />
              </div>
              <span className="sub-metric-value">
                {capabilityScore.components?.regime_balance_score || 0} / 100
                <small> (UP: {capabilityScore.raw_inputs?.up_recall_pct}% / DN: {capabilityScore.raw_inputs?.down_recall_pct}%)</small>
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Model Coverage & Artifact Availability Toggle */}
      <div className="coverage-matrix-accordion">
        <button
          type="button"
          className="coverage-matrix-btn"
          onClick={() => setShowMatrix(!showMatrix)}
        >
          <div className="coverage-btn-title">
            <Layers size={16} />
            <span>Multi-Stock Model Coverage & Isolation Matrix (NSE Top 5)</span>
          </div>
          <span className="coverage-status-tag">
            {showMatrix ? 'Hide Coverage Grid ▲' : 'View Authorized Models ▼'}
          </span>
        </button>

        {showMatrix && (
          <div className="coverage-matrix-content">
            <div className="matrix-warning-box">
              <ShieldAlert size={16} />
              <span>
                <strong>Cross-Asset Isolation Guarantee:</strong> EXP-05 and EXP-06 research challenger
                models (GRU V2) were trained strictly on RELIANCE.NS. In accordance with quantitative standards,
                Reliance weights and scalers are <strong>never</strong> applied to TCS, INFY, or bank equities.
              </span>
            </div>

            <div className="accuracy-table-wrapper">
              <table className="accuracy-table matrix-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Company Name</th>
                    <th>Production CNN-LSTM V1</th>
                    <th>EXP-05 Walk-Forward</th>
                    <th>GRU V2 Challenger</th>
                    <th>Archived Test Rows</th>
                    <th>Research Status</th>
                  </tr>
                </thead>
                <tbody>
                  {coverageMatrix.map((item) => (
                    <tr key={item.symbol} className={item.symbol === symbol ? 'highlight-row' : ''}>
                      <td>
                        <strong>{item.symbol}</strong>
                      </td>
                      <td>{item.name}</td>
                      <td>
                        <span className="matrix-status-yes">✓ Supported</span>
                      </td>
                      <td>
                        {item.exp05_walkforward ? (
                          <span className="matrix-status-yes">✓ Available</span>
                        ) : (
                          <span className="matrix-status-no">— Restricted</span>
                        )}
                      </td>
                      <td>
                        {item.gru_v2_challenger ? (
                          <span className="matrix-status-yes">✓ Active Challenger</span>
                        ) : (
                          <span className="matrix-status-no">— Not Trained</span>
                        )}
                      </td>
                      <td>
                        <span className="matrix-count-tag">{item.total_archived_predictions} rows</span>
                      </td>
                      <td>
                        <small className="matrix-notes">{item.evaluation_status}</small>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Filter Toolbar */}
      <div className="historical-filter-toolbar">
        <div className="filter-item">
          <label htmlFor="model-select">Model Architecture:</label>
          <select
            id="model-select"
            className="filter-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {(data?.available_models || []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-item">
          <label htmlFor="fold-select">Test Fold:</label>
          <select
            id="fold-select"
            className="filter-select"
            value={selectedFold}
            onChange={(e) => setSelectedFold(e.target.value)}
          >
            <option value="ALL">All Folds Combined</option>
            {(data?.available_folds || []).map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-item">
          <label htmlFor="experiment-select">Experiment:</label>
          <select
            id="experiment-select"
            className="filter-select"
            value={selectedExperiment}
            onChange={(e) => setSelectedExperiment(e.target.value)}
          >
            <option value="ALL">All Experiments</option>
            {(data?.available_experiments || []).map((exp) => (
              <option key={exp} value={exp}>
                {exp}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          className="historical-refresh-btn"
          onClick={fetchHistoricalData}
          disabled={loading}
          aria-label="Refresh historical test predictions"
        >
          <RotateCw size={13} className={loading ? 'accuracy-spinning' : ''} />
          {loading ? 'Refreshing…' : 'Update Slice'}
        </button>
      </div>

      {error && (
        <div className="accuracy-banner accuracy-banner-error" role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>Unable to Load Historical Predictions:</strong> {error}
          </div>
          <button type="button" className="accuracy-retry-btn" onClick={fetchHistoricalData}>
            Retry
          </button>
        </div>
      )}

      {/* Statistics Scorecard Row */}
      <div className="historical-stats-grid">
        <div className="historical-stat-card">
          <span className="stat-label">Evaluated Historical Sessions</span>
          <div className="stat-val-row">
            <span className="stat-val">{metrics.sample_size || 0}</span>
            <span className="stat-unit">test dates</span>
          </div>
          <span className="stat-desc">Zero training data; strictly out-of-sample</span>
        </div>

        <div className="historical-stat-card">
          <span className="stat-label">Price MAE (Mean Absolute Error)</span>
          <div className="stat-val-row">
            <span className="stat-val">
              {metrics.price_mae !== null && metrics.price_mae !== undefined
                ? `₹${metrics.price_mae.toFixed(2)}`
                : '—'}
            </span>
          </div>
          <span className="stat-desc">
            Naive Baseline MAE: ₹{data?.naive_comparison_metrics?.price_mae || '—'}
          </span>
        </div>

        <div className="historical-stat-card">
          <span className="stat-label">Directional Hit Rate</span>
          <div className="stat-val-row">
            <span
              className={`stat-val ${
                metrics.directional_accuracy_pct >= 50 ? 'accent-green' : 'accent-amber'
              }`}
            >
              {metrics.directional_accuracy_pct !== null && metrics.directional_accuracy_pct !== undefined
                ? `${metrics.directional_accuracy_pct.toFixed(1)}%`
                : '—'}
            </span>
          </div>
          <span className="stat-desc">
            Correctly anticipated sign(r̂) == sign(r)
          </span>
        </div>

        <div className="historical-stat-card">
          <span className="stat-label">Matthews Correlation (MCC)</span>
          <div className="stat-val-row">
            <span
              className={`stat-val ${
                metrics.mcc > 0 ? 'accent-green' : metrics.mcc < 0 ? 'accent-red' : ''
              }`}
            >
              {metrics.mcc !== null && metrics.mcc !== undefined
                ? (metrics.mcc > 0 ? `+${metrics.mcc.toFixed(4)}` : `${metrics.mcc.toFixed(4)}`)
                : 'N/A'}
            </span>
          </div>
          <span className="stat-desc">Balanced binary directional correlation</span>
        </div>

        <div className="historical-stat-card">
          <span className="stat-label">UP vs. DOWN Recall</span>
          <div className="stat-val-row">
            <span className="stat-val text-sm">
              UP: {metrics.up_recall_pct ?? '—'}% · DN: {metrics.down_recall_pct ?? '—'}%
            </span>
          </div>
          <span className="stat-desc">Directional asymmetry breakdown</span>
        </div>
      </div>

      {/* Historical Predicted vs Actual Interactive Timeline */}
      <div className="historical-chart-card">
        <div className="chart-card-header">
          <div>
            <h4>Actual vs. Predicted Closing Prices (Walk-Forward Test Timeline)</h4>
            <p>
              Displaying {chartData.length} genuine historical out-of-sample predictions.
              Gaps between non-contiguous fold slices are accurately preserved.
            </p>
          </div>
          <div className="chart-legend-pills">
            <span className="legend-pill" style={{ color: '#10b981' }}>
              ● Actual Historical Close
            </span>
            <span className="legend-pill" style={{ color: '#0ea5e9' }}>
              -- {selectedModel} Forecast
            </span>
          </div>
        </div>

        {chartData.length > 0 ? (
          <div className="historical-chart-container" style={{ width: '100%', height: 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.35} />
                <XAxis dataKey="target_date" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <YAxis
                  domain={['auto', 'auto']}
                  stroke="#94a3b8"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(val) => `₹${val}`}
                />
                <Tooltip content={<HistoricalChartTooltip modelName={selectedModel} />} />
                <Line
                  type="monotone"
                  dataKey="actual_close"
                  name="Actual Market Close"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#10b981' }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="predicted_close"
                  name="Predicted Close"
                  stroke="#0ea5e9"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 2.5, fill: '#0ea5e9' }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="chart-empty-placeholder">
            <Info size={32} />
            <p>No historical prediction observations match the selected criteria.</p>
          </div>
        )}
      </div>

      {/* Residual Distribution & Error Histogram */}
      {residualHistogram.length > 0 && (
        <div className="historical-residual-card">
          <div className="residual-header">
            <div>
              <h4>Residual & Error Distribution Histogram</h4>
              <p>Frequency distribution of prediction residuals: (Predicted Price − Actual Price) in Rupees.</p>
            </div>
            <div className="residual-summary-tags">
              <span>Mean Bias: <strong>{metrics.residuals?.mean > 0 ? `+₹${metrics.residuals.mean}` : `₹${metrics.residuals?.mean}`}</strong></span>
              <span>Std Dev: <strong>₹{metrics.residuals?.std}</strong></span>
              <span>Range: <strong>[₹{metrics.residuals?.min}, ₹{metrics.residuals?.max}]</strong></span>
            </div>
          </div>

          <div style={{ width: '100%', height: 200, marginTop: 12 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={residualHistogram} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                <XAxis dataKey="range" stroke="#94a3b8" tick={{ fontSize: 10 }} interval={0} angle={-25} textAnchor="end" />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(val) => [`${val} sessions`, 'Count']}
                  labelFormatter={(lbl) => `Residual Range: ${lbl}`}
                />
                <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Historical Prediction Table */}
      <div className="historical-table-section">
        <div className="table-header-row">
          <div>
            <h4>Test-Fold Prediction Audit Log</h4>
            <p>
              Showing {filteredRecords.length} historical test observations for {selectedModel} on {symbol}.
            </p>
          </div>

          <div className="table-search-box">
            <Search size={14} />
            <input
              type="text"
              placeholder="Search target date or fold…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="table-search-input"
            />
          </div>
        </div>

        <div className="accuracy-table-wrapper sticky-table-wrapper">
          <table className="accuracy-table historical-detail-table">
            <thead>
              <tr>
                <th className="sticky-col col-date">
                  Target Date
                  <HelpTooltip title="Target Date" text="The historical date that was predicted out-of-sample." />
                </th>
                <th className="sticky-col col-model">Model</th>
                <th>
                  Fold
                  <HelpTooltip title="Time-Slice Fold" text="The chronological walk-forward fold (e.g. Fold 1, 2, or 3) testing past historical epochs without peeking." />
                </th>
                <th>
                  Reference Close
                  <HelpTooltip title="Previous Close" text="The market price the day before the target date." />
                </th>
                <th>
                  Predicted Close
                  <HelpTooltip title="Predicted Close" text="Model's forecasted price for that historical day." />
                </th>
                <th>
                  Actual Close
                  <HelpTooltip title="Actual Close" text="Real price observed in historical market records." />
                </th>
                <th>
                  Cash Error (₹)
                  <HelpTooltip term="mae" />
                </th>
                <th>
                  Percent Error (%)
                  <HelpTooltip term="bps" />
                </th>
                <th>Pred Dir</th>
                <th>Actual Dir</th>
                <th>
                  Result
                  <HelpTooltip term="win_rate" />
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map((r, i) => {
                const isCorrect = r.direction_correct;
                return (
                  <tr key={r.record_id || `${r.target_date}-${i}`}>
                    <td className="sticky-col col-date">
                      <span className="table-date">{r.target_date}</span>
                    </td>
                    <td className="sticky-col col-model">
                      <span className="model-chip" style={{ borderColor: r.model_color }}>
                        {r.model_name}
                      </span>
                    </td>
                    <td>
                      <span className="fold-tag">{r.fold}</span>
                    </td>
                    <td>₹{r.previous_close?.toFixed(2)}</td>
                    <td>
                      <strong>₹{r.predicted_close?.toFixed(2)}</strong>
                    </td>
                    <td>
                      <span className="actual-price-cell">₹{r.actual_close?.toFixed(2)}</span>
                    </td>
                    <td>
                      <span className="error-cell">₹{r.absolute_error?.toFixed(2)}</span>
                    </td>
                    <td>{r.percentage_error !== null ? `${r.percentage_error.toFixed(2)}%` : '—'}</td>
                    <td>
                      <span className={`dir-tag ${r.predicted_direction?.toLowerCase()}`}>
                        {r.predicted_direction}
                      </span>
                    </td>
                    <td>
                      <span className={`dir-tag ${r.actual_direction?.toLowerCase()}`}>
                        {r.actual_direction}
                      </span>
                    </td>
                    <td>
                      {isCorrect ? (
                        <span className="outcome-tag outcome-correct">
                          <CheckCircle2 size={12} />
                          CORRECT
                        </span>
                      ) : (
                        <span className="outcome-tag outcome-incorrect">
                          <XCircle size={12} />
                          INCORRECT
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {filteredRecords.length === 0 && (
                <tr>
                  <td colSpan="11" className="empty-table-cell">
                    No historical prediction records match the query.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function HistoricalChartTooltip({ active, payload, label, modelName }) {
  if (!active || !payload || !payload.length) return null;
  const pData = payload[0]?.payload || {};

  return (
    <div className="accuracy-chart-tooltip">
      <div className="tooltip-header">
        <span className="tooltip-date">Target Date: {label}</span>
        <span className="tooltip-fold-tag">{pData.fold}</span>
      </div>

      <div className="tooltip-body">
        <div className="tooltip-row actual-row">
          <span>Actual Market Close:</span>
          <strong>₹{pData.actual_close?.toFixed(2)}</strong>
        </div>

        <div className="tooltip-row">
          <span style={{ color: '#0ea5e9' }}>{modelName}:</span>
          <strong>
            ₹{pData.predicted_close?.toFixed(2)} (Err: ₹{pData.absolute_error?.toFixed(2)})
          </strong>
        </div>

        <div className="tooltip-row">
          <span>Direction Match:</span>
          <strong style={{ color: pData.direction_correct ? '#10b981' : '#f43f5e' }}>
            {pData.direction_correct ? 'Correct' : 'Incorrect'}
          </strong>
        </div>
      </div>
    </div>
  );
}

