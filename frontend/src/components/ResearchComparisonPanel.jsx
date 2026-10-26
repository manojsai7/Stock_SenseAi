import React, { useEffect, useState, useCallback } from 'react';
import { getResearchModelComparison } from '../services/api';
import './ResearchComparisonPanel.css';

export default function ResearchComparisonPanel({ symbol = 'RELIANCE.NS' }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchComparison = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const res = await getResearchModelComparison(symbol);
      setData(res);
    } catch (err) {
      console.warn('Research comparison fetch failed:', err.message);
      setError(err.message || 'Unable to retrieve research comparison data');
      setData(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [symbol]);

  useEffect(() => {
    fetchComparison();
  }, [fetchComparison]);

  if (loading) {
    return (
      <section className="fintech-card research-panel-card" aria-busy="true" role="status">
        <div className="research-loading-state">
          <p>Loading prospective research model comparison for {symbol}…</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="fintech-card research-panel-card" role="region" aria-label="Research Comparison Error">
        <div className="research-error-state">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div>
            <strong>Unable to Load Research Model Comparison</strong>
            <p>{error}</p>
            <button className="secondary-button" onClick={() => fetchComparison(true)}>
              Retry
            </button>
          </div>
        </div>
      </section>
    );
  }

  // Unsupported stock check (e.g. TCS.NS, INFY.NS when research is locked to RELIANCE.NS)
  if (!data?.supported) {
    return (
      <section className="fintech-card research-panel-card" role="region" aria-label="Research Model Comparison">
        <div className="research-unsupported-card">
          <div className="research-unsupported-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            <span>Prospective Research Comparison: RELIANCE.NS Only</span>
          </div>
          <p>
            The live prospective evaluation protocol (EXP-05) is currently active and locked exclusively for{' '}
            <strong>RELIANCE.NS</strong>. Selected symbol <strong>{symbol}</strong> does not have an active prospective challenger evaluation stream.
          </p>
          <small>
            To prevent cross-symbol contamination, Colab checkpoints and prospective logs are evaluated strictly against their pre-registered equity. Switch to <strong>Reliance Industries</strong> in the equity selector to view live comparative records.
          </small>
        </div>
      </section>
    );
  }

  const { sample_counts, models = [], resolved_observations = [], pending_forecasts = [] } = data;

  return (
    <section className="fintech-card research-panel-card" role="region" aria-label="Research Model Comparison">
      {/* Header */}
      <div className="research-header">
        <div>
          <div className="research-eyebrow">
            <span>PROSPECTIVE EVALUATION · EXPERIMENT EXP-05</span>
          </div>
          <h2>Research Model Comparison & Evaluation</h2>
          <p>
            Head-to-head out-of-sample forward evaluation on Indian markets. Predictions are logged prior to market
            outcome and scored only after exchange session close.
          </p>
        </div>
        <button
          className="secondary-button research-refresh-btn"
          onClick={() => fetchComparison(true)}
          disabled={refreshing}
          aria-label="Refresh research evaluation data"
        >
          <span className={refreshing ? 'research-spinning' : ''}>↻</span> Refresh Log
        </button>
      </div>

      {/* Progress & Sample Size Counts */}
      <div className="research-counts-grid">
        <div className="research-count-box">
          <span>Target Scope</span>
          <strong>{data.target_sample_size || '60–100 Sessions'}</strong>
          <small>Formal Statistical Target</small>
        </div>
        <div className="research-count-box">
          <span>Resolved Sessions</span>
          <strong>{sample_counts.resolved_sessions} (N = {sample_counts.resolved_sessions})</strong>
          <small>{sample_counts.resolved_predictions} Scored Forecasts</small>
        </div>
        <div className="research-count-box">
          <span>Pending Forecasts</span>
          <strong>{sample_counts.pending_sessions} Session Active</strong>
          <small>{sample_counts.pending_predictions} Predictions Awaiting Close</small>
        </div>
        <div className="research-count-box">
          <span>Protocol Status</span>
          <strong>{data.status || 'LOCKED'}</strong>
          <small>Pre-Registered Weights</small>
        </div>
      </div>

      {/* Side-by-Side Model Summary Cards */}
      <div className="research-models-grid">
        {models.map((m) => {
          const roleClass =
            m.key === 'production_cnn_lstm'
              ? 'role-production'
              : m.key === 'colab_exp05_fold3'
              ? 'role-challenger'
              : 'role-baseline';

          const badgeClass =
            m.key === 'production_cnn_lstm'
              ? 'badge-production'
              : m.key === 'colab_exp05_fold3'
              ? 'badge-challenger'
              : 'badge-baseline';

          return (
            <div key={m.key} className={`research-model-card ${roleClass}`}>
              <div className="model-card-header">
                <span className={`model-badge ${badgeClass}`}>{m.role}</span>
                <span className="model-card-title">{m.name}</span>
              </div>

              <div className="model-card-meta">
                <div>Cutoff: <strong>{m.training_cutoff}</strong></div>
                {m.hash && m.hash !== 'zero_parameter_baseline' && (
                  <div>Hash: <code>{m.hash.substring(0, 16)}…</code></div>
                )}
              </div>

              <div className="model-metrics-box">
                <div className="metric-item">
                  <span>Price MAE</span>
                  <strong>{m.summary_metrics.price_mae != null ? `₹${m.summary_metrics.price_mae}` : 'N/A'}</strong>
                </div>
                <div className="metric-item">
                  <span>Dir Accuracy</span>
                  <strong>
                    {m.summary_metrics.directional_accuracy_pct != null
                      ? `${m.summary_metrics.directional_accuracy_pct}%`
                      : 'N/A'}
                  </strong>
                </div>
              </div>

              <div className="model-card-rule">
                <span>Rule: {m.decision_rule}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Stream 1: Pending Forecasts Awaiting Market Close */}
      {pending_forecasts.length > 0 && (
        <div className="stream-section">
          <div className="stream-title">
            <span>Pending Forward Forecast</span>
            <span className="status-tag tag-pending">Awaiting Session Outcome</span>
          </div>
          <div className="stream-table-wrapper">
            <table className="stream-table">
              <thead>
                <tr>
                  <th>Target Session</th>
                  <th>Input Cutoff</th>
                  <th>Baseline Close</th>
                  <th>Production Forecast</th>
                  <th>Colab Fold 3 Forecast</th>
                  <th>Naive Baseline</th>
                </tr>
              </thead>
              <tbody>
                {pending_forecasts.map((pf) => {
                  const prod = pf.predictions['Production CNN-LSTM'] || {};
                  const colab = pf.predictions['Colab EXP-05 Fold 3 (Calibrated)'] || {};
                  const naive = pf.predictions['Naive Persistence'] || {};

                  return (
                    <tr key={pf.target_date}>
                      <td><strong>{pf.target_date}</strong></td>
                      <td className="mono-num">{pf.input_window_end}</td>
                      <td className="mono-num">₹{pf.previous_close?.toFixed(2)}</td>
                      <td>
                        <span className="mono-num">₹{prod.predicted_close?.toFixed(2)}</span>{' '}
                        <span className={`dir-pill dir-${(prod.predicted_direction || '').toLowerCase()}`}>
                          {prod.predicted_direction}
                        </span>
                      </td>
                      <td>
                        <span className="mono-num">₹{colab.predicted_close?.toFixed(2)}</span>{' '}
                        <span className={`dir-pill dir-${(colab.predicted_direction || '').toLowerCase()}`}>
                          {colab.predicted_direction}
                        </span>
                      </td>
                      <td>
                        <span className="mono-num">₹{naive.predicted_close?.toFixed(2)}</span>{' '}
                        <span className="dir-pill dir-flat">FLAT</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Stream 2: Resolved Historical Observations */}
      {resolved_observations.length > 0 && (
        <div className="stream-section">
          <div className="stream-title">
            <span>Resolved Prospective Observation Stream</span>
            <span className="status-tag tag-resolved">Market Outcome Finalized</span>
          </div>
          <div className="stream-table-wrapper">
            <table className="stream-table">
              <thead>
                <tr>
                  <th>Target Session</th>
                  <th>Actual Close</th>
                  <th>Production CNN-LSTM</th>
                  <th>Colab Fold 3 (Challenger)</th>
                  <th>Naive Persistence</th>
                </tr>
              </thead>
              <tbody>
                {resolved_observations.map((ro) => {
                  const prod = ro.predictions['Production CNN-LSTM'] || {};
                  const colab = ro.predictions['Colab EXP-05 Fold 3 (Calibrated)'] || {};
                  const naive = ro.predictions['Naive Persistence'] || {};

                  return (
                    <tr key={ro.target_date}>
                      <td>
                        <strong>{ro.target_date}</strong>
                        <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>
                          Cutoff: {ro.input_window_end} (₹{ro.previous_close?.toFixed(2)})
                        </div>
                      </td>
                      <td>
                        <strong className="mono-num">₹{ro.actual_close?.toFixed(2)}</strong>{' '}
                        <span className={`dir-pill dir-${(ro.actual_direction || '').toLowerCase()}`}>
                          {ro.actual_direction}
                        </span>
                      </td>
                      <td>
                        <div className="mono-num">₹{prod.predicted_close?.toFixed(2)}</div>
                        <div>
                          Error: <span className="error-val error-lowest">₹{prod.price_error_abs?.toFixed(2)}</span>{' '}
                          <span className={`correct-badge ${prod.direction_correct ? 'correct-yes' : 'correct-no'}`}>
                            {prod.direction_correct ? '✓ Hit' : '✗ Miss'}
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="mono-num">₹{colab.predicted_close?.toFixed(2)}</div>
                        <div>
                          Error: <span className="error-val">₹{colab.price_error_abs?.toFixed(2)}</span>{' '}
                          <span className={`correct-badge ${colab.direction_correct ? 'correct-yes' : 'correct-no'}`}>
                            {colab.direction_correct ? '✓ Hit' : '✗ Miss'}
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="mono-num">₹{naive.predicted_close?.toFixed(2)}</div>
                        <div>
                          Error: <span className="error-val">₹{naive.price_error_abs?.toFixed(2)}</span>{' '}
                          <span className={`correct-badge ${naive.direction_correct ? 'correct-yes' : 'correct-no'}`}>
                            {naive.direction_correct ? '✓ Hit' : '✗ Miss'}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Scientific Integrity Disclaimer Banner */}
      <div className="research-disclaimer-box" role="note">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        <p>
          <strong>Methodological & Statistical Integrity Notice:</strong> On the resolved October 26 session, the production model achieved the lowest price error (₹11.57 vs ₹11.80 for naive persistence, Δ = ₹0.23). However, <strong>a single observation (N=1) carries zero statistical significance</strong> and cannot establish model superiority or predictive edge. Accumulating <strong>60 to 100 prospective sessions</strong> is required before running Diebold-Mariano hypothesis tests.
        </p>
      </div>
    </section>
  );
}
