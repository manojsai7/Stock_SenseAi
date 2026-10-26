import React, { useCallback, useEffect, useState } from 'react';
import { Activity, AlertCircle, RefreshCw, ShieldAlert } from 'lucide-react';
import { getFeatureSensitivity } from '../services/api';
import './FeatureExplainabilityPanel.css';

const FEATURE_LABELS = {
  return_1d: '1-day return',
  return_5d: '5-day return',
  return_20d: '20-day return',
  intraday_return: 'Intraday return',
  hl_range_ratio: 'High-low range ratio',
  close_to_ma20: 'Close vs. MA20',
  volume_ratio: 'Volume ratio',
};

const signed = (value, digits = 2) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '—';
  return `${numeric > 0 ? '+' : ''}${numeric.toFixed(digits)}`;
};

const price = (value) => Number.isFinite(Number(value))
  ? `₹${Number(value).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  : '—';

function directionLabel(value) {
  if (value === 'BULLISH') return 'Up';
  if (value === 'BEARISH') return 'Down';
  return 'Flat';
}

export default function FeatureExplainabilityPanel({ symbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadSensitivity = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const result = await getFeatureSensitivity(symbol);
      setData(result);
    } catch (err) {
      setData(null);
      setError(err?.message || 'Feature sensitivity is currently unavailable.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [symbol]);

  useEffect(() => {
    let active = true;
    setData(null);
    setLoading(true);
    setError(null);
    getFeatureSensitivity(symbol)
      .then((result) => { if (active) setData(result); })
      .catch((err) => { if (active) setError(err?.message || 'Feature sensitivity is currently unavailable.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [symbol]);

  const features = data?.feature_sensitivities || [];
  const maxMagnitude = Math.max(0.01, ...features.map((item) => Math.abs(Number(item.delta_return_bps) || 0)));

  return (
    <section className="fintech-card explainability-card" aria-labelledby="feature-sensitivity-title" aria-busy={loading || refreshing}>
      <div className="explainability-header">
        <div>
          <div className="explainability-eyebrow"><Activity size={14} /> MODEL DIAGNOSTICS</div>
          <h2 id="feature-sensitivity-title">Local input sensitivity</h2>
          <p>How does the frozen model's next-session forecast change when one input feature is neutralized?</p>
        </div>
        <button
          type="button"
          className="btn-secondary explainability-refresh"
          onClick={() => loadSensitivity(true)}
          disabled={loading || refreshing}
        >
          <RefreshCw size={14} className={refreshing ? 'explainability-spinning' : ''} />
          {refreshing ? 'Refreshing…' : 'Refresh analysis'}
        </button>
      </div>

      {error && (
        <div className="explainability-error" role="alert">
          <AlertCircle size={18} />
          <div><strong>Analysis unavailable</strong><p>{error}</p><button type="button" className="btn-secondary" onClick={() => loadSensitivity(true)} disabled={refreshing}>Try again</button></div>
        </div>
      )}

      {loading && !data && !error && <div className="explainability-empty" role="status">Fetching completed market data and calculating sensitivity…</div>}

      {data && (
        <>
          <div className="explainability-meta-grid">
            <div><span>Symbol</span><strong>{data.symbol}</strong></div>
            <div><span>Input window</span><strong>{data.input_window_start} → {data.input_window_end}</strong></div>
            <div><span>Data status</span><strong className={`explainability-status status-${String(data.data_status || 'unknown').toLowerCase()}`}>{data.data_status || 'UNKNOWN'}</strong></div>
            <div><span>Training data cutoff</span><strong>{data.model_training_cutoff || 'Not recorded'}</strong></div>
          </div>

          <div className="explainability-baseline">
            <div><span>Frozen model baseline</span><strong>{price(data.baseline_predicted_close)}</strong><small>{signed(Number(data.baseline_predicted_return) * 100, 4)}% · {directionLabel(data.baseline_direction)}</small></div>
            <div><span>Last completed close</span><strong>{price(data.last_close)}</strong><small>Target session: {data.target_date}</small></div>
            <div><span>Provider</span><strong>{data.data_provider_source || 'Unspecified'}</strong><small>Data as of {data.data_as_of_date || data.input_window_end}</small></div>
          </div>

          <div className="explainability-table" role="table" aria-label="Feature neutralization sensitivity">
            <div className="explainability-table-head" role="row">
              <span role="columnheader">Feature</span>
              <span role="columnheader">Return shift</span>
              <span role="columnheader">Price shift</span>
              <span role="columnheader">Direction changed</span>
            </div>
            {features.map((item) => {
              const delta = Number(item.delta_return_bps) || 0;
              const width = Math.max(2, Math.min(100, Math.abs(delta) / maxMagnitude * 100));
              return (
                <div className="explainability-row" role="row" key={item.feature}>
                  <div className="explainability-feature" role="cell">
                    <strong>{FEATURE_LABELS[item.feature] || item.feature}</strong>
                    <code>{item.feature}</code>
                    <small>Latest value: {signed(item.latest_session_value, 5)}</small>
                  </div>
                  <div className="explainability-shift" role="cell">
                    <strong className={delta > 0 ? 'shift-positive' : delta < 0 ? 'shift-negative' : ''}>{signed(delta)} bps</strong>
                    <div className="explainability-track"><span className={delta < 0 ? 'shift-fill-negative' : 'shift-fill-positive'} style={{ width: `${width}%` }} /></div>
                  </div>
                  <div role="cell" className="explainability-price-shift">₹{signed(item.delta_predicted_close)}</div>
                  <div role="cell"><span className={`explainability-flip ${item.direction_flipped ? 'flip-yes' : 'flip-no'}`}>{item.direction_flipped ? 'Yes' : 'No'}</span><small>{directionLabel(item.direction_after_neutralization)}</small></div>
                </div>
              );
            })}
          </div>

          <div className="explainability-method-note">
            <ShieldAlert size={17} />
            <p><strong>Interpret carefully.</strong> {data.reference_definition} The feature is neutralized across all 30 sessions, not just the latest row. This measures sensitivity of one forecast to an input perturbation; it does not establish causation, feature value on other dates, or predictive skill.</p>
          </div>
          <div className="explainability-footer">Method: {data.method}. Pipeline: {data.pipeline_version}. Model: {data.model_version}.</div>
        </>
      )}
    </section>
  );
}
