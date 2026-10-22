import React, { lazy, Suspense, useEffect, useRef, useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Table,
  Cpu,
  AlertCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { triggerHaptic } from '../lib/haptics';

const ParticleButton = lazy(() => import('./kokonutui/particle-button'));

export default function ForecastPanel({
  prediction,
  onRunPrediction,
  predicting = false,
  error = null,
  onRetry,
  symbol,
  currency = '₹'
}) {
  const [showCandles, setShowCandles] = useState(false);
  const [showModelSpecs, setShowModelSpecs] = useState(false);
  const [forecastButtonStatus, setForecastButtonStatus] = useState('idle');
  const forecastStatusTimer = useRef(null);

  useEffect(() => () => clearTimeout(forecastStatusTimer.current), []);
  useEffect(() => {
    clearTimeout(forecastStatusTimer.current);
    setForecastButtonStatus('idle');
  }, [symbol]);

  const runForecastFromButton = async () => {
    if (predicting) return;
    clearTimeout(forecastStatusTimer.current);
    triggerHaptic();
    setForecastButtonStatus('pending');
    const succeeded = await onRunPrediction();
    setForecastButtonStatus(succeeded ? 'success' : 'error');
    triggerHaptic(succeeded ? 'success' : 'error');
    forecastStatusTimer.current = setTimeout(() => setForecastButtonStatus('idle'), succeeded ? 1800 : 2600);
  };

  // Directional signal determination
  const direction = prediction?.directional_signal || prediction?.direction || 'FLAT';
  const isUp = direction === 'BULLISH' || direction === 'UP';
  const isDown = direction === 'BEARISH' || direction === 'DOWN';

  const formatPrice = (val) => {
    if (val === null || val === undefined || isNaN(Number(val))) return '—';
    return `${currency}${Number(val).toLocaleString('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  };

  return (
    <div className="fintech-card forecast-card" style={{ padding: '24px', marginBottom: '20px' }}>
      {/* Forecast Header & Action */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '3px' }}>
            <h2 style={{
              fontSize: '1rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.01em',
              margin: 0
            }}>
              Next-Day Model Forecast
            </h2>
            <span style={{
              backgroundColor: 'var(--surface-subtle)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.6875rem',
              fontWeight: 600,
              padding: '1px 6px',
              borderRadius: '4px'
            }}>
              1-Day Horizon
            </span>
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            Regression forecast for next trading session based on historical 30-day sequence.
          </p>
        </div>

        {/* Generate / Run Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Suspense fallback={<button className="btn-primary particle-forecast-button" type="button" disabled>Preparing forecast control…</button>}>
            <ParticleButton
              className="btn-primary particle-forecast-button"
              status={forecastButtonStatus}
              disabled={predicting}
              onClick={runForecastFromButton}
              pendingLabel="Calculating forecast"
              successLabel="Forecast received"
              errorLabel="Forecast unavailable"
              particles
              icon={<RefreshCw size={14} />}
            >
              {prediction ? 'Recalculate forecast' : 'Generate forecast'}
            </ParticleButton>
          </Suspense>
        </div>
      </div>

      {/* State 1: Localized Error State (e.g. ICICIBANK.NS or server error) */}
      {error ? (
        <div style={{
          backgroundColor: 'var(--surface-subtle)',
          borderRadius: '8px',
          border: '1px solid var(--border-subtle)',
          padding: '20px',
          marginBottom: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
            <AlertCircle size={20} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                Forecast Unavailable for {symbol}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '12px' }}>
                {error}
              </div>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="btn-secondary"
                  disabled={predicting}
                  style={{ fontSize: '0.78125rem' }}
                >
                  <RefreshCw size={13} />
                  <span>Retry Calculation</span>
                </button>
              )}
            </div>
          </div>
        </div>
      ) : predicting && !prediction ? (
        /* State 2: Progress Skeleton when running first inference */
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px',
          marginBottom: '20px'
        }}>
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="skeleton-shimmer" style={{ height: '90px', borderRadius: '8px' }} />
          ))}
        </div>
      ) : prediction ? (
        /* State 3: Successful Forecast Display */
        <div>
          <div
            role="status"
            style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '6px',
              padding: '9px 12px',
              marginBottom: '14px',
              color: 'var(--text-secondary)',
              fontSize: '0.75rem',
              lineHeight: 1.5
            }}
          >
            <strong style={{ color: 'var(--text-primary)' }}>Forecast input data:</strong>{' '}
            {prediction.data_status === 'ONLINE'
              ? 'Retrieved online'
              : prediction.data_status === 'OFFLINE'
                ? 'Packaged historical snapshot'
                : 'Source not verified'}
            {prediction.data_as_of_date ? ` · Completed session ${prediction.data_as_of_date}` : ''}
            {prediction.data_provider_source ? ` · ${prediction.data_provider_source}` : ''}
            {' · This is daily prediction input data; the quote above is provided separately.'}
          </div>

          {/* Financial Metrics Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
            gap: '12px',
            marginBottom: '18px'
          }}>
            {/* 1. Predicted Next Session Close */}
            <div style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px', fontWeight: 500 }}>
                Forecasted Close ({prediction.target_date || 'Next Session'})
              </div>
              <div className="tabular-num" style={{
                fontSize: '1.625rem',
                fontWeight: 800,
                color: isUp ? 'var(--primary-green)' : isDown ? 'var(--accent-red)' : 'var(--text-primary)'
              }}>
                {formatPrice(prediction.predicted_close)}
              </div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Target Session: {prediction.target_date || 'Next Trading Day'}
              </div>
            </div>

            {/* 2. Reference Completed Close */}
            <div style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px', fontWeight: 500 }}>
                Reference Session Close
              </div>
              <div className="tabular-num" style={{
                fontSize: '1.625rem',
                fontWeight: 700,
                color: 'var(--text-primary)'
              }}>
                {formatPrice(prediction.last_close)}
              </div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Session: {prediction.latest_completed_session || prediction.last_observed_date || 'Latest'}
              </div>
            </div>

            {/* 3. Expected Price Movement */}
            <div style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 500 }}>
                Expected Delta
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span className={isUp ? 'badge-up' : isDown ? 'badge-down' : 'badge-neutral'}>
                  {isUp ? <TrendingUp size={14} /> : isDown ? <TrendingDown size={14} /> : <Minus size={14} />}
                  <span>{direction}</span>
                </span>
              </div>
              <div className="tabular-num" style={{
                fontSize: '0.875rem',
                fontWeight: 600,
                color: isUp ? 'var(--primary-green)' : isDown ? 'var(--accent-red)' : 'var(--text-secondary)'
              }}>
                {Number(prediction.expected_change) > 0 ? '+' : ''}{Number(prediction.expected_change).toFixed(2)} {currency} ({Number(prediction.expected_change_pct) > 0 ? '+' : ''}{Number(prediction.expected_change_pct).toFixed(2)}%)
              </div>
            </div>

            {/* 4. Directional Accuracy Benchmark */}
            <div style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px', fontWeight: 500 }}>
                Historical Directional Accuracy
              </div>
              <div className="tabular-num" style={{
                fontSize: prediction.historical_directional_accuracy != null ? '1.625rem' : '1.25rem',
                fontWeight: 800,
                color: prediction.historical_directional_accuracy != null ? 'var(--primary-green)' : 'var(--text-muted)'
              }}>
                {prediction.historical_directional_accuracy != null
                  ? `${prediction.historical_directional_accuracy}%`
                  : prediction.directional_accuracy_benchmark != null
                  ? `${prediction.directional_accuracy_benchmark}%`
                  : 'N/A'}
              </div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                {prediction.historical_directional_accuracy != null
                  ? 'Held-Out Chronological Test Set'
                  : 'Not evaluated on an untouched test set'}
              </div>
            </div>
          </div>

          {/* Model Specification Disclosure Strip */}
          <div style={{
            backgroundColor: 'var(--surface-subtle)',
            borderRadius: '6px',
            padding: '10px 14px',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            marginBottom: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '8px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                <span>Model: <strong>{prediction.model_version || 'v1.0'}</strong></span>
                <span>Architecture: <strong>{prediction.architecture || 'CNN-LSTM (Conv1D + LSTM)'}</strong></span>
                <span>Input: <strong>{prediction.input_window_size || 30} Completed Sessions</strong></span>
                <span>Features: <strong>{prediction.features_used?.length || 7} Features</strong></span>
              </div>

              {prediction.recent_input_candles?.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowCandles(!showCandles)}
                  className="btn-secondary"
                  style={{ padding: '3px 8px', fontSize: '0.71875rem' }}
                >
                  <Table size={12} />
                  <span>{showCandles ? 'Hide 30-Day Window' : 'Inspect 30-Day Window'}</span>
                </button>
              )}
            </div>

            {/* Pipeline Methodology Disclosure */}
            <div style={{
              fontSize: '0.6875rem',
              color: 'var(--text-muted)',
              borderTop: '1px solid var(--border-subtle)',
              paddingTop: '6px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '4px'
            }}>
              <span>
                {prediction.model_version === 'cnn_lstm_stationary_v1'
                  ? 'Methodology: Estimates next-session return (r̂) from 7 stationary relative features, then reconstructs price: Close × (1 + r̂).'
                  : 'Methodology: Direct price-target regression over 6 normalized OHLCV+MA20 features.'}
              </span>
              <span>
                Cutoff: <strong>{prediction.latest_completed_session || prediction.last_observed_date}</strong> → Target: <strong>{prediction.target_date}</strong>
              </span>
            </div>
          </div>

          {/* Collapsible 30-Day Input Sequence Table */}
          {showCandles && prediction.recent_input_candles && (
            <div className="table-scroll input-sequence-scroll" role="region" aria-label="Recent model input candles" tabIndex={0} style={{
              marginBottom: '14px',
              border: '1px solid var(--border-subtle)',
              borderRadius: '6px',
              overflow: 'hidden',
              maxHeight: '220px',
              overflowY: 'auto'
            }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
                <thead style={{ backgroundColor: 'var(--surface-subtle)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <tr>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>#</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>Date</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>Open</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>High</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>Low</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>Close</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>MA20</th>
                    <th style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.recent_input_candles.map((candle, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '5px 10px', color: 'var(--text-muted)' }}>{idx + 1}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px', fontWeight: 500 }}>{candle.date}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px' }}>{currency}{candle.open}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px', color: 'var(--primary-green)' }}>{currency}{candle.high}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px', color: 'var(--accent-red)' }}>{currency}{candle.low}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px', fontWeight: 600 }}>{currency}{candle.close}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px', color: 'var(--accent-blue)' }}>{currency}{candle.ma20}</td>
                      <td className="tabular-num" style={{ padding: '5px 10px', color: 'var(--text-muted)' }}>{Number(candle.volume || 0).toLocaleString('en-IN')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        /* State 4: Initial Prompt before inference */
        <div style={{
          backgroundColor: 'var(--surface-subtle)',
          borderRadius: '8px',
          border: '1px dashed var(--border-default)',
          padding: '24px',
          textAlign: 'center',
          color: 'var(--text-secondary)',
          fontSize: '0.875rem'
        }}>
          Click <strong>"Generate Forecast"</strong> above to run inference using the trained CNN-LSTM model.
        </div>
      )}

      {/* Mandatory Financial Disclaimer */}
      <div style={{
        marginTop: '12px',
        paddingTop: '10px',
        borderTop: '1px solid var(--border-subtle)',
        fontSize: '0.6875rem',
        color: 'var(--text-muted)',
        lineHeight: 1.4
      }}>
        Model estimate based on historical market data. It may differ materially from the actual closing price. Statistical forecast intended for academic research; not financial advice.
      </div>
    </div>
  );
}
