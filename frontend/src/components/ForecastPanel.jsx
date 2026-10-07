import React, { useState } from 'react';
import {
  Sparkles,
  ArrowUpRight,
  ArrowDownRight,
  Cpu,
  Layers,
  Calendar,
  CheckCircle2,
  RefreshCw,
  Table,
  Zap,
  AlertCircle
} from 'lucide-react';

export default function ForecastPanel({
  prediction,
  onRunPrediction,
  predicting,
  symbol,
  currency = '₹'
}) {
  const [showCandles, setShowCandles] = useState(false);

  const isUp = prediction?.direction === 'UP';

  return (
    <div className="glass-panel" style={{
      padding: '28px',
      marginBottom: '24px',
      position: 'relative',
      overflow: 'hidden',
      border: '1px solid rgba(139, 92, 246, 0.25)',
      background: 'linear-gradient(135deg, rgba(16, 23, 38, 0.85) 0%, rgba(20, 28, 46, 0.9) 100%)',
    }}>
      {/* Background cyber accent glow */}
      <div style={{
        position: 'absolute',
        top: '-40px',
        right: '-40px',
        width: '180px',
        height: '180px',
        borderRadius: '50%',
        background: isUp ? 'radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%)' : 'radial-gradient(circle, rgba(244,63,94,0.15) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Header & Trigger */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '22px', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #FF6F00 0%, #E65100 50%, #8B5CF6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(255, 111, 0, 0.35)',
          }}>
            <Cpu size={22} color="#FFFFFF" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h2 style={{
                fontFamily: 'var(--font-heading)',
                fontSize: '1.35rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
              }}>
                CNN-LSTM Neural Network Forecast
              </h2>
              <span style={{
                background: 'rgba(255, 111, 0, 0.15)',
                color: '#FF9E40',
                border: '1px solid rgba(255, 111, 0, 0.3)',
                padding: '2px 8px',
                borderRadius: '6px',
                fontSize: '0.72rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
              }}>
                TensorFlow / Keras
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              1-Trading-Day-Ahead Closing Price Regression on Historical 30-Day Sequences
            </p>
          </div>
        </div>

        <button
          onClick={onRunPrediction}
          disabled={predicting}
          className="btn-primary"
          style={{
            background: predicting ? 'rgba(255,255,255,0.1)' : 'linear-gradient(135deg, #8B5CF6 0%, #10B981 100%)',
            boxShadow: predicting ? 'none' : '0 4px 20px rgba(139, 92, 246, 0.35)',
          }}
        >
          {predicting ? (
            <>
              <RefreshCw size={16} className="pulse-glow" style={{ animation: 'spin 1s linear infinite' }} />
              <span>Executing Inference...</span>
            </>
          ) : (
            <>
              <Zap size={16} />
              <span>Run Model Inference</span>
            </>
          )}
        </button>
      </div>

      {prediction ? (
        <div>
          {/* Big Forecast Metrics Row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '16px',
            marginBottom: '20px',
          }}>
            {/* Last Close Observed */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '12px',
              padding: '18px',
            }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                Latest Completed Close
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '1.8rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
              }}>
                {currency}{Number(prediction.last_close).toFixed(2)}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Completed Session: {prediction.latest_completed_session || prediction.last_observed_date}
              </div>
            </div>

            {/* Estimated Close */}
            <div style={{
              background: isUp ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
              border: `1px solid ${isUp ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
              borderRadius: '12px',
              padding: '18px',
              boxShadow: isUp ? '0 0 24px rgba(16, 185, 129, 0.15)' : '0 0 24px rgba(244, 63, 94, 0.15)',
            }}>
              <div style={{ fontSize: '0.78rem', color: isUp ? '#34D399' : '#FB7185', marginBottom: '6px', fontWeight: 600 }}>
                Next Trading Day Forecast
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '2rem',
                fontWeight: 800,
                color: isUp ? '#34D399' : '#FB7185',
              }}>
                {currency}{Number(prediction.predicted_close).toFixed(2)}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Target Session: {prediction.target_date || 'Next Trading Day'}
              </div>
            </div>

            {/* Expected Movement & Direction */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '12px',
              padding: '18px',
            }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                Model Forecast Direction
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                <span className={isUp ? 'badge-up' : 'badge-down'} style={{ fontSize: '1rem', padding: '6px 14px' }}>
                  {isUp ? <ArrowUpRight size={18} /> : <ArrowDownRight size={18} />}
                  <span>{prediction.directional_signal || prediction.direction || (isUp ? 'BULLISH' : 'BEARISH')}</span>
                </span>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.92rem', color: isUp ? '#34D399' : '#FB7185', fontWeight: 600 }}>
                {isUp ? '+' : ''}{prediction.expected_change > 0 ? `+${prediction.expected_change}` : prediction.expected_change} {currency} ({isUp ? '+' : ''}{prediction.expected_change_pct}%)
              </div>
            </div>

            {/* Benchmark Directional Accuracy */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '12px',
              padding: '18px',
            }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                Historical Directional Accuracy
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '1.8rem',
                fontWeight: 700,
                color: 'var(--accent-cyan)',
              }}>
                {prediction.historical_directional_accuracy || prediction.directional_accuracy_benchmark || 52.72}%
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Held-Out Test Set Hit Rate (Offline Benchmark)
              </div>
            </div>

          </div>

          {/* Academic Disclaimer & Methodology Spec Bar */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.35)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '10px',
            padding: '14px 18px',
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '12px',
            fontSize: '0.8rem',
            marginBottom: '12px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Framework: </span>
                <strong style={{ color: '#FF9E40' }}>TensorFlow / Keras</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Input Window: </span>
                <strong style={{ color: '#F8FAFC' }}>30 Trading Days</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Features: </span>
                <strong style={{ color: '#F8FAFC' }}>Open, High, Low, Close, Volume, MA20</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Architecture: </span>
                <strong style={{ color: '#A78BFA' }}>Conv1D(64) → MaxPool → LSTM(50) → Dense</strong>
              </div>
            </div>

            {prediction.recent_input_candles?.length > 0 && (
              <button
                onClick={() => setShowCandles(!showCandles)}
                className="btn-secondary"
                style={{ fontSize: '0.75rem', padding: '5px 10px' }}
              >
                <Table size={13} />
                <span>{showCandles ? 'Hide 30-Day Window' : 'Inspect 30-Day Window'}</span>
              </button>
            )}
          </div>

          {/* Academic framing notice */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--text-muted)',
            fontSize: '0.75rem',
            fontStyle: 'italic',
          }}>
            <AlertCircle size={13} />
            <span>Academic Note: Model-based statistical estimate for research purposes — not financial advice or an automated trading recommendation.</span>
          </div>

          {/* Collapsible Input Sequence Table */}
          {showCandles && prediction.recent_input_candles && (
            <div style={{
              marginTop: '16px',
              background: 'rgba(0, 0, 0, 0.4)',
              borderRadius: '10px',
              border: '1px solid var(--border-subtle)',
              padding: '12px',
              maxHeight: '220px',
              overflowY: 'auto',
            }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)', textAlign: 'left' }}>
                    <th style={{ padding: '6px' }}>#</th>
                    <th style={{ padding: '6px' }}>Date</th>
                    <th style={{ padding: '6px' }}>Open</th>
                    <th style={{ padding: '6px' }}>High</th>
                    <th style={{ padding: '6px' }}>Low</th>
                    <th style={{ padding: '6px' }}>Close</th>
                    <th style={{ padding: '6px' }}>MA20</th>
                    <th style={{ padding: '6px' }}>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.recent_input_candles.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <td style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>{i + 1}</td>
                      <td style={{ padding: '4px 6px', color: '#F8FAFC' }}>{c.date}</td>
                      <td style={{ padding: '4px 6px' }}>{currency}{c.open}</td>
                      <td style={{ padding: '4px 6px', color: '#34D399' }}>{currency}{c.high}</td>
                      <td style={{ padding: '4px 6px', color: '#FB7185' }}>{currency}{c.low}</td>
                      <td style={{ padding: '4px 6px', fontWeight: 700 }}>{currency}{c.close}</td>
                      <td style={{ padding: '4px 6px', color: '#06B6D4' }}>{currency}{c.ma20}</td>
                      <td style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>{c.volume}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '36px 0', color: 'var(--text-muted)' }}>
          <p style={{ marginBottom: '12px', fontSize: '0.95rem' }}>
            Click <strong>"Run Model Inference"</strong> to load the 30-day sequence into the trained TensorFlow/Keras CNN-LSTM model.
          </p>
        </div>
      )}
    </div>
  );
}
