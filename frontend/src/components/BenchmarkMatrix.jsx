import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';
import { Award, CheckCircle, Info, TrendingUp } from 'lucide-react';

export default function BenchmarkMatrix({ metricsData, symbol, currency = '₹' }) {
  if (!metricsData) return null;

  const comparison = metricsData.metrics_comparison || {};
  const trajectory = metricsData.test_trajectory || [];

  const naive = comparison.naive || {};
  const lstm = comparison.lstm || {};
  const cnnLstm = comparison.cnn_lstm || {};

  return (
    <div className="fintech-card benchmark-card" style={{ padding: '24px', marginBottom: '20px' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: '18px',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h2 style={{
              fontSize: '1rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.01em',
              margin: 0
            }}>
              Model Benchmark & Comparative Evaluation
            </h2>
            <span style={{
              backgroundColor: 'var(--green-tint)',
              color: 'var(--primary-green)',
              border: '1px solid var(--green-border)',
              fontSize: '0.6875rem',
              fontWeight: 600,
              padding: '1px 6px',
              borderRadius: '4px'
            }}>
              Held-Out Test Set (15%)
            </span>
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            Empirical evaluation of CNN-LSTM against Standard LSTM and Naive Persistence baselines on unseen chronological test observations.
          </p>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          backgroundColor: 'var(--surface-subtle)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '6px',
          padding: '4px 10px',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)'
        }}>
          <CheckCircle size={14} color="var(--primary-green)" />
          <span>Strict Zero Data Leakage (Scaler fitted on Train only)</span>
        </div>
      </div>

      {/* Research Context & Pipeline Distinction Banner */}
      <div style={{
        backgroundColor: 'var(--surface-subtle)',
        borderRadius: '6px',
        border: '1px solid var(--border-subtle)',
        padding: '8px 12px',
        marginBottom: '16px',
        fontSize: '0.75rem',
        color: 'var(--text-secondary)',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <Info size={15} color="var(--accent-blue)" style={{ flexShrink: 0 }} />
        <span>
          <strong>Research Benchmark Reference:</strong> The table below documents historical offline evaluation of the 6-feature Legacy Price-Target Model on a held-out test split. The active live forecasting pipeline uses the 7-feature Stationary-Return Model (Stage A validation selection; not presented as an untouched future test set).
        </span>
      </div>

      {/* Comparison Metrics Table */}
      <div className="table-scroll" role="region" aria-label={`${symbol} legacy benchmark comparison table`} tabIndex={0} style={{ overflowX: 'auto', marginBottom: '18px' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.8125rem',
          textAlign: 'left'
        }}>
          <thead>
            <tr style={{
              borderBottom: '1px solid var(--border-subtle)',
              backgroundColor: 'var(--surface-subtle)',
              color: 'var(--text-secondary)',
              fontSize: '0.6875rem',
              textTransform: 'uppercase',
              letterSpacing: '0.04em'
            }}>
              <th style={{ padding: '8px 12px' }}>Model</th>
              <th style={{ padding: '8px 12px' }}>Implementation</th>
              <th style={{ padding: '8px 12px', textAlign: 'right' }}>MAE ({currency}) ↓</th>
              <th style={{ padding: '8px 12px', textAlign: 'right' }}>RMSE ({currency}) ↓</th>
              <th style={{ padding: '8px 12px', textAlign: 'right' }}>MAPE (%) ↓</th>
              <th style={{ padding: '8px 12px', textAlign: 'right' }}>Directional Acc (%) ↑</th>
              <th style={{ padding: '8px 12px' }}>Role</th>
            </tr>
          </thead>
          <tbody>
            {/* Naive Persistence Baseline */}
            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <td style={{ padding: '10px 12px', fontWeight: 600 }}>Naive Persistence</td>
              <td style={{ padding: '10px 12px', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                ŷₜ₊₁ = Closeₜ
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right' }}>
                {naive.mae ? Number(naive.mae).toFixed(2) : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right' }}>
                {naive.rmse ? Number(naive.rmse).toFixed(2) : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right' }}>
                {naive.mape ? `${Number(naive.mape).toFixed(2)}%` : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600 }}>
                {naive.directional_accuracy ? `${Number(naive.directional_accuracy).toFixed(1)}%` : '—'}
              </td>
              <td style={{ padding: '10px 12px' }}>
                <span style={{ fontSize: '0.6875rem', backgroundColor: 'var(--surface-subtle)', color: 'var(--text-secondary)', padding: '2px 6px', borderRadius: '4px' }}>
                  Econometric Baseline
                </span>
              </td>
            </tr>

            {/* Standard LSTM */}
            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <td style={{ padding: '10px 12px', fontWeight: 600, color: 'var(--accent-blue)' }}>Standard LSTM</td>
              <td style={{ padding: '10px 12px', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                TensorFlow (50 units)
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right' }}>
                {lstm.mae ? Number(lstm.mae).toFixed(2) : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right' }}>
                {lstm.rmse ? Number(lstm.rmse).toFixed(2) : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right' }}>
                {lstm.mape ? `${Number(lstm.mape).toFixed(2)}%` : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: 'var(--accent-blue)' }}>
                {lstm.directional_accuracy ? `${Number(lstm.directional_accuracy).toFixed(1)}%` : '—'}
              </td>
              <td style={{ padding: '10px 12px' }}>
                <span style={{ fontSize: '0.6875rem', backgroundColor: 'var(--blue-tint)', color: 'var(--accent-blue)', padding: '2px 6px', borderRadius: '4px' }}>
                  Recurrent Baseline
                </span>
              </td>
            </tr>

            {/* Proposed CNN-LSTM */}
            <tr style={{
              backgroundColor: 'var(--green-tint)',
              borderBottom: '1px solid var(--green-border)'
            }}>
              <td style={{ padding: '10px 12px', fontWeight: 700, color: 'var(--primary-green)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Award size={15} color="var(--primary-green)" />
                <span>CNN-LSTM Hybrid</span>
              </td>
              <td style={{ padding: '10px 12px', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                Conv1D(64) + LSTM(50)
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: 'var(--primary-green)' }}>
                {cnnLstm.mae ? Number(cnnLstm.mae).toFixed(2) : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: 'var(--primary-green)' }}>
                {cnnLstm.rmse ? Number(cnnLstm.rmse).toFixed(2) : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: 'var(--primary-green)' }}>
                {cnnLstm.mape ? `${Number(cnnLstm.mape).toFixed(2)}%` : '—'}
              </td>
              <td className="tabular-num" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 800, color: 'var(--primary-green)', fontSize: '0.875rem' }}>
                {cnnLstm.directional_accuracy ? `${Number(cnnLstm.directional_accuracy).toFixed(1)}%` : '—'}
              </td>
              <td style={{ padding: '10px 12px' }}>
                <span className="badge-up" style={{ fontSize: '0.6875rem' }}>
                  Proposed Model
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Academic Findings Box */}
      <div style={{
        backgroundColor: 'var(--surface-subtle)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '8px',
        padding: '12px 16px',
        marginBottom: '20px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px'
      }}>
        <Info size={16} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.78125rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <strong style={{ color: 'var(--text-primary)' }}>Empirical Finding:</strong> Stock price levels exhibit strong econometric persistence, making Naive Persistence a competitive point-estimate baseline for next-day close. The proposed <strong>CNN-LSTM architecture</strong> extracts localized temporal features via 1D convolutions before sequential recurrence, delivering strong directional classification guidance across major NSE equities while preserving leakage-safe train/test splits.
        </div>
      </div>

      {/* Test Trajectory Visualization */}
      {trajectory.length > 0 && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <TrendingUp size={15} color="var(--primary-green)" />
              <span>Unseen Test Horizon: Actual vs Model Trajectory</span>
            </h3>
            <span style={{ fontSize: '0.71875rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Sample: {trajectory.length} Test Sessions
            </span>
          </div>

          <div style={{ width: '100%', height: '280px', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid var(--border-subtle)', padding: '12px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectory} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E4EAE6" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} fontFamily="var(--font-mono)" minTickGap={30} />
                <YAxis stroke="var(--text-muted)" fontSize={10} fontFamily="var(--font-mono)" domain={['auto', 'auto']} orientation="right" width={65} tickFormatter={(v) => `${currency}${v}`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid var(--border-default)',
                    borderRadius: '6px',
                    fontSize: '0.75rem',
                    fontFamily: 'var(--font-mono)',
                    boxShadow: 'var(--shadow-md)'
                  }}
                />
                <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: '0.75rem' }} />
                <Line type="monotone" dataKey="actual" name="Actual Price" stroke="#17211C" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="cnn_lstm" name="CNN-LSTM" stroke="#159B62" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="lstm" name="Standard LSTM" stroke="#0284C7" strokeWidth={1.5} strokeDasharray="3 3" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
