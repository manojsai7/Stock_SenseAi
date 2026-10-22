import React, { useEffect, useRef } from 'react';
import { X, Cpu, Layers, GitBranch, ShieldCheck } from 'lucide-react';

export default function ArchitectureModal({ isOpen, onClose }) {
  const dialogRef = useRef(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return undefined;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialogRef.current?.focus();
    function handleKeyDown(event) {
      if (event.key === 'Escape') onCloseRef.current();
      if (event.key !== 'Tab') return;
      const focusable = dialogRef.current?.querySelectorAll('button, a[href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement) previousFocus.focus();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      onClick={onClose}
      role="presentation"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'rgba(16, 24, 20, 0.45)',
        backdropFilter: 'blur(4px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="architecture-dialog-title"
        tabIndex={-1}
        style={{
          maxWidth: '880px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          backgroundColor: '#FFFFFF',
          borderRadius: '12px',
          border: '1px solid var(--border-default)',
          boxShadow: 'var(--shadow-lg)',
          padding: '28px',
          position: 'relative',
        }}
      >
        {/* Close Button */}
        <button
          type="button"
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '18px',
            right: '18px',
            backgroundColor: 'var(--surface-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
          }}
          aria-label="Close architecture guide"
        >
          <X size={16} />
        </button>

        {/* Modal Header */}
        <div style={{ marginBottom: '20px', paddingRight: '40px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Cpu size={20} color="var(--primary-green)" />
            <h2 style={{
              fontSize: '1.25rem',
              fontWeight: 800,
              color: 'var(--text-primary)',
              letterSpacing: '-0.02em',
              margin: 0
            }}>
              <span id="architecture-dialog-title">
              System Architecture & Viva Defense Reference
              </span>
            </h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8125rem' }}>
            Academic deep-learning specification: time-series sequence preprocessing, CNN-LSTM hybrid topology, and leakage-safe validation.
          </p>
        </div>

        {/* Model Framework Summary Banner */}
        <div style={{
          backgroundColor: 'var(--surface-subtle)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
        }}>
          <div>
            <div style={{ fontSize: '0.6875rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.04em' }}>
              Engine
            </div>
            <div style={{ fontSize: '0.9375rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              TensorFlow / Keras Neural Network Pipeline
            </div>
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            color: 'var(--primary-green)',
            backgroundColor: 'var(--green-tint)',
            border: '1px solid var(--green-border)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontWeight: 600
          }}>
            Conv1D(64) → MaxPool1D → LSTM(50) → Dense(1)
          </div>
        </div>

        {/* Section 1: End-to-End Pipeline */}
        <div style={{ marginBottom: '22px' }}>
          <h3 style={{
            fontSize: '0.9375rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            marginBottom: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <GitBranch size={16} color="var(--primary-green)" />
            <span>1. End-to-End Pipeline (Zero Data Leakage)</span>
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
            gap: '8px',
            textAlign: 'center',
            fontSize: '0.75rem',
          }}>
            <div style={{ backgroundColor: 'var(--surface-subtle)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--primary-green)', fontWeight: 700 }}>1. Ingestion</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '0.6875rem' }}>NSE OHLCV (Daily)</div>
            </div>
            <div style={{ backgroundColor: 'var(--surface-subtle)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>2. Features</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '0.6875rem' }}>OHLCV + 20-Day SMA</div>
            </div>
            <div style={{ backgroundColor: 'var(--surface-subtle)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>3. Split (70/15/15)</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '0.6875rem' }}>Strict Chronological</div>
            </div>
            <div style={{ backgroundColor: 'var(--surface-subtle)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--primary-green)', fontWeight: 700 }}>4. Scaling</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '0.6875rem' }}>Train Partition ONLY</div>
            </div>
            <div style={{ backgroundColor: 'var(--surface-subtle)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--text-primary)', fontWeight: 700 }}>5. CNN-LSTM</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '0.6875rem' }}>30-Day Window</div>
            </div>
            <div style={{ backgroundColor: 'var(--surface-subtle)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--primary-green)', fontWeight: 700 }}>6. Benchmark</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '0.6875rem' }}>Unseen Test Horizon</div>
            </div>
          </div>
        </div>

        {/* Section 2: Why CNN-LSTM? */}
        <div style={{ marginBottom: '22px' }}>
          <h3 style={{
            fontSize: '0.9375rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            marginBottom: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <Layers size={16} color="var(--primary-green)" />
            <span>2. Architectural Synergy: Conv1D + LSTM</span>
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
            <div style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '14px'
            }}>
              <h4 style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                1D Convolutions (Feature Extraction)
              </h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', lineHeight: 1.5 }}>
                Local candlestick waveforms, momentum breakouts, and volume anomalies occur over short 3–5 day windows. 1D convolutions act as learned filters to isolate translation-invariant technical patterns.
              </p>
            </div>

            <div style={{
              backgroundColor: 'var(--surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '14px'
            }}>
              <h4 style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                Recurrent LSTM (Sequential Memory)
              </h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', lineHeight: 1.5 }}>
                Standard feedforward networks lack memory of previous steps. The LSTM maintains cell state and gated transitions (Input, Forget, Output) to capture multi-week trend momentum across the 30-day sequence.
              </p>
            </div>
          </div>
        </div>

        {/* Section 3: Viva Questions & Defenses */}
        <div>
          <h3 style={{
            fontSize: '0.9375rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            marginBottom: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <ShieldCheck size={16} color="var(--primary-green)" />
            <span>3. Viva Defense Guide & Academic Justifications</span>
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
            <div style={{ backgroundColor: 'var(--surface-subtle)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '10px 14px' }}>
              <strong style={{ color: 'var(--text-primary)', fontSize: '0.8125rem' }}>
                Q: Why not use Intraday (1-minute / 5-minute) bars?
              </strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '3px', lineHeight: 1.4 }}>
                Intraday market microstructure is dominated by transient order routing and bid-ask bounces. Free public APIs provide at most 60 days of intraday data. Daily candles offer 5 years of macroeconomic cycles, preventing severe overfitting.
              </p>
            </div>

            <div style={{ backgroundColor: 'var(--surface-subtle)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '10px 14px' }}>
              <strong style={{ color: 'var(--text-primary)', fontSize: '0.8125rem' }}>
                Q: What is the difference between the Legacy and Stationary pipelines?
              </strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '3px', lineHeight: 1.4 }}>
                <strong>Legacy Pipeline:</strong> Uses 6 raw OHLCV+MA20 price features with MinMax normalization to predict tomorrow's closing price level directly. Can suffer from price-level drift when stock prices move outside historical ranges.<br />
                <strong>Stationary Pipeline (v1):</strong> Uses 7 relative, strictly stationary features (1d, 5d, 20d returns, intraday return, high-low range ratio, close-to-MA20, volume ratio) to forecast the next-session return (r̂). The closing price is reconstructed mathematically via Closeₜ × (1 + r̂), guaranteeing structural boundedness and eliminating non-stationary drift.
              </p>
            </div>

            <div style={{ backgroundColor: 'var(--surface-subtle)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '10px 14px' }}>
              <strong style={{ color: 'var(--text-primary)', fontSize: '0.8125rem' }}>
                Q: Why is Naive Persistence competitive in price MAE?
              </strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '3px', lineHeight: 1.4 }}>
                Under the Efficient Market Hypothesis, stock price levels resemble martingales where tomorrow's price is close to today's price. Naive persistence achieves low MAE in low-volatility regimes. CNN-LSTM adds distinct value in <strong>Directional Accuracy</strong> by forecasting inflection points.
              </p>
            </div>

            <div style={{ backgroundColor: 'var(--surface-subtle)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '10px 14px' }}>
              <strong style={{ color: 'var(--text-primary)', fontSize: '0.8125rem' }}>
                Q: How is zero Data Leakage guaranteed?
              </strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '3px', lineHeight: 1.4 }}>
                Data splits are strictly chronological. All preprocessing parameters (scalers) are fitted strictly on the Train set. Validation and Test splits are transformed without refitting. Features at session t strictly use data at or before session t.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
