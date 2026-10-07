import React from 'react';
import { X, Cpu, Layers, GitBranch, Database, ShieldAlert, CheckCircle, Zap, BookOpen, BarChart2, ShieldCheck } from 'lucide-react';

export default function ArchitectureModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      background: 'rgba(0, 0, 0, 0.8)',
      backdropFilter: 'blur(8px)',
      zIndex: 100,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
    }}>
      <div className="glass-panel" style={{
        maxWidth: '920px',
        width: '100%',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '32px',
        position: 'relative',
        background: 'rgba(14, 19, 31, 0.98)',
        border: '1px solid rgba(139, 92, 246, 0.4)',
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(255, 255, 255, 0.08)',
            border: 'none',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-primary)',
            cursor: 'pointer',
          }}
        >
          <X size={18} />
        </button>

        {/* Modal Header */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <Cpu size={24} color="#FF9E40" />
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', fontWeight: 700, color: '#F8FAFC' }}>
              StockSense AI — System Design & Viva Defense Architecture
            </h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Defensible technical methodology: Deep Learning Framework, Leakage-Free Data Pipeline, and Experimental Protocol.
          </p>
        </div>

        {/* Deep Learning Framework Banner */}
        <div style={{
          background: 'linear-gradient(90deg, rgba(255, 111, 0, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%)',
          border: '1px solid rgba(255, 111, 0, 0.3)',
          borderRadius: '12px',
          padding: '16px 20px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
        }}>
          <div>
            <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#FF9E40', fontWeight: 700, letterSpacing: '0.05em' }}>
              Deep Learning Engine
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF', marginTop: '2px' }}>
              TensorFlow 2.17+ with Keras Neural Network API
            </div>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#C4B5FD', background: 'rgba(0,0,0,0.3)', padding: '6px 12px', borderRadius: '6px' }}>
            Conv1D → MaxPooling1D → LSTM → Dropout → Dense
          </div>
        </div>

        {/* Section 1: End-to-End Pipeline */}
        <div style={{ marginBottom: '26px', background: 'rgba(255,255,255,0.02)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <GitBranch size={18} />
            <span>1. End-to-End Execution Pipeline (Zero Leakage)</span>
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
            gap: '8px',
            textAlign: 'center',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
          }}>
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ color: 'var(--accent-green)', fontWeight: 700 }}>1. Ingestion</div>
              <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>Yahoo Finance (5Y NSE OHLCV)</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>2. Features</div>
              <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>OHLCV + 20-Day SMA</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ color: '#FF9E40', fontWeight: 700 }}>3. Split (70/15/15)</div>
              <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>Strict Chronological Split</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ color: 'var(--accent-violet)', fontWeight: 700 }}>4. Scaling</div>
              <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>MinMax (Train Fit ONLY)</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ color: '#38BDF8', fontWeight: 700 }}>5. CNN-LSTM</div>
              <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>TensorFlow / Keras</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ color: '#34D399', fontWeight: 700 }}>6. Evaluation</div>
              <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>Unseen Test Set Horizon</div>
            </div>
          </div>
        </div>

        {/* Section 2: Why CNN-LSTM? */}
        <div style={{ marginBottom: '26px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#A78BFA', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} />
            <span>2. Architectural Rationale: Conv1D + LSTM Synergy</span>
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
            <div style={{ background: 'rgba(139, 92, 246, 0.08)', border: '1px solid rgba(139, 92, 246, 0.25)', borderRadius: '10px', padding: '16px' }}>
              <h4 style={{ color: '#C4B5FD', fontSize: '0.9rem', marginBottom: '6px' }}>1D Convolution (Conv1D + MaxPool)</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', lineHeight: 1.5 }}>
                Local financial waveform patterns (e.g. abrupt candlestick reversals, volume surges, moving average crossovers) occur over 3- to 5-day intervals. 1D convolutions act as learned feature filters to extract translation-invariant localized representations.
              </p>
            </div>
            <div style={{ background: 'rgba(6, 182, 212, 0.08)', border: '1px solid rgba(6, 182, 212, 0.25)', borderRadius: '10px', padding: '16px' }}>
              <h4 style={{ color: '#67E8F9', fontSize: '0.9rem', marginBottom: '6px' }}>Recurrent LSTM Layer</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', lineHeight: 1.5 }}>
                Standard feedforward networks cannot model chronological dependencies. The LSTM maintains a cell state with Input, Forget, and Output gating mechanisms that retain macroeconomic trend momentum across the 30-day temporal window.
              </p>
            </div>
          </div>
        </div>

        {/* Section 3: Viva Questions & Scientific Defenses */}
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#34D399', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} />
            <span>3. Viva Defense Guide & Academic Justifications</span>
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '12px 16px' }}>
              <strong style={{ color: '#F8FAFC', fontSize: '0.85rem' }}>Q: Why did you not use Intraday (hourly/minute) data?</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px', lineHeight: 1.4 }}>
                Intraday market data contains severe microstructure noise (bid-ask bounces, order routing delays) and free APIs restrict intraday history to 60 days. Daily data provides 5 years of stable macroeconomic history, enabling deep learning models to learn generalizable market cycles without overfitting.
              </p>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '12px 16px' }}>
              <strong style={{ color: '#F8FAFC', fontSize: '0.85rem' }}>Q: Why is the Naive Persistence Baseline competitive in MAE?</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px', lineHeight: 1.4 }}>
                Under the Efficient Market Hypothesis (EMH), stock price levels approximate a martingale where tomorrow's expected price equals today's price. Consequently, Naive Persistence yields low MAE in calm regimes. The proposed CNN-LSTM adds value primarily in <strong>Directional Accuracy (Hit Rate)</strong> by anticipating turns rather than merely repeating yesterday's price.
              </p>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '12px 16px' }}>
              <strong style={{ color: '#F8FAFC', fontSize: '0.85rem' }}>Q: How do you guarantee zero Data Leakage?</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px', lineHeight: 1.4 }}>
                Splitting is strictly chronological (70% Train, 15% Validation, 15% Test). All preprocessing parameters (MinMaxScaler min/max) are fitted exclusively on the 70% Train partition. Validation and Test sets are transformed without fitting. No future timestamps or targets ever enter model training.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
