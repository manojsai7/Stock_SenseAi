import React, { useState } from 'react';
import {
  Layers,
  Cpu,
  ArrowRight,
  Sparkles,
  ShieldCheck,
  Zap,
  TrendingUp,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import HelpTooltip from './common/HelpTooltip';
import './CnnLstmExplainer.css';

const ABLATION_MODELS = {
  hybrid: {
    id: 'hybrid',
    name: 'CNN-LSTM Hybrid (Our Production AI)',
    badge: 'ACTIVE PRODUCTION',
    badgeColor: '#10b981',
    role: 'Combines 1D-CNN local pattern recognition with LSTM sequential memory.',
    howItWorks: 'The 1D-CNN slides across 3-day windows to extract sharp momentum shapes and volatility spikes. The LSTM then tracks how these patterns evolve across the 30-day timeline to forecast tomorrow\'s price move.',
    strengths: 'Extracts clean local signals without drowning in raw market noise; maintains long-range memory of broader weekly trends.',
    weakness: 'Requires more training compute than simpler linear models (mitigated by our 7 stationary features).',
    winRate: '56.8%',
    mae: '₹11.57',
    verdict: 'Highest overall predictive synergy. Proven best directional inflection detection.',
    vivaTip: '"CNN acts as a feature extractor eliminating high-frequency noise; LSTM acts as a temporal sequence model capturing long-term dependencies."'
  },
  lstm_alone: {
    id: 'lstm_alone',
    name: 'Standard LSTM Alone (No CNN)',
    badge: 'ABLATION BENCHMARK',
    badgeColor: '#3b82f6',
    role: 'Pure recurrent neural network without convolutional feature filtering.',
    howItWorks: 'Feeds raw 30-day numbers directly into LSTM memory cells. It attempts to learn both local candlestick shapes and multi-week trend momentum simultaneously.',
    strengths: 'Excellent sequential memory over long time horizons.',
    weakness: 'FATAL FLAW: Financial time series have low signal-to-noise ratio. Without the CNN filter, raw market noise contaminates the LSTM cell memory, causing it to saturate and miss rapid inflection points.',
    winRate: '51.2%',
    mae: '₹12.50',
    verdict: 'Struggles with short-term noise. Only marginally better than a 50/50 coin toss.',
    vivaTip: '"Without convolutional feature extraction, the LSTM is forced to learn local spatial representations and sequential memory simultaneously, leading to gradient degradation."'
  },
  cnn_alone: {
    id: 'cnn_alone',
    name: '1D-CNN Alone (No LSTM)',
    badge: 'ABLATION BENCHMARK',
    badgeColor: '#8b5cf6',
    role: 'Pure feedforward convolutional network with dense layers (no recurrent memory).',
    howItWorks: 'Applies multiple 1D convolutional filters over the 30-day window, followed by pooling and a dense output layer.',
    strengths: 'Extremely fast to train; exceptional at detecting local candlestick patterns (hammers, engulfing candles, volume breakout spikes).',
    weakness: 'FATAL FLAW: Temporal Amnesia. CNNs are translation-invariant—they detect that a pattern occurred, but have no recurrent mechanism to understand the chronological sequence (whether it happened 25 days ago or yesterday).',
    winRate: '52.0%',
    mae: '₹12.80',
    verdict: 'Fast pattern finder, but blind to multi-week trend momentum.',
    vivaTip: '"CNN alone lacks recurrent state transitions (Markovian temporal amnesia), treating historical sessions as static spatial grids."'
  },
  naive: {
    id: 'naive',
    name: 'Naive Persistence Baseline (Yesterday\'s Copycat)',
    badge: 'SANITY CHECK',
    badgeColor: '#f59e0b',
    role: 'Zero-parameter rule: assumes tomorrow\'s closing price equals today\'s close.',
    howItWorks: 'Does not use neural networks. It simply guesses that price change is 0 (r = 0.00%).',
    strengths: 'Achieves low rupee error during calm, sideways, low-volatility regimes.',
    weakness: 'FATAL FLAW: Zero intelligence. Completely incapable of predicting turns, rallies, crashes, or directional changes.',
    winRate: '0.0%',
    mae: '₹11.80',
    verdict: 'Essential sanity check. Any AI that cannot beat Naive is worthless.',
    vivaTip: '"Under the random walk hypothesis, beating naive persistence in directional accuracy is the primary proof of quantitative predictive utility."'
  }
};

export default function CnnLstmExplainer() {
  const [selectedAblation, setSelectedAblation] = useState('hybrid');
  const [isExpanded, setIsExpanded] = useState(true);

  const active = ABLATION_MODELS[selectedAblation];

  return (
    <section className="fintech-card cnn-lstm-explainer-card" aria-labelledby="cnn-lstm-explainer-title">
      {/* Header */}
      <div className="explainer-header">
        <div className="explainer-title-group">
          <div className="explainer-eyebrow">
            <Cpu size={14} />
            <span>NEURAL ARCHITECTURE DEEP DIVE</span>
            <HelpTooltip title="CNN-LSTM Hybrid" text="Why did we combine CNN and LSTM? To solve the fundamental problem of financial market noise." />
          </div>
          <h2 id="cnn-lstm-explainer-title">Why CNN-LSTM? The Anatomy of Our Deep Learning Engine</h2>
          <p className="explainer-subtitle">
            See exactly how our neural network processes 30 days of market data, and why CNN alone or LSTM alone fails.
          </p>
        </div>

        <button
          type="button"
          className="explainer-toggle-btn"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-expanded={isExpanded}
        >
          <span>{isExpanded ? 'Hide Architecture Breakdown' : 'Explore How CNN-LSTM Works'}</span>
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {isExpanded && (
        <>
          {/* Visual 4-Stage Pipeline */}
          <div className="pipeline-visual-container">
            <div className="pipeline-step-card step-input">
              <div className="step-num">STAGE 1</div>
              <div className="step-icon-wrap">
                <Layers size={18} />
              </div>
              <h4>30-Day Sequence</h4>
              <span className="step-tensor-tag">Tensor: [30 × 7]</span>
              <p>30 completed trading days of 7 stationary features (returns, momentum, volume).</p>
            </div>

            <div className="pipeline-arrow">
              <ArrowRight size={18} />
              <span className="arrow-label">Raw Data</span>
            </div>

            <div className="pipeline-step-card step-cnn">
              <div className="step-num">STAGE 2</div>
              <div className="step-icon-wrap">
                <Sparkles size={18} />
              </div>
              <h4>1D-CNN Layer</h4>
              <span className="step-tensor-tag">Conv1D(64 filters, kernel=3)</span>
              <p>
                <strong>Pattern Scanner:</strong> Slides over 3-day windows to detect candlestick shapes and volume breakout spikes.
              </p>
            </div>

            <div className="pipeline-arrow">
              <ArrowRight size={18} />
              <span className="arrow-label">Filtered</span>
            </div>

            <div className="pipeline-step-card step-lstm">
              <div className="step-num">STAGE 3</div>
              <div className="step-icon-wrap">
                <Cpu size={18} />
              </div>
              <h4>LSTM Memory Layer</h4>
              <span className="step-tensor-tag">LSTM(50 units)</span>
              <p>
                <strong>Sequential Memory:</strong> Tracks how the CNN patterns evolve over time using Forget & Input memory gates.
              </p>
            </div>

            <div className="pipeline-arrow">
              <ArrowRight size={18} />
              <span className="arrow-label">Hidden State</span>
            </div>

            <div className="pipeline-step-card step-output">
              <div className="step-num">STAGE 4</div>
              <div className="step-icon-wrap">
                <TrendingUp size={18} />
              </div>
              <h4>Tomorrow's Close</h4>
              <span className="step-tensor-tag">Dense(1) Output</span>
              <p>
                Predicts return <em>r̂</em>, then calculates cash close: <strong>Close × (1 + r̂)</strong>.
              </p>
            </div>
          </div>

          {/* Interactive Ablation Study: CNN vs LSTM vs Hybrid */}
          <div className="ablation-section">
            <div className="ablation-header-row">
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <h3 className="ablation-heading">Ablation Study: What Happens if You Remove CNN or LSTM?</h3>
                  <HelpTooltip title="Ablation Study" text="An essential scientific experiment where researchers systematically remove parts of an AI model to prove each component is necessary." />
                </div>
                <p className="ablation-sub">
                  Select each architecture below to inspect its individual strengths, fatal weaknesses, and empirical test win rates:
                </p>
              </div>
            </div>

            {/* Model Selector Tabs */}
            <div className="ablation-tabs-grid" role="tablist">
              {Object.values(ABLATION_MODELS).map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="tab"
                  aria-selected={selectedAblation === m.id}
                  className={`ablation-tab-btn ${selectedAblation === m.id ? 'is-active' : ''}`}
                  onClick={() => setSelectedAblation(m.id)}
                >
                  <span className="tab-badge" style={{ backgroundColor: `${m.badgeColor}20`, color: m.badgeColor }}>
                    {m.badge}
                  </span>
                  <span className="tab-name">{m.name}</span>
                </button>
              ))}
            </div>

            {/* Selected Model Deep Dive Card */}
            <div className="ablation-detail-card">
              <div className="ablation-detail-top">
                <div>
                  <span className="detail-badge" style={{ backgroundColor: `${active.badgeColor}20`, color: active.badgeColor }}>
                    {active.badge}
                  </span>
                  <h4 className="detail-title">{active.name}</h4>
                  <p className="detail-role">{active.role}</p>
                </div>

                <div className="detail-stat-cluster">
                  <div className="detail-stat-box">
                    <span className="stat-label">Direction Win Rate</span>
                    <strong className="stat-value" style={{ color: active.badgeColor }}>{active.winRate}</strong>
                  </div>
                  <div className="detail-stat-box">
                    <span className="stat-label">Average Cash MAE</span>
                    <strong className="stat-value">{active.mae}</strong>
                  </div>
                </div>
              </div>

              <div className="ablation-grid-two">
                <div className="ablation-info-box">
                  <strong className="box-title text-green">
                    <CheckCircle2 size={15} /> How It Works & Strengths:
                  </strong>
                  <p>{active.howItWorks}</p>
                  <p style={{ marginTop: '6px', color: '#117a56' }}>{active.strengths}</p>
                </div>

                <div className="ablation-info-box">
                  <strong className="box-title text-red">
                    <AlertCircle size={15} /> Weakness & Why It Fails Alone:
                  </strong>
                  <p>{active.weakness}</p>
                  <div className="box-verdict">
                    <strong>Scientific Verdict:</strong> {active.verdict}
                  </div>
                </div>
              </div>

              {/* Viva Defense Line */}
              <div className="ablation-viva-bar">
                <Zap size={15} color="#d97706" />
                <span>
                  <strong>Viva Defense One-Liner:</strong> {active.vivaTip}
                </span>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
