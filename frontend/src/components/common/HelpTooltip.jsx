import React, { useState, useRef, useEffect } from 'react';
import { HelpCircle, X } from 'lucide-react';

export const GLOSSARY_TERMS = {
  bps: {
    term: 'Basis Points (bps)',
    category: 'Market Metric',
    simple: 'A tiny unit of percentage. 1 bps = 0.01% (1/100th of 1%). 100 bps = 1.00%.',
    example: 'A +25 bps gain on a ₹1,000 stock means a small ₹2.50 nudge (+0.25%). Stock market shifts are often small fractions, so traders use bps for precision.',
  },
  mae: {
    term: 'Mean Absolute Error (MAE)',
    category: 'Accuracy Metric',
    simple: 'The average cash miss in Rupees between the AI\'s guess and what actually traded.',
    example: 'An MAE of ₹11.57 means on an average day, the AI\'s predicted closing price was off by approximately ₹11.57.',
  },
  rmse: {
    term: 'Root Mean Squared Error (RMSE)',
    category: 'Accuracy Metric',
    simple: 'Like MAE, but gives extra heavy punishment to rare, catastrophic blunders.',
    example: 'If an AI is usually off by ₹5 but makes one huge blunder of ₹60, RMSE jumps sharply to warn you of instability.',
  },
  mcc: {
    term: 'Matthews Correlation Coefficient (MCC)',
    category: 'Scientific Skill Score',
    simple: 'A gold-standard score from -1 to +1 proving the AI has real predictive skill, not just lucky coin flips.',
    example: '+1 = Perfect foresight; 0 = Pure 50/50 luck (coin toss); -1 = Always wrong. A positive MCC proves the neural network has an edge.',
  },
  win_rate: {
    term: 'Directional Win Rate',
    category: 'Performance',
    simple: 'The percentage of days the AI correctly called whether the stock closed UP or DOWN.',
    example: 'In real-world equity markets, 54%–58% win rate is considered strong by top quantitative hedge funds due to high market noise.',
  },
  stationary_return: {
    term: 'Stationary Return Model (V1)',
    category: 'Machine Learning',
    simple: 'Predicting percentage moves instead of raw Rupee numbers to eliminate inflation drift.',
    example: 'Like predicting whether an elevator goes up 1 floor or down 1 floor, rather than guessing its exact height above sea level in meters.',
  },
  naive_baseline: {
    term: 'Naive Persistence Baseline',
    category: 'Benchmark',
    simple: 'The simplest baseline: guessing tomorrow\'s price will be exactly identical to today\'s close.',
    example: 'If Reliance closed at ₹1,367 today, Naive simply guesses ₹1,367 tomorrow. Any worthwhile AI model must beat Naive to be considered valuable.',
  },
  shadow_mode: {
    term: 'Shadow Observation Mode',
    category: 'System Architecture',
    simple: 'Testing a new challenger AI (GRU V2) silently in the background alongside the main production AI.',
    example: 'It generates real prospective forecasts before market open, but does not serve regular users until 60–100 live sessions prove it is better than V1.',
  },
  walk_forward: {
    term: 'Walk-Forward Testing',
    category: 'Academic Rigor',
    simple: 'A time-machine simulation where the AI is tested across historical years strictly in chronological order.',
    example: 'The model trains on Year 1, tests on Year 2, then re-trains on Years 1–2 to test on Year 3. It never peeks ahead into the future.',
  },
  up_recall: {
    term: 'Rally Catch Rate (UP Recall)',
    category: 'Directional Metric',
    simple: 'How often the AI successfully identified an upward green market day.',
    example: '70% UP Recall means out of 10 market rallies, the AI successfully called 7 of them.',
  },
  down_recall: {
    term: 'Drop Warning Rate (DOWN Recall)',
    category: 'Directional Metric',
    simple: 'How often the AI successfully warned about a downward red market day.',
    example: 'High DOWN recall is critical for risk management so investors can prepare for downturns.',
  },
  capability_score: {
    term: 'Model Capability Score',
    category: 'Composite Index',
    simple: 'An overall health rating out of 100 combining Direction Win Rate (35%), Skill MCC (30%), Cash Accuracy (20%), and Bull/Bear Balance (15%).',
    example: 'Scores above 60 indicate moderate predictive consistency; scores above 75 indicate strong multi-regime reliability.',
  },
  return_1d: {
    term: 'Yesterday\'s Momentum (1-Day Return)',
    category: 'AI Input Feature',
    simple: 'The percentage price change of the stock during the most recent completed market day.',
    example: 'Measures immediate short-term momentum to see if buyers or sellers carried speed into the close.',
  },
  return_5d: {
    term: '1-Week Momentum (5-Day Return)',
    category: 'AI Input Feature',
    simple: 'The cumulative percentage change across the last 5 trading days (one full trading week).',
    example: 'Captures weekly swings to identify short-term trends.',
  },
  return_20d: {
    term: '1-Month Momentum (20-Day Return)',
    category: 'AI Input Feature',
    simple: 'The percentage change across the last 20 trading days (roughly one trading month).',
    example: 'Shows medium-term directional bias and whether the stock is in a broader rally or pullback.',
  },
  intraday_return: {
    term: 'Intraday Buying Pressure',
    category: 'AI Input Feature',
    simple: 'The percentage change from today\'s opening bell price to today\'s closing bell price.',
    example: 'A positive intraday return means buyers took control after market open and pushed prices higher into the close.',
  },
  hl_range_ratio: {
    term: 'Daily Price Volatility Spread',
    category: 'AI Input Feature',
    simple: 'The gap between the day\'s highest price and lowest price, divided by the closing price.',
    example: 'A large range ratio means heavy volatility and intense tug-of-war between bulls and bears.',
  },
  close_to_ma20: {
    term: 'Distance from 20-Day Average',
    category: 'AI Input Feature',
    simple: 'How far current price has stretched away from its 20-day moving average benchmark.',
    example: 'If price is far above MA20, it is stretched and might pull back; if near MA20, it is at a normal resting level.',
  },
  volume_ratio: {
    term: 'Trading Volume Surge',
    category: 'AI Input Feature',
    simple: 'Today\'s share trading volume compared to the 20-day average volume.',
    example: 'A volume ratio of 1.8 means 80% more shares were traded than usual, signaling major institutional activity.',
  },
  sma20: {
    term: '20-Day Moving Average (SMA 20)',
    category: 'Technical Indicator',
    simple: 'The average closing price over the last 20 trading days (1 month).',
    example: 'Acts as a short-term trend line. Prices trading above SMA 20 suggest short-term strength.',
  },
  sma50: {
    term: '50-Day Moving Average (SMA 50)',
    category: 'Technical Indicator',
    simple: 'The average closing price over the last 50 trading days (approx 2.5 months).',
    example: 'A key medium-term barometer used by institutional investors worldwide.',
  },
  bollinger: {
    term: 'Bollinger Bands',
    category: 'Technical Indicator',
    simple: 'A volatility envelope around the price. 95% of normal trading price action stays inside these upper and lower boundaries.',
    example: 'Touching the upper band means the stock is momentarily stretched high; touching the lower band means momentarily stretched low.',
  },
  rsi: {
    term: 'Relative Strength Index (RSI)',
    category: 'Technical Indicator',
    simple: 'A market speedometer scored from 0 to 100.',
    example: 'Above 70 = Hot / Overbought (risk of short-term pullback). Below 30 = Cold / Oversold (potential rebound zone).',
  },
  macd: {
    term: 'MACD (Trend Momentum)',
    category: 'Technical Indicator',
    simple: 'Moving Average Convergence Divergence. Compares fast momentum vs slow momentum.',
    example: 'When the fast line crosses above the slow line, it signals fresh bullish acceleration.',
  },
};

export default function HelpTooltip({
  term,
  title,
  text,
  example,
  category,
  position = 'top',
  children,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  const entry = term && GLOSSARY_TERMS[term] ? GLOSSARY_TERMS[term] : null;
  const displayTitle = title || entry?.term || 'Financial Term';
  const displayText = text || entry?.simple || '';
  const displayExample = example || entry?.example || '';
  const displayCategory = category || entry?.category || null;

  useEffect(() => {
    if (!isOpen) return;
    function handleClickOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    }
    function handleKeyDown(e) {
      if (e.key === 'Escape') setIsOpen(false);
    }
    document.addEventListener('pointerdown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  return (
    <span
      ref={containerRef}
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        verticalAlign: 'middle',
      }}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        aria-label={`Explain ${displayTitle}`}
        title={`Click for simple explanation: ${displayTitle}`}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '3px',
          background: isOpen ? 'rgba(17, 122, 86, 0.15)' : 'rgba(0, 0, 0, 0.04)',
          color: isOpen ? '#117A56' : '#64748b',
          border: isOpen ? '1px solid rgba(17, 122, 86, 0.3)' : '1px solid rgba(0, 0, 0, 0.08)',
          borderRadius: '9999px',
          padding: children ? '2px 7px' : '2px',
          fontSize: '0.6875rem',
          fontWeight: 600,
          cursor: 'pointer',
          lineHeight: 1,
          transition: 'all 0.15s ease',
          margin: '0 3px',
        }}
      >
        {children && <span>{children}</span>}
        <HelpCircle size={12} style={{ opacity: 0.8 }} />
      </button>

      {isOpen && (
        <div
          role="dialog"
          aria-label={displayTitle}
          style={{
            position: 'absolute',
            bottom: position === 'top' ? 'calc(100% + 8px)' : 'auto',
            top: position === 'bottom' ? 'calc(100% + 8px)' : 'auto',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '280px',
            maxWidth: '85vw',
            backgroundColor: '#ffffff',
            color: '#1e293b',
            borderRadius: '12px',
            boxShadow: '0 12px 30px rgba(0, 0, 0, 0.18), 0 2px 6px rgba(0, 0, 0, 0.08)',
            border: '1px solid rgba(0, 0, 0, 0.08)',
            padding: '14px 16px',
            zIndex: 1000,
            fontSize: '0.8125rem',
            lineHeight: 1.45,
            textAlign: 'left',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '8px' }}>
            <div>
              {displayCategory && (
                <span
                  style={{
                    display: 'inline-block',
                    fontSize: '0.625rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    fontWeight: 700,
                    color: '#0284c7',
                    backgroundColor: '#e0f2fe',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    marginBottom: '4px',
                  }}
                >
                  {displayCategory}
                </span>
              )}
              <h4 style={{ margin: 0, fontSize: '0.875rem', fontWeight: 700, color: '#0f172a', letterSpacing: '-0.01em' }}>
                {displayTitle}
              </h4>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
              }}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#94a3b8',
                cursor: 'pointer',
                padding: '2px',
                display: 'inline-flex',
              }}
              aria-label="Close explanation"
            >
              <X size={14} />
            </button>
          </div>

          {/* Simple Explanation */}
          <p style={{ margin: '0 0 8px', color: '#334155', fontSize: '0.8125rem' }}>
            {displayText}
          </p>

          {/* Example / Context */}
          {displayExample && (
            <div
              style={{
                backgroundColor: '#f8fafc',
                borderLeft: '3px solid #117A56',
                borderRadius: '0 6px 6px 0',
                padding: '6px 10px',
                fontSize: '0.75rem',
                color: '#475569',
              }}
            >
              <strong style={{ color: '#0f172a' }}>Real-World Meaning: </strong>
              {displayExample}
            </div>
          )}
        </div>
      )}
    </span>
  );
}
