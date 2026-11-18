import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  Search,
  BookOpen,
  Sparkles,
  TrendingUp,
  Brain,
  ShieldCheck,
  HelpCircle,
  BarChart2,
  CheckCircle,
  Lightbulb,
  Award
} from 'lucide-react';
import './PlainEnglishGuideModal.css';

const GUIDE_CARDS = [
  // --- SECTION: CORE FAQ & REALITY CHECK ---
  {
    id: 'future-predict',
    category: 'Reality Check',
    title: 'Can AI really predict the future of stock prices?',
    icon: Sparkles,
    badge: 'Fundamental FAQ',
    summary: 'No model has a crystal ball. StockSense AI generates statistical probabilistic estimates, not deterministic certainties.',
    explanation: 'Stock markets are noisy, non-linear dynamic systems influenced by global events, earnings, geopolitics, and human psychology. Deep learning models cannot guarantee exact prices. Instead, our CNN-LSTM estimates the conditional expected price shift based on historical price momentum and technical structure.',
    vivaTip: 'Always answer: "Our objective is not speculative prophecy, but rigorous statistical estimation of next-session conditional returns while controlling for data leakage."'
  },
  {
    id: 'random-walk',
    category: 'Reality Check',
    title: 'Isn\'t the stock market completely random?',
    icon: HelpCircle,
    badge: 'Efficient Market Hypothesis',
    summary: 'Weak-form market efficiency says past prices cannot yield excess risk-adjusted returns, but short-term temporal dependencies and volatility clustering exist.',
    explanation: 'While asset prices approximate a random walk over long horizons, financial econometrics shows short-term autocorrelation, mean reversion, and momentum exist across short multi-day windows. Our CNN-LSTM extracts these subtle sequential patterns.',
    vivaTip: 'Mention Mandelbrot and volatility clustering: "Asset returns are not Gaussian white noise; they exhibit heavy tails and volatility clustering that sequential neural networks can capture."'
  },
  {
    id: 'sentiment',
    category: 'Reality Check',
    title: 'Why didn\'t you include Twitter or News Sentiment?',
    icon: ShieldCheck,
    badge: 'Design Decision',
    summary: 'News and social feeds introduce extreme noise, unverified lag, look-ahead bias, and bot manipulation that degrade financial reliability.',
    explanation: 'Many student projects scrape tweets, which introduces severe latency and fake bot sentiment. Academic literature shows unverified sentiment pipelines lead to overfitted models. StockSense AI intentionally focuses on pure, audited, tick-verified price-volume action.',
    vivaTip: 'Tell faculty: "We prioritized empirical data integrity and zero data leakage over noisy social sentiment heuristics."'
  },

  // --- SECTION: METRICS & ACCURACY ---
  {
    id: 'bps',
    category: 'Metrics & Terms',
    title: 'What is a Basis Point (bps)?',
    icon: BarChart2,
    badge: 'Financial Unit',
    summary: 'A tiny unit of percentage. 1 bps = 0.01% (one-hundredth of 1%). 100 bps = 1.00%.',
    explanation: 'Daily stock movements are usually small fractions of a percent (+0.12%, -0.35%). Expressing moves in basis points (+12 bps, -35 bps) allows quantitative traders and algorithmic models to communicate without rounding errors.',
    vivaTip: 'Remember: 1 bps = 0.0001 in decimal. A 50 bps move on ₹1,000 is ₹5.00.'
  },
  {
    id: 'mae-rmse',
    category: 'Metrics & Terms',
    title: 'What is MAE vs RMSE in Rupees?',
    icon: TrendingUp,
    badge: 'Cash Error',
    summary: 'MAE is the average cash miss; RMSE heavily penalizes large catastrophic blunders.',
    explanation: 'MAE (Mean Absolute Error) tells you: "On an average day, how many Rupees was the prediction off?" (e.g., ±₹11.50). RMSE (Root Mean Squared Error) squares the errors before averaging, meaning a single ₹50 miss hurts the score much more than five ₹10 misses.',
    vivaTip: 'Examiners love this: "MAE provides direct rupee interpretability, while RMSE measures sensitivity to extreme outlier forecast errors."'
  },
  {
    id: 'mcc',
    category: 'Metrics & Terms',
    title: 'What is Matthews Correlation Coefficient (MCC)?',
    icon: Award,
    badge: 'Gold Standard',
    summary: 'A score from -1 to +1 proving the AI has genuine predictive skill, not just lucky coin tosses.',
    explanation: 'Accuracy alone is deceptive: if a stock rises 90% of days in a bull run, an AI that simply says "UP" every day gets 90% accuracy while having zero intelligence! MCC calculates the correlation between predicted and actual directions across all four confusion matrix quadrants (+1 = perfect skill, 0 = random coin flip, -1 = completely opposite).',
    vivaTip: '"MCC is mathematically robust to regime class imbalances, unlike raw directional accuracy."'
  },
  {
    id: 'up-down-recall',
    category: 'Metrics & Terms',
    title: 'What are UP Recall and DOWN Recall?',
    icon: BarChart2,
    badge: 'Regime Balance',
    summary: 'UP Recall is the rally detection rate; DOWN Recall is the market drop warning rate.',
    explanation: 'UP Recall answers: "Out of all days the market actually went up, what percentage did the AI catch?" DOWN Recall answers: "Out of all days the market dropped, what percentage did the AI warn us about?" An ideal model balances both.',
    vivaTip: 'Point out: "In single-trend samples with no market drops, DOWN recall is mathematically undefined, and our dashboard correctly shows it as unavailable rather than fabricating zero."'
  },

  // --- SECTION: AI ARCHITECTURE & METHODOLOGY ---
  {
    id: 'stationary-returns',
    category: 'AI Architecture',
    title: 'Why predict percentage returns instead of raw prices?',
    icon: Brain,
    badge: 'Core Architecture',
    summary: 'Raw stock prices are non-stationary and drift with inflation, causing deep learning models to suffer spurious correlations.',
    explanation: 'If a stock climbs from ₹1,000 to ₹3,000 over 5 years, raw price numbers never repeat. A neural network trained on raw prices simply memorizes numbers. By converting prices into stationary relative percentage returns: r_t = ln(P_t / P_{t-1}), the statistical distribution remains stable across decades. Once the AI predicts the return r̂, we reconstruct the cash price: P_{t+1} = P_t × (1 + r̂).',
    vivaTip: 'High-value viva line: "We enforce covariance stationarity by taking logarithmic price returns, verified via the Augmented Dickey-Fuller (ADF) test."'
  },
  {
    id: 'cnn-lstm',
    category: 'AI Architecture',
    title: 'Why combine CNN and LSTM together?',
    icon: Brain,
    badge: 'Hybrid Network',
    summary: '1D-CNN acts as a local pattern scanner; LSTM acts as a temporal memory engine.',
    explanation: 'A standard LSTM can struggle with raw noisy numbers. We place a 1D Convolutional layer first: it slides over multi-day candles like a filter, extracting local momentum and volatility patterns. The output is fed into LSTM (Long Short-Term Memory) cells, which track long-range temporal trends and remember whether momentum is accelerating or decaying.',
    vivaTip: '"CNN performs local spatial feature extraction across technical features; LSTM models long-range sequential temporal dependencies."'
  },
  {
    id: 'naive-baseline',
    category: 'AI Architecture',
    title: 'What is the Naive Persistence Benchmark?',
    icon: ShieldCheck,
    badge: 'Sanity Check',
    summary: 'The simplest possible rule: guessing tomorrow\'s price will be exactly identical to today\'s close.',
    explanation: 'In financial time series, tomorrow\'s price is usually very close to today\'s price. If a complex neural network with 100,000 parameters cannot beat this simple "yesterday\'s copycat" rule, the AI is overfitting or adding negative value. StockSense AI compares every model against Naive on every session.',
    vivaTip: '"Beating the naive random-walk persistence benchmark is the foundational hurdle in quantitative finance."'
  },
  {
    id: 'data-leakage',
    category: 'AI Architecture',
    title: 'What is Zero Data Leakage?',
    icon: ShieldCheck,
    badge: 'Academic Honesty',
    summary: 'Preventing future information from accidentally leaking into the model\'s training phase.',
    explanation: 'A common mistake in student projects is fitting scalers (e.g. MinMaxScaler) on the entire dataset. That gives the model knowledge of future highs and lows before it predicts! In StockSense AI, scalers and normalizers are strictly fitted on the training split only, and frozen before evaluating test sets.',
    vivaTip: '"We strictly enforce temporal ordering: all feature normalization and scaling parameters are estimated solely on historical training folds."'
  },
  {
    id: 'shadow-mode',
    category: 'AI Architecture',
    title: 'What is Shadow Evaluation Mode (GRU V2)?',
    icon: Brain,
    badge: 'Production Safety',
    summary: 'Running an experimental model in the background on live data before trusting it with real users.',
    explanation: 'In production engineering, you never replace a working production model (V1) just because a new model looked slightly better in a lab notebook. In StockSense AI, GRU V2 runs in "Shadow Mode": it produces genuine forward forecasts each day, logged immutably before market open, and is scored after market close. Only after 60–100 live sessions will it be considered for promotion.',
    vivaTip: '"Shadow evaluation provides prospective out-of-sample validation under real market conditions without operational risk."'
  },

  // --- SECTION: CHARTING & TECHNICALS ---
  {
    id: 'candlestick',
    category: 'Trading Basics',
    title: 'How do I read a Candlestick in 10 seconds?',
    icon: BarChart2,
    badge: 'Chart Reading',
    summary: 'Green means price rose; Red means price fell. The thick body shows Open & Close; the thin wicks show the Day\'s Extreme High & Low.',
    explanation: 'Each candle represents one full trading day. The rectangular body represents the range between the market opening bell (09:15 IST) and closing bell (15:30 IST). The thin lines (wicks/shadows) stick out to show the highest and lowest prices traded during the day.',
    vivaTip: 'Green candle: Open at bottom, Close at top. Red candle: Open at top, Close at bottom.'
  },
  {
    id: 'indicators',
    category: 'Trading Basics',
    title: 'What do SMA, Bollinger Bands, and RSI mean?',
    icon: TrendingUp,
    badge: 'Technical Signals',
    summary: 'SMA shows the smoothed trend; Bollinger Bands show normal volatility borders; RSI is a momentum speedometer.',
    explanation: '• SMA 20/50: The average price over 20 or 50 days, smoothing out daily market noise.\n• Bollinger Bands: Upper and lower bands where 95% of normal price action stays.\n• RSI (Relative Strength Index): 0 to 100 score. Over 70 means buyers got overheated (overbought); under 30 means panic selling has pushed prices into a bargain zone (oversold).',
    vivaTip: '"Our stationary feature pipeline incorporates SMA distance, volatility ratios, and multi-day returns as continuous normalized inputs."'
  }
];

const CATEGORIES = ['All', 'Reality Check', 'Metrics & Terms', 'AI Architecture', 'Trading Basics'];

export default function PlainEnglishGuideModal({ isOpen, onClose }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');
  const dialogRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const filteredCards = GUIDE_CARDS.filter((card) => {
    const matchesCategory = activeCategory === 'All' || card.category === activeCategory;
    const query = searchQuery.trim().toLowerCase();
    if (!query) return matchesCategory;
    const matchesQuery =
      card.title.toLowerCase().includes(query) ||
      card.summary.toLowerCase().includes(query) ||
      card.explanation.toLowerCase().includes(query) ||
      card.vivaTip.toLowerCase().includes(query) ||
      card.badge.toLowerCase().includes(query);
    return matchesCategory && matchesQuery;
  });

  return (
    <div
      className="guide-modal-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="guide-modal-container"
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="guide-modal-title"
      >
        {/* Header */}
        <div className="guide-modal-header">
          <div className="guide-header-title-group">
            <div className="guide-header-eyebrow">
              <BookOpen size={15} />
              <span>STOCKSENSE AI HUMAN GUIDE & VIVA COMPANION</span>
            </div>
            <h2 id="guide-modal-title">Plain English Guide & Viva Rehearsal</h2>
            <p>
              Every financial term, AI architecture decision, and error metric explained in plain human language.
            </p>
          </div>
          <button
            type="button"
            className="guide-close-btn"
            onClick={onClose}
            aria-label="Close guide"
          >
            <X size={18} />
          </button>
        </div>

        {/* Search & Category Filter Toolbar */}
        <div className="guide-toolbar">
          <div className="guide-search-wrap">
            <Search size={16} />
            <input
              type="search"
              placeholder="Search concepts, formulas, or viva questions (e.g. bps, MAE, stationarity, random walk)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              aria-label="Search guide concepts"
              autoFocus
            />
            {searchQuery && (
              <button
                type="button"
                className="guide-search-clear"
                onClick={() => setSearchQuery('')}
              >
                <X size={13} />
              </button>
            )}
          </div>

          <div className="guide-categories" role="tablist">
            {CATEGORIES.map((cat) => (
              <button
                key={cat}
                type="button"
                role="tab"
                aria-selected={activeCategory === cat}
                className={`guide-category-pill ${activeCategory === cat ? 'is-active' : ''}`}
                onClick={() => setActiveCategory(cat)}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Cards Grid */}
        <div className="guide-cards-grid">
          {filteredCards.length > 0 ? (
            filteredCards.map((card) => {
              const IconComponent = card.icon;
              return (
                <div key={card.id} className="guide-concept-card">
                  <div className="guide-card-top">
                    <div className="guide-card-icon-wrap">
                      <IconComponent size={18} />
                    </div>
                    <div className="guide-card-badge-wrap">
                      <span className="guide-card-cat">{card.category}</span>
                      <span className="guide-card-badge">{card.badge}</span>
                    </div>
                  </div>

                  <h3 className="guide-card-title">{card.title}</h3>

                  <p className="guide-card-summary">{card.summary}</p>

                  <div className="guide-card-body">
                    <p>{card.explanation}</p>
                  </div>

                  <div className="guide-card-viva">
                    <div className="guide-viva-header">
                      <Lightbulb size={13} color="#D97706" />
                      <strong>Viva Examiner Tip / How to explain:</strong>
                    </div>
                    <p>{card.vivaTip}</p>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="guide-empty-state">
              <HelpCircle size={32} />
              <h3>No matching concepts found</h3>
              <p>Try searching for "bps", "MAE", "returns", or "CNN-LSTM".</p>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setSearchQuery('');
                  setActiveCategory('All');
                }}
              >
                Reset Search
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="guide-modal-footer">
          <div className="guide-footer-left">
            <CheckCircle size={14} color="#117A56" />
            <span>Academic Integrity: Zero Data Leakage · Pure Price-Action Modeling · Honest Uncertainty</span>
          </div>
          <button type="button" className="btn-primary" onClick={onClose}>
            Got It, Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
