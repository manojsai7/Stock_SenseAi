import React from 'react';
import { Activity, Cpu, BookOpen, Search } from 'lucide-react';

const POPULAR_STOCKS = [
  { symbol: 'TCS.NS', label: 'TCS' },
  { symbol: 'RELIANCE.NS', label: 'RELIANCE' },
  { symbol: 'INFY.NS', label: 'INFOSYS' },
  { symbol: 'HDFCBANK.NS', label: 'HDFC BANK' },
  { symbol: 'ICICIBANK.NS', label: 'ICICI BANK' },
];

export default function Navbar({ currentSymbol, onSelectSymbol, onOpenArchitecture, searchQuery, setSearchQuery, onSearchSubmit }) {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(7, 9, 14, 0.85)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '12px 24px',
    }}>
      <div style={{
        maxWidth: '1440px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #8B5CF6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(16, 185, 129, 0.4)',
          }}>
            <Activity size={22} color="#FFFFFF" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{
                fontFamily: 'var(--font-heading)',
                fontSize: '1.35rem',
                fontWeight: 800,
                letterSpacing: '-0.02em',
                background: 'linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #10B981 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                StockSense AI
              </span>
              <span style={{
                fontSize: '0.68rem',
                fontFamily: 'var(--font-mono)',
                background: 'rgba(16, 185, 129, 0.15)',
                color: '#10B981',
                padding: '2px 8px',
                borderRadius: '6px',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                fontWeight: 600,
              }}>
                CNN-LSTM v1.0
              </span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Deep Learning Stock Price Forecasting
            </div>
          </div>
        </div>

        {/* Quick Tickers & Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {POPULAR_STOCKS.map((s) => {
              const isActive = currentSymbol === s.symbol;
              return (
                <button
                  key={s.symbol}
                  onClick={() => onSelectSymbol(s.symbol)}
                  className={`btn-secondary ${isActive ? 'active' : ''}`}
                  style={{
                    padding: '6px 12px',
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                  }}
                >
                  {s.label}
                </button>
              );
            })}
          </div>

          {/* Search Box */}
          <form onSubmit={onSearchSubmit} style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
            <Search size={15} color="var(--text-muted)" style={{ position: 'absolute', left: '12px' }} />
            <input
              type="text"
              placeholder="Search ticker (e.g. TCS.NS, NVDA)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '10px',
                padding: '7px 12px 7px 34px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontSize: '0.85rem',
                width: '210px',
                outline: 'none',
                transition: 'all 0.2s ease',
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--accent-green)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
            />
          </form>

          {/* Architecture / Viva Guide Button */}
          <button
            onClick={onOpenArchitecture}
            className="btn-secondary"
            style={{
              borderColor: 'rgba(139, 92, 246, 0.4)',
              background: 'rgba(139, 92, 246, 0.1)',
              color: '#C4B5FD',
            }}
          >
            <BookOpen size={15} />
            <span>Architecture & Viva</span>
          </button>

          {/* System Connection Status Pill */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.25)',
            borderRadius: '20px',
            padding: '4px 10px',
            fontSize: '0.72rem',
            fontWeight: 600,
            color: '#34D399',
            fontFamily: 'var(--font-mono)'
          }}>
            <span style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              background: '#10B981',
              boxShadow: '0 0 6px #10B981',
              display: 'inline-block'
            }} />
            <span>LIVE FEED</span>
          </div>
        </div>
      </div>
    </header>

  );
}
