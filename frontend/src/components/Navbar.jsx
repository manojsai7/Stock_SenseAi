import React, { useState, useEffect, useRef } from 'react';
import { Search, BookOpen, Activity, X } from 'lucide-react';
import { searchStocks } from '../services/api';

const WATCHLIST_STOCKS = [
  { symbol: 'TCS.NS', label: 'TCS', name: 'Tata Consultancy' },
  { symbol: 'RELIANCE.NS', label: 'RELIANCE', name: 'Reliance Industries' },
  { symbol: 'INFY.NS', label: 'INFY', name: 'Infosys' },
  { symbol: 'HDFCBANK.NS', label: 'HDFCBANK', name: 'HDFC Bank' },
  { symbol: 'ICICIBANK.NS', label: 'ICICIBANK', name: 'ICICI Bank' },
];

export default function Navbar({
  currentSymbol,
  onSelectSymbol,
  onOpenArchitecture,
  marketStatus = 'CLOSED'
}) {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const searchContainerRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Debounced search when user types
  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setSearchResults([]);
      setSearching(false);
      return;
    }

    setSearching(true);
    const timer = setTimeout(async () => {
      try {
        const results = await searchStocks(trimmed);
        setSearchResults(results || []);
        setIsDropdownOpen(true);
      } catch (err) {
        console.warn('Search error:', err);
      } finally {
        setSearching(false);
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [query]);

  const handleSelectFromSearch = (symbol) => {
    onSelectSymbol(symbol);
    setQuery('');
    setIsDropdownOpen(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (searchResults.length > 0) {
        handleSelectFromSearch(searchResults[0].symbol);
      } else if (query.trim()) {
        const formatted = query.trim().toUpperCase();
        handleSelectFromSearch(formatted.includes('.') ? formatted : `${formatted}.NS`);
      }
    } else if (e.key === 'Escape') {
      setIsDropdownOpen(false);
    }
  };

  return (
    <header style={{
      backgroundColor: '#FFFFFF',
      borderBottom: '1px solid var(--border-subtle)',
      position: 'sticky',
      top: 0,
      zIndex: 40,
      boxShadow: 'var(--shadow-sm)'
    }}>
      <div style={{
        maxWidth: '1440px',
        margin: '0 auto',
        padding: '12px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Brand identity */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            backgroundColor: 'var(--primary-green)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#FFFFFF'
          }}>
            <Activity size={18} strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{
                fontSize: '1.125rem',
                fontWeight: 800,
                color: 'var(--text-primary)',
                letterSpacing: '-0.02em'
              }}>
                StockSense AI
              </span>
              <span style={{
                fontSize: '0.6875rem',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                backgroundColor: 'var(--surface-subtle)',
                padding: '1px 6px',
                borderRadius: '4px'
              }}>
                NSE Forecasting
              </span>
            </div>
          </div>
        </div>

        {/* Watchlist Quick Tickers */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          overflowX: 'auto',
          maxWidth: '100%',
          paddingBottom: '2px'
        }}>
          {WATCHLIST_STOCKS.map(item => {
            const isActive = currentSymbol === item.symbol;
            return (
              <button
                key={item.symbol}
                type="button"
                onClick={() => onSelectSymbol(item.symbol)}
                style={{
                  backgroundColor: isActive ? 'var(--green-tint)' : 'var(--surface-subtle)',
                  color: isActive ? 'var(--primary-green)' : 'var(--text-secondary)',
                  border: isActive ? '1px solid var(--green-border)' : '1px solid var(--border-subtle)',
                  padding: '5px 12px',
                  borderRadius: '6px',
                  fontSize: '0.78125rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: isActive ? 700 : 500,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.15s ease'
                }}
                aria-pressed={isActive}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        {/* Search & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Search Box */}
          <div ref={searchContainerRef} style={{ position: 'relative' }}>
            <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
              <Search size={14} color="var(--text-muted)" style={{ position: 'absolute', left: '10px', pointerEvents: 'none' }} />
              <input
                type="text"
                placeholder="Search symbol (e.g. TCS)..."
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setIsDropdownOpen(true);
                }}
                onFocus={() => setIsDropdownOpen(true)}
                onKeyDown={handleKeyDown}
                style={{
                  backgroundColor: 'var(--surface-subtle)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '6px',
                  padding: '6px 28px 6px 30px',
                  color: 'var(--text-primary)',
                  fontSize: '0.8125rem',
                  fontFamily: 'var(--font-sans)',
                  width: '210px',
                  outline: 'none',
                  transition: 'border-color 0.15s ease'
                }}
              />
              {query && (
                <button
                  type="button"
                  onClick={() => {
                    setQuery('');
                    setSearchResults([]);
                    setIsDropdownOpen(false);
                  }}
                  style={{
                    position: 'absolute',
                    right: '8px',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Live Search Suggestion Dropdown */}
            {isDropdownOpen && query.trim() && (
              <div style={{
                position: 'absolute',
                top: 'calc(100% + 4px)',
                right: 0,
                width: '280px',
                backgroundColor: '#FFFFFF',
                border: '1px solid var(--border-default)',
                borderRadius: '8px',
                boxShadow: 'var(--shadow-lg)',
                zIndex: 60,
                maxHeight: '260px',
                overflowY: 'auto',
                padding: '4px'
              }}>
                {searching ? (
                  <div style={{ padding: '12px', fontSize: '0.78125rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                    Searching equities...
                  </div>
                ) : searchResults.length > 0 ? (
                  searchResults.map(stock => (
                    <button
                      key={stock.symbol}
                      type="button"
                      onClick={() => handleSelectFromSearch(stock.symbol)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        padding: '8px 10px',
                        borderRadius: '4px',
                        background: currentSymbol === stock.symbol ? 'var(--green-tint)' : 'none',
                        border: 'none',
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '0.8125rem',
                        transition: 'background-color 0.1s ease'
                      }}
                      onMouseEnter={(e) => {
                        if (currentSymbol !== stock.symbol) e.currentTarget.style.backgroundColor = 'var(--surface-subtle)';
                      }}
                      onMouseLeave={(e) => {
                        if (currentSymbol !== stock.symbol) e.currentTarget.style.backgroundColor = 'transparent';
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {stock.symbol}
                        </div>
                        <div style={{ fontSize: '0.71875rem', color: 'var(--text-secondary)' }}>
                          {stock.name}
                        </div>
                      </div>
                      <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                        {stock.exchange}
                      </span>
                    </button>
                  ))
                ) : (
                  <div style={{ padding: '12px', fontSize: '0.78125rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                    No matching equities found
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Architecture & Viva Button */}
          <button
            type="button"
            onClick={onOpenArchitecture}
            className="btn-secondary"
            style={{ fontSize: '0.78125rem', padding: '6px 12px' }}
          >
            <BookOpen size={14} color="var(--primary-green)" />
            <span>Architecture & Viva</span>
          </button>
        </div>
      </div>
    </header>
  );
}
