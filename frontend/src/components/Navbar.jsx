import React, { useEffect, useRef, useState } from 'react';
import { BookOpen, Search, X } from 'lucide-react';
import { searchStocks } from '../services/api';
import { triggerHaptic } from '../lib/haptics';

const WATCHLIST_STOCKS = [
  { symbol: 'TCS.NS', label: 'TCS' },
  { symbol: 'RELIANCE.NS', label: 'RELIANCE' },
  { symbol: 'INFY.NS', label: 'INFY' },
  { symbol: 'HDFCBANK.NS', label: 'HDFC BANK' },
  { symbol: 'ICICIBANK.NS', label: 'ICICI BANK' },
];

export default function Navbar({ currentSymbol, onSelectSymbol, onOpenArchitecture, marketStatus = 'CLOSED' }) {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const searchContainerRef = useRef(null);

  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setSearchResults([]);
      setSearching(false);
      setSearchError('');
      return undefined;
    }
    let active = true;
    setSearching(true);
    setSearchError('');
    const timer = setTimeout(async () => {
      try {
        const results = await searchStocks(trimmed);
        if (active) setSearchResults(results || []);
      } catch (error) {
        if (active) setSearchError(error.message || 'Equity search is unavailable.');
      } finally {
        if (active) setSearching(false);
      }
    }, 250);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [query]);

  useEffect(() => {
    const onPointerDown = (event) => {
      if (!searchContainerRef.current?.contains(event.target)) setIsDropdownOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, []);

  const selectSymbol = (symbol) => {
    triggerHaptic();
    onSelectSymbol(symbol);
    setQuery('');
    setSearchResults([]);
    setIsDropdownOpen(false);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (searchResults.length) selectSymbol(searchResults[0].symbol);
      else if (query.trim()) {
        const symbol = query.trim().toUpperCase();
        selectSymbol(symbol.includes('.') ? symbol : `${symbol}.NS`);
      }
    }
    if (event.key === 'Escape') setIsDropdownOpen(false);
  };

  return (
    <header className="site-header">
      <div className="header-inner">
        <a className="wordmark" href="#top" aria-label="StockSense AI home">
          <span>StockSense <em>AI</em></span>
          <small>MARKET RESEARCH / INDIA</small>
        </a>

        <div className="header-market-state" aria-label={`Market status: ${marketStatus}`}>
          <span className={`market-light ${marketStatus === 'OPEN' ? 'is-open' : ''}`} />
          <span>{marketStatus.replaceAll('_', ' ')}</span>
          <span className="market-zone">NSE · IST</span>
        </div>

        <div className="header-actions">
          <div className="search-wrap" ref={searchContainerRef}>
            <Search aria-hidden="true" size={16} />
            <input
              type="search"
              aria-label="Search stocks by symbol or company"
              aria-expanded={Boolean(query.trim()) && isDropdownOpen}
              aria-controls="stock-search-results"
              placeholder="Search equities"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setIsDropdownOpen(true);
              }}
              onFocus={() => setIsDropdownOpen(true)}
              onKeyDown={handleKeyDown}
            />
            {query && <button className="search-clear" type="button" aria-label="Clear search" onClick={() => { setQuery(''); setSearchResults([]); }}><X size={14} /></button>}
            {isDropdownOpen && query.trim() && (
              <div className="search-results" id="stock-search-results" role="listbox" aria-label="Stock search results">
                {searching ? <div className="search-message" role="status">Searching equities…</div>
                  : searchError ? <div className="search-message" role="alert">{searchError}</div>
                    : searchResults.length ? searchResults.map((stock) => (
                      <button className="search-result" key={stock.symbol} type="button" role="option" aria-selected={currentSymbol === stock.symbol} onClick={() => selectSymbol(stock.symbol)}>
                        <span><strong>{stock.symbol}</strong><small>{stock.name}</small></span>
                        <small>{stock.exchange || 'NSE'}</small>
                      </button>
                    )) : <div className="search-message">No matching equities. Press Enter to try the NSE symbol.</div>}
              </div>
            )}
          </div>
          <button className="research-link" type="button" onClick={onOpenArchitecture}><BookOpen size={15} /><span>Methodology</span></button>
        </div>
      </div>

      <nav className="watchlist" aria-label="Popular NSE stocks">
        {WATCHLIST_STOCKS.map((stock) => (
          <button key={stock.symbol} type="button" onClick={() => selectSymbol(stock.symbol)} aria-pressed={currentSymbol === stock.symbol} className={`watchlist-item ${currentSymbol === stock.symbol ? 'is-active' : ''}`}>
            {stock.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
