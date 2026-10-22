import React from 'react';
import { Clock, TrendingDown, TrendingUp } from 'lucide-react';

const price = (value) => value === null || value === undefined || !Number.isFinite(Number(value))
  ? '—'
  : `₹${Number(value).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const volume = (value) => value === null || value === undefined || !Number.isFinite(Number(value))
  ? '—'
  : Number(value).toLocaleString('en-IN');

function DataStatus({ status, source }) {
  const normalized = String(status || 'UNKNOWN').toUpperCase();
  const label = normalized === 'LIVE' ? 'Live feed'
    : normalized === 'STALE' ? 'Cached data'
      : normalized === 'FALLBACK' || normalized === 'OFFLINE' ? 'Fallback data'
        : normalized === 'UNKNOWN' ? 'Quote status unknown' : normalized.replaceAll('_', ' ');
  const className = normalized === 'LIVE' ? 'status-live'
    : normalized === 'STALE' ? 'status-stale'
      : normalized === 'FALLBACK' || normalized === 'OFFLINE' ? 'status-fallback' : 'status-unknown';
  return <span className={`data-status ${className}`}><i />{label}{source ? ` · ${source}` : ''}</span>;
}

export default function StockHeader({ info, quote, loading = false }) {
  if (loading && !info && !quote) {
    return <section className="stock-overview fintech-card" aria-label="Loading selected equity" aria-busy="true">
      <div className="hero-copy"><span className="section-index">SELECTED EQUITY</span><div className="skeleton-shimmer hero-skeleton-title" /><div className="skeleton-shimmer hero-skeleton-price" /></div>
      <div className="hero-stats">{[1, 2, 3, 4].map((key) => <div className="skeleton-shimmer hero-skeleton-stat" key={key} />)}</div>
    </section>;
  }

  const symbol = quote?.symbol || info?.symbol || 'NSE EQUITY';
  const displaySymbol = quote?.display_symbol || symbol.replace(/\.NS$/i, '');
  const name = info?.name || displaySymbol;
  const isPositive = Number(quote?.change) >= 0;
  const changeAvailable = quote?.change !== null && quote?.change !== undefined;
  const updated = quote?.last_updated || quote?.timestamp;
  const timeLabel = updated ? new Date(updated).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata' }) : null;
  const dayRange = quote?.day_low !== null && quote?.day_low !== undefined && quote?.day_high !== null && quote?.day_high !== undefined
    ? `${price(quote.day_low)} — ${price(quote.day_high)}` : '—';
  const range52 = info?.fifty_two_week_low && info?.fifty_two_week_high
    ? `${price(info.fifty_two_week_low)} — ${price(info.fifty_two_week_high)}` : '—';

  return (
    <section className="stock-overview fintech-card" aria-labelledby="stock-title">
      <div className="hero-copy">
        <span className="section-index stock-eyebrow">SELECTED EQUITY <i /> NSE / CASH</span>
        <div className="stock-title-row">
          <h1 id="stock-title">{displaySymbol}</h1>
          <span className="stock-sector">{info?.sector || 'Listed equity'}</span>
        </div>
        <p className="stock-company">{name}</p>
        <div className="hero-price-row">
          <strong className="hero-price tabular-num">{quote ? price(quote.price) : '—'}</strong>
          {changeAvailable && <span className={`hero-change ${isPositive ? 'is-positive' : 'is-negative'}`}>
            {isPositive ? <TrendingUp size={15} /> : <TrendingDown size={15} />}
            {isPositive ? '+' : ''}{Number(quote.change).toFixed(2)} ({isPositive ? '+' : ''}{Number(quote.change_pct).toFixed(2)}%)
          </span>}
        </div>
        <div className="hero-source-line">
          <DataStatus status={quote?.data_status} source={quote?.source} />
          {timeLabel && <span className="quote-time"><Clock size={13} /> Updated {timeLabel} IST</span>}
          {!quote && <span className="quote-time">Quote unavailable · retrying automatically</span>}
        </div>
      </div>

      <dl className="hero-stats">
        <div className="hero-stat hero-stat-range"><dt>DAY RANGE</dt><dd className="tabular-num">{dayRange}</dd></div>
        <div className="hero-stat"><dt>VOLUME</dt><dd className="tabular-num">{volume(quote?.volume)}</dd></div>
        <div className="hero-stat"><dt>52-WEEK RANGE</dt><dd className="tabular-num">{range52}</dd></div>
        <div className="hero-stat"><dt>MARKET CAP</dt><dd className="tabular-num">{info?.market_cap ? `₹${(Number(info.market_cap) / 1e7).toLocaleString('en-IN', { maximumFractionDigits: 2 })} Cr` : '—'}</dd></div>
      </dl>
    </section>
  );
}
