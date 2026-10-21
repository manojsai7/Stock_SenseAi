import React from 'react';
import { TrendingUp, TrendingDown, Clock, Building, ShieldCheck, AlertCircle } from 'lucide-react';

export default function StockHeader({ info, quote, loading = false }) {
  if (loading && !info && !quote) {
    return (
      <div className="fintech-card" style={{ padding: '24px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
          <div>
            <div className="skeleton-shimmer" style={{ width: '160px', height: '24px', marginBottom: '8px' }} />
            <div className="skeleton-shimmer" style={{ width: '220px', height: '16px', marginBottom: '16px' }} />
            <div className="skeleton-shimmer" style={{ width: '180px', height: '36px' }} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 110px)', gap: '12px' }}>
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="skeleton-shimmer" style={{ height: '60px', borderRadius: '8px' }} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Quote data takes precedence for live pricing, info data provides fundamentals
  const currentPrice = quote?.price ?? info?.current_price;
  const changeVal = quote?.change ?? info?.change ?? 0;
  const changePct = quote?.change_pct ?? info?.change_pct ?? 0;
  const isPositive = changeVal >= 0;

  const symbol = quote?.symbol || info?.symbol || 'NSE Equity';
  const exchange = info?.exchange || 'NSE';
  const sector = info?.sector || 'Equities';
  const name = info?.name || quote?.display_symbol || symbol;

  const currencySymbol = '₹';
  const marketStatus = quote?.market_status || 'CLOSED';
  const dataStatus = quote?.data_status || 'LIVE';
  const lastUpdated = quote?.last_updated || quote?.timestamp;

  const formatPrice = (val) => {
    if (val === null || val === undefined || isNaN(Number(val))) return '—';
    return `${currencySymbol}${Number(val).toLocaleString('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  };

  const formatLargeINR = (val) => {
    if (val === null || val === undefined || isNaN(Number(val)) || Number(val) === 0) return '—';
    const num = Number(val);
    // Indian Crore representation (1 Cr = 10,000,000)
    if (num >= 1e7) {
      return `${currencySymbol}${(num / 1e7).toLocaleString('en-IN', { maximumFractionDigits: 2 })} Cr`;
    }
    if (num >= 1e5) {
      return `${currencySymbol}${(num / 1e5).toLocaleString('en-IN', { maximumFractionDigits: 2 })} Lakh`;
    }
    return `${currencySymbol}${num.toLocaleString('en-IN')}`;
  };

  const formatVolume = (val) => {
    if (val === null || val === undefined || isNaN(Number(val)) || Number(val) === 0) return '—';
    return Number(val).toLocaleString('en-IN');
  };

  const formatTime = (isoStr) => {
    if (!isoStr) return null;
    try {
      const dt = new Date(isoStr);
      return dt.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return null;
    }
  };

  const renderMarketBadge = () => {
    switch (marketStatus) {
      case 'OPEN':
        return (
          <span style={{
            backgroundColor: 'var(--green-tint)',
            color: 'var(--primary-green)',
            border: '1px solid var(--green-border)',
            fontSize: '0.6875rem',
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: '4px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--primary-green)' }} />
            MARKET OPEN
          </span>
        );
      case 'PRE_MARKET':
        return (
          <span style={{
            backgroundColor: 'var(--amber-tint)',
            color: 'var(--accent-amber)',
            border: '1px solid var(--amber-border)',
            fontSize: '0.6875rem',
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: '4px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--accent-amber)' }} />
            PRE-MARKET
          </span>
        );
      case 'POST_MARKET':
        return (
          <span style={{
            backgroundColor: 'var(--blue-tint)',
            color: 'var(--accent-blue)',
            border: '1px solid var(--blue-border)',
            fontSize: '0.6875rem',
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: '4px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)' }} />
            POST-MARKET
          </span>
        );
      default:
        return (
          <span style={{
            backgroundColor: 'var(--surface-subtle)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.6875rem',
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: '4px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--text-muted)' }} />
            MARKET CLOSED
          </span>
        );
    }
  };

  const dayLow = quote?.day_low ?? info?.day_low;
  const dayHigh = quote?.day_high ?? info?.day_high;
  const week52Low = info?.fifty_two_week_low;
  const week52High = info?.fifty_two_week_high;
  const volume = quote?.volume ?? info?.volume;
  const marketCap = info?.market_cap;
  const timeFormatted = formatTime(lastUpdated);

  return (
    <div className="fintech-card" style={{ padding: '24px', marginBottom: '20px' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        gap: '24px'
      }}>
        {/* Left: Company Identity & Headline Quote */}
        <div style={{ flex: '1 1 320px' }}>
          {/* Header Row: Symbol, Exchange, Sector, Market Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
            <h1 style={{
              fontSize: '1.5rem',
              fontWeight: 800,
              color: 'var(--text-primary)',
              letterSpacing: '-0.02em',
              margin: 0
            }}>
              {symbol}
            </h1>
            <span style={{
              backgroundColor: 'var(--surface-subtle)',
              color: 'var(--text-secondary)',
              fontSize: '0.6875rem',
              fontWeight: 600,
              padding: '2px 6px',
              borderRadius: '4px',
              fontFamily: 'var(--font-mono)'
            }}>
              {exchange}
            </span>
            {sector && (
              <span style={{
                backgroundColor: 'var(--surface-subtle)',
                color: 'var(--text-secondary)',
                fontSize: '0.6875rem',
                fontWeight: 500,
                padding: '2px 6px',
                borderRadius: '4px'
              }}>
                {sector}
              </span>
            )}
            {renderMarketBadge()}
          </div>

          {/* Company Full Name */}
          <div style={{
            color: 'var(--text-secondary)',
            fontSize: '0.875rem',
            marginBottom: '16px'
          }}>
            {name}
          </div>

          {/* Headline Price & Daily Movement */}
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap' }}>
            <span style={{
              fontSize: '2.25rem',
              fontWeight: 800,
              color: 'var(--text-primary)',
              letterSpacing: '-0.03em',
              lineHeight: 1
            }} className="tabular-num">
              {formatPrice(currentPrice)}
            </span>

            <div className={isPositive ? 'badge-up' : 'badge-down'}>
              {isPositive ? <TrendingUp size={15} /> : <TrendingDown size={15} />}
              <span>
                {isPositive ? '+' : ''}{Number(changeVal).toFixed(2)} ({isPositive ? '+' : ''}{Number(changePct).toFixed(2)}%)
              </span>
            </div>
          </div>

          {/* Source and Data Freshness Strip */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            marginTop: '10px',
            flexWrap: 'wrap'
          }}>
            {timeFormatted && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                <Clock size={12} />
                <span>Updated {timeFormatted} IST</span>
              </span>
            )}
            <span>•</span>
            <span>NSE Official Sessions</span>
            {dataStatus === 'STALE' && (
              <span style={{
                backgroundColor: 'var(--amber-tint)',
                color: 'var(--accent-amber)',
                fontSize: '0.6875rem',
                padding: '1px 5px',
                borderRadius: '3px',
                fontWeight: 600
              }}>
                Cached
              </span>
            )}
          </div>
        </div>

        {/* Right: Key Stats Summary Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: '10px',
          flex: '1 1 360px',
          maxWidth: '560px'
        }}>
          {/* Day Range */}
          <div style={{
            backgroundColor: 'var(--surface-subtle)',
            padding: '10px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)'
          }}>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', marginBottom: '3px', fontWeight: 500 }}>
              Day Range
            </div>
            <div className="tabular-num" style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {dayLow && dayHigh ? `${formatPrice(dayLow)} - ${formatPrice(dayHigh)}` : '—'}
            </div>
          </div>

          {/* Volume */}
          <div style={{
            backgroundColor: 'var(--surface-subtle)',
            padding: '10px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)'
          }}>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', marginBottom: '3px', fontWeight: 500 }}>
              Session Volume
            </div>
            <div className="tabular-num" style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {formatVolume(volume)}
            </div>
          </div>

          {/* 52-Week Range */}
          <div style={{
            backgroundColor: 'var(--surface-subtle)',
            padding: '10px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)'
          }}>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', marginBottom: '3px', fontWeight: 500 }}>
              52-Week Range
            </div>
            <div className="tabular-num" style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {week52Low && week52High ? `${formatPrice(week52Low)} - ${formatPrice(week52High)}` : '—'}
            </div>
          </div>

          {/* Market Cap */}
          <div style={{
            backgroundColor: 'var(--surface-subtle)',
            padding: '10px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)'
          }}>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', marginBottom: '3px', fontWeight: 500 }}>
              Market Cap
            </div>
            <div className="tabular-num" style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {formatLargeINR(marketCap)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
