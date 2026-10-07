import React from 'react';
import { TrendingUp, TrendingDown, Clock, Building, Activity, Database, CheckCircle, AlertCircle } from 'lucide-react';

export default function StockHeader({ info, quote, loading }) {
  if (loading && !info && !quote) {
    return (
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <div style={{ height: '32px', width: '240px', background: 'rgba(255,255,255,0.05)', borderRadius: '6px', marginBottom: '12px' }} />
        <div style={{ height: '48px', width: '180px', background: 'rgba(255,255,255,0.05)', borderRadius: '6px' }} />
      </div>
    );
  }

  const currentPrice = quote?.price ?? info?.current_price;
  const prevClose = quote?.previous_close ?? info?.previous_close;
  const changeVal = quote?.change ?? info?.change ?? 0;
  const changePct = quote?.change_pct ?? info?.change_pct ?? 0;
  const isPositive = changeVal >= 0;

  const symbol = quote?.symbol || info?.symbol || 'EQUITY';
  const exchange = info?.exchange || 'NSE';
  const sector = info?.sector || 'Large Cap';
  const name = info?.name || quote?.display_symbol || symbol;

  const currencySymbol = (quote?.currency === 'INR' || info?.currency === 'INR') ? '₹' : '$';
  const marketStatus = quote?.market_status || 'CLOSED';
  const dataStatus = quote?.data_status || 'LIVE';
  const lastUpdated = quote?.last_updated || quote?.timestamp;

  const formatPrice = (val) => {
    if (val === null || val === undefined) return '--';
    return `${currencySymbol}${Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatLargeNum = (val) => {
    if (!val) return '--';
    if (val >= 1e12) return `${(val / 1e12).toFixed(2)}T`;
    if (val >= 1e9) return `${(val / 1e9).toFixed(2)}B`;
    if (val >= 1e7) return `${(val / 1e7).toFixed(2)}Cr`;
    if (val >= 1e6) return `${(val / 1e6).toFixed(2)}M`;
    return Number(val).toLocaleString();
  };

  const formatTime = (isoStr) => {
    if (!isoStr) return '';
    try {
      const dt = new Date(isoStr);
      return dt.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return '';
    }
  };

  const renderMarketBadge = () => {
    switch (marketStatus) {
      case 'OPEN':
        return (
          <span style={{
            background: 'rgba(16, 185, 129, 0.15)',
            color: '#34D399',
            border: '1px solid rgba(16, 185, 129, 0.35)',
            fontSize: '0.72rem',
            fontWeight: 700,
            padding: '3px 10px',
            borderRadius: '20px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontFamily: 'var(--font-mono)'
          }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10B981', display: 'inline-block', boxShadow: '0 0 8px #10B981' }} />
            MARKET OPEN (09:15-15:30 IST)
          </span>
        );
      case 'PRE_MARKET':
        return (
          <span style={{
            background: 'rgba(245, 158, 11, 0.15)',
            color: '#FBBF24',
            border: '1px solid rgba(245, 158, 11, 0.35)',
            fontSize: '0.72rem',
            fontWeight: 700,
            padding: '3px 10px',
            borderRadius: '20px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontFamily: 'var(--font-mono)'
          }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#F59E0B', display: 'inline-block' }} />
            PRE-MARKET (09:00-09:15 IST)
          </span>
        );
      case 'POST_MARKET':
        return (
          <span style={{
            background: 'rgba(59, 130, 246, 0.15)',
            color: '#60A5FA',
            border: '1px solid rgba(59, 130, 246, 0.35)',
            fontSize: '0.72rem',
            fontWeight: 700,
            padding: '3px 10px',
            borderRadius: '20px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontFamily: 'var(--font-mono)'
          }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#3B82F6', display: 'inline-block' }} />
            POST-MARKET (15:30-16:00 IST)
          </span>
        );
      case 'WEEKEND':
        return (
          <span style={{
            background: 'rgba(255, 255, 255, 0.06)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.72rem',
            fontWeight: 600,
            padding: '3px 10px',
            borderRadius: '20px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontFamily: 'var(--font-mono)'
          }}>
            ● WEEKEND (MARKET CLOSED)
          </span>
        );
      case 'HOLIDAY':
        return (
          <span style={{
            background: 'rgba(255, 255, 255, 0.06)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.72rem',
            fontWeight: 600,
            padding: '3px 10px',
            borderRadius: '20px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontFamily: 'var(--font-mono)'
          }}>
            ● TRADING HOLIDAY
          </span>
        );
      default:
        return (
          <span style={{
            background: 'rgba(255, 255, 255, 0.06)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.72rem',
            fontWeight: 600,
            padding: '3px 10px',
            borderRadius: '20px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontFamily: 'var(--font-mono)'
          }}>
            ● MARKET CLOSED
          </span>
        );
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px' }}>
        {/* Left Info */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px', flexWrap: 'wrap' }}>
            <h1 style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '1.75rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.01em',
              margin: 0,
            }}>
              {symbol}
            </h1>
            <span style={{
              background: 'rgba(255, 255, 255, 0.08)',
              color: 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '3px 8px',
              borderRadius: '6px',
              fontFamily: 'var(--font-mono)',
            }}>
              {exchange}
            </span>
            <span style={{
              background: 'rgba(6, 182, 212, 0.1)',
              color: 'var(--accent-cyan)',
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '3px 8px',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}>
              <Building size={12} />
              {sector}
            </span>
            {renderMarketBadge()}
          </div>

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '14px' }}>
            {name}
          </p>

          {/* Current Price & Delta */}
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '14px', flexWrap: 'wrap' }}>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '2.4rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.02em',
            }}>
              {formatPrice(currentPrice)}
            </span>

            <div className={isPositive ? 'badge-up' : 'badge-down'} style={{ fontSize: '0.88rem', padding: '5px 12px' }}>
              {isPositive ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span>
                {isPositive ? '+' : ''}{Number(changeVal).toFixed(2)} ({isPositive ? '+' : ''}{Number(changePct).toFixed(2)}%)
              </span>
            </div>

            {/* Freshness & Attribution */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {lastUpdated && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Clock size={12} />
                  Updated: {formatTime(lastUpdated)} IST
                </span>
              )}
              <span>•</span>
              <span>Source: Yahoo Finance</span>
              {dataStatus === 'STALE' && (
                <span style={{
                  background: 'rgba(245, 158, 11, 0.15)',
                  color: '#FBBF24',
                  padding: '1px 6px',
                  borderRadius: '4px',
                  fontSize: '0.68rem',
                  fontWeight: 600
                }}>
                  STALE CACHE
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right Stats Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: '12px',
          minWidth: '320px',
        }}>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Day Range</div>
            <div style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {formatPrice(quote?.day_low ?? info?.day_low)} - {formatPrice(quote?.day_high ?? info?.day_high)}
            </div>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Volume (Day)</div>
            <div style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {formatLargeNum(quote?.volume ?? info?.volume)}
            </div>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>52W Range</div>
            <div style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {formatPrice(info?.fifty_two_week_low)} - {formatPrice(info?.fifty_two_week_high)}
            </div>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Market Cap</div>
            <div style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {currencySymbol}{formatLargeNum(info?.market_cap)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

