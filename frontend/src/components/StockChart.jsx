import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { RefreshCw, AlertCircle, BarChart2 } from 'lucide-react';

const PERIODS = [
  { key: '1mo', label: '1M' },
  { key: '3mo', label: '3M' },
  { key: '6mo', label: '6M' },
  { key: '1y', label: '1Y' },
  { key: '5y', label: '5Y' },
];

export default function StockChart({
  historyData = [],
  period = '1y',
  onPeriodChange,
  loading = false,
  error = null,
  onRetry,
  currency = '₹'
}) {
  const [showMA20, setShowMA20] = useState(true);

  // Robust domain calculation to prevent Recharts [NaN, NaN] blank chart failure
  const { minPrice, maxPrice, validData } = useMemo(() => {
    if (!Array.isArray(historyData) || historyData.length === 0) {
      return { minPrice: 'auto', maxPrice: 'auto', validData: [] };
    }

    let min = Infinity;
    let max = -Infinity;
    const cleanList = [];

    for (let i = 0; i < historyData.length; i++) {
      const item = historyData[i];
      if (!item) continue;

      const closeVal = Number(item.close);
      if (typeof closeVal === 'number' && !isNaN(closeVal) && isFinite(closeVal) && closeVal > 0) {
        if (closeVal < min) min = closeVal;
        if (closeVal > max) max = closeVal;
        cleanList.push({
          ...item,
          close: closeVal,
          open: Number(item.open) || closeVal,
          high: Number(item.high) || closeVal,
          low: Number(item.low) || closeVal,
          volume: Number(item.volume) || 0,
          ma20: (item.ma20 !== null && item.ma20 !== undefined && !isNaN(Number(item.ma20))) ? Number(item.ma20) : null
        });
      }
    }

    if (!isFinite(min) || !isFinite(max) || cleanList.length === 0) {
      return { minPrice: 'auto', maxPrice: 'auto', validData: [] };
    }

    const padding = (max - min) * 0.04 || max * 0.02;
    return {
      minPrice: Math.max(0, Math.floor(min - padding)),
      maxPrice: Math.ceil(max + padding),
      validData: cleanList
    };
  }, [historyData]);

  // Formatter for Y-axis currency ticks
  const formatYAxis = (val) => {
    if (typeof val !== 'number' || isNaN(val)) return '';
    return `${currency}${Number(val).toLocaleString('en-IN')}`;
  };

  // Formatter for X-axis date labels
  const formatXAxis = (dateStr) => {
    if (!dateStr || typeof dateStr !== 'string') return '';
    const parts = dateStr.split('-');
    if (parts.length >= 3) {
      return `${parts[2]}/${parts[1]}`;
    }
    return dateStr;
  };

  // Custom financial tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div style={{
          backgroundColor: '#FFFFFF',
          border: '1px solid var(--border-default)',
          borderRadius: '8px',
          padding: '10px 14px',
          boxShadow: 'var(--shadow-md)',
          fontSize: '0.78125rem',
          minWidth: '180px',
        }}>
          <div style={{
            color: 'var(--text-secondary)',
            marginBottom: '6px',
            fontWeight: 600,
            borderBottom: '1px solid var(--border-subtle)',
            paddingBottom: '4px'
          }}>
            {label}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '4px 16px', fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Close</span>
            <span style={{ textAlign: 'right', fontWeight: 700, color: 'var(--text-primary)' }}>
              {currency}{item.close?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>

            <span style={{ color: 'var(--text-secondary)' }}>Open</span>
            <span style={{ textAlign: 'right', color: 'var(--text-primary)' }}>
              {currency}{item.open?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>

            <span style={{ color: 'var(--text-secondary)' }}>High</span>
            <span style={{ textAlign: 'right', color: 'var(--primary-green)' }}>
              {currency}{item.high?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>

            <span style={{ color: 'var(--text-secondary)' }}>Low</span>
            <span style={{ textAlign: 'right', color: 'var(--accent-red)' }}>
              {currency}{item.low?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>

            {item.ma20 && (
              <>
                <span style={{ color: 'var(--accent-blue)' }}>20-Day SMA</span>
                <span style={{ textAlign: 'right', color: 'var(--accent-blue)', fontWeight: 600 }}>
                  {currency}{item.ma20?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </>
            )}

            <span style={{ color: 'var(--text-secondary)' }}>Volume</span>
            <span style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
              {Number(item.volume || 0).toLocaleString('en-IN')}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fintech-card" style={{ padding: '20px 24px', marginBottom: '20px' }}>
      {/* Chart Toolbar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '16px',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        {/* Left: Section Title & Indicator Toggles */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          <div>
            <h2 style={{
              fontSize: '1rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.01em'
            }}>
              Price History
            </h2>
          </div>

          {/* MA20 Overlay Toggle */}
          <button
            type="button"
            onClick={() => setShowMA20(!showMA20)}
            className={`btn-secondary ${showMA20 ? 'active' : ''}`}
            style={{ padding: '4px 10px', fontSize: '0.75rem' }}
            aria-pressed={showMA20}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: showMA20 ? 'var(--accent-blue)' : 'var(--text-muted)',
              display: 'inline-block'
            }} />
            <span>20-Day SMA</span>
          </button>
        </div>

        {/* Right: Period Switches */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          backgroundColor: 'var(--surface-subtle)',
          padding: '3px',
          borderRadius: '8px',
          border: '1px solid var(--border-subtle)'
        }}>
          {PERIODS.map(p => {
            const isSelected = period === p.key;
            return (
              <button
                key={p.key}
                type="button"
                onClick={() => onPeriodChange(p.key)}
                style={{
                  backgroundColor: isSelected ? 'var(--surface-card)' : 'transparent',
                  color: isSelected ? 'var(--primary-green)' : 'var(--text-secondary)',
                  border: isSelected ? '1px solid var(--border-subtle)' : '1px solid transparent',
                  padding: '4px 11px',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: isSelected ? 700 : 500,
                  cursor: 'pointer',
                  boxShadow: isSelected ? 'var(--shadow-sm)' : 'none',
                  transition: 'all 0.15s ease'
                }}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Chart Body / State Handlers */}
      <div style={{ width: '100%', height: '360px', position: 'relative' }}>
        {loading ? (
          /* Layout-stable Loading Skeleton */
          <div style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            padding: '16px 0'
          }}>
            <div className="skeleton-shimmer" style={{ width: '100%', height: '280px', borderRadius: '8px' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px' }}>
              {[1, 2, 3, 4, 5, 6].map(i => (
                <div key={i} className="skeleton-shimmer" style={{ width: '48px', height: '12px' }} />
              ))}
            </div>
          </div>
        ) : error ? (
          /* Error State */
          <div style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
            backgroundColor: 'var(--surface-subtle)',
            borderRadius: '8px',
            border: '1px dashed var(--border-default)',
            padding: '24px',
            textAlign: 'center'
          }}>
            <AlertCircle size={28} color="var(--accent-red)" />
            <div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                Unable to load chart data
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', maxWidth: '340px' }}>
                {error}
              </div>
            </div>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="btn-secondary"
                style={{ fontSize: '0.8125rem', marginTop: '4px' }}
              >
                <RefreshCw size={14} />
                <span>Retry Chart</span>
              </button>
            )}
          </div>
        ) : validData.length === 0 ? (
          /* Empty State */
          <div style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            backgroundColor: 'var(--surface-subtle)',
            borderRadius: '8px',
            border: '1px dashed var(--border-default)',
            padding: '24px',
            textAlign: 'center'
          }}>
            <BarChart2 size={32} color="var(--text-muted)" />
            <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              No historical data available for period "{period.toUpperCase()}"
            </div>
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
              Try selecting a wider date range or retry fetching.
            </p>
            <button
              type="button"
              onClick={() => onPeriodChange('1y')}
              className="btn-secondary"
              style={{ fontSize: '0.8125rem', marginTop: '6px' }}
            >
              Reset to 1Y View
            </button>
          </div>
        ) : (
          /* Successful Chart Render */
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={validData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="fintechGreenGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#159B62" stopOpacity={0.18} />
                  <stop offset="100%" stopColor="#159B62" stopOpacity={0.0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#E4EAE6" vertical={false} />

              <XAxis
                dataKey="date"
                stroke="var(--text-muted)"
                tickLine={false}
                fontSize={11}
                fontFamily="var(--font-mono)"
                tickFormatter={formatXAxis}
                minTickGap={40}
              />

              <YAxis
                domain={[minPrice, maxPrice]}
                stroke="var(--text-muted)"
                tickLine={false}
                fontSize={11}
                fontFamily="var(--font-mono)"
                tickFormatter={formatYAxis}
                orientation="right"
                width={70}
              />

              <Tooltip content={<CustomTooltip />} />

              <Area
                type="monotone"
                dataKey="close"
                name="Closing Price"
                stroke="#159B62"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#fintechGreenGradient)"
                isAnimationActive={false}
              />

              {showMA20 && (
                <Line
                  type="monotone"
                  dataKey="ma20"
                  name="20-Day SMA"
                  stroke="#0284C7"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  isAnimationActive={false}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
