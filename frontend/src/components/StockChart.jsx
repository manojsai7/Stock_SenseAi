import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { Sliders, Eye, TrendingUp } from 'lucide-react';

const PERIODS = [
  { key: '1mo', label: '1M' },
  { key: '3mo', label: '3M' },
  { key: '6mo', label: '6M' },
  { key: '1y', label: '1Y' },
  { key: '5y', label: '5Y' },
];

export default function StockChart({ historyData, period, onPeriodChange, loading, currency = '₹' }) {
  const [showMA20, setShowMA20] = useState(true);

  if (loading) {
    return (
      <div className="glass-panel" style={{ height: '420px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="pulse-glow" style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--accent-green)' }} />
          <span>Streaming Market Candle Sequences...</span>
        </div>
      </div>
    );
  }

  const data = historyData || [];

  // Calculate min and max for chart bounds
  const prices = data.map(d => d.close).filter(Boolean);
  const minPrice = prices.length ? Math.floor(Math.min(...prices) * 0.98) : 'auto';
  const maxPrice = prices.length ? Math.ceil(Math.max(...prices) * 1.02) : 'auto';

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div style={{
          background: 'rgba(14, 19, 31, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
          borderRadius: '8px',
          padding: '12px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
          fontFamily: 'var(--font-sans)',
          fontSize: '0.8rem',
        }}>
          <div style={{ color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>{label}</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '4px 12px', fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Close:</span>
            <span style={{ color: '#F8FAFC', fontWeight: 700 }}>{currency}{item.close?.toFixed(2)}</span>

            <span style={{ color: 'var(--text-muted)' }}>Open:</span>
            <span>{currency}{item.open?.toFixed(2)}</span>

            <span style={{ color: 'var(--text-muted)' }}>High:</span>
            <span style={{ color: '#34D399' }}>{currency}{item.high?.toFixed(2)}</span>

            <span style={{ color: 'var(--text-muted)' }}>Low:</span>
            <span style={{ color: '#FB7185' }}>{currency}{item.low?.toFixed(2)}</span>

            {item.ma20 && (
              <>
                <span style={{ color: '#06B6D4' }}>MA20:</span>
                <span style={{ color: '#06B6D4', fontWeight: 600 }}>{currency}{item.ma20?.toFixed(2)}</span>
              </>
            )}

            <span style={{ color: 'var(--text-muted)' }}>Volume:</span>
            <span>{Number(item.volume || 0).toLocaleString()}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-panel" style={{ padding: '20px 24px', marginBottom: '24px' }}>
      {/* Chart Toolbar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <h2 style={{
            fontFamily: 'var(--font-heading)',
            fontSize: '1.15rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
          }}>
            Historical Price Dynamics
          </h2>

          {/* Indicators Toggle */}
          <button
            onClick={() => setShowMA20(!showMA20)}
            className={`btn-secondary ${showMA20 ? 'active' : ''}`}
            style={{ padding: '4px 10px', fontSize: '0.75rem' }}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: showMA20 ? '#06B6D4' : 'var(--text-muted)',
              display: 'inline-block',
            }} />
            <span>MA20 (20-Day SMA)</span>
          </button>
        </div>

        {/* Time Period Switches */}
        <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.03)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          {PERIODS.map(p => (
            <button
              key={p.key}
              onClick={() => onPeriodChange(p.key)}
              style={{
                background: period === p.key ? 'rgba(16, 185, 129, 0.2)' : 'transparent',
                color: period === p.key ? '#34D399' : 'var(--text-secondary)',
                border: period === p.key ? '1px solid rgba(16, 185, 129, 0.4)' : 'none',
                padding: '4px 12px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Canvas */}
      <div style={{ width: '100%', height: '380px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.35}/>
                <stop offset="95%" stopColor="#10B981" stopOpacity={0.0}/>
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />

            <XAxis
              dataKey="date"
              stroke="var(--text-muted)"
              tickLine={false}
              fontSize={11}
              fontFamily="var(--font-mono)"
              tickFormatter={(str) => {
                const parts = str?.split('-');
                return parts ? `${parts[1]}/${parts[2]}` : '';
              }}
              minTickGap={40}
            />

            <YAxis
              domain={[minPrice, maxPrice]}
              stroke="var(--text-muted)"
              tickLine={false}
              fontSize={11}
              fontFamily="var(--font-mono)"
              tickFormatter={(val) => `${currency}${val}`}
              orientation="right"
            />

            <Tooltip content={<CustomTooltip />} />

            <Area
              type="monotone"
              dataKey="close"
              name="Close Price"
              stroke="#10B981"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#priceGradient)"
            />

            {showMA20 && (
              <Line
                type="monotone"
                dataKey="ma20"
                name="20-Day MA"
                stroke="#06B6D4"
                strokeWidth={1.8}
                strokeDasharray="4 4"
                dot={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
