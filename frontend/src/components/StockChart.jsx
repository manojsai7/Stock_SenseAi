import React, { useMemo, useState } from 'react';
import { Activity, AlertCircle, RefreshCw } from 'lucide-react';
import { CandlestickChart } from './charts/candlestick-chart';
import { Candlestick } from './charts/candlestick';
import { Grid } from './charts/grid';
import { XAxis } from './charts/x-axis';
import { YAxis } from './charts/y-axis';
import { ChartTooltip } from './charts/tooltip/chart-tooltip';
import { useChart } from './charts/chart-context';
import { triggerHaptic } from '../lib/haptics';

const PERIODS = [
  { key: '1mo', label: '1M' },
  { key: '3mo', label: '3M' },
  { key: '6mo', label: '6M' },
  { key: '1y', label: '1Y' },
  { key: '5y', label: '5Y' },
];

const numberValue = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const result = Number(value);
  return Number.isFinite(result) && result > 0 ? result : null;
};

function formatPrice(value, currency) {
  return `${currency}${Number(value).toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function MovingAverageOverlay({ color = '#7a8d37' }) {
  const { data, xScale, yScale, xAccessor } = useChart();
  const d = data
    .filter((point) => numberValue(point.ma20))
    .map((point) => {
      const x = xScale(xAccessor(point));
      const y = yScale(Number(point.ma20));
      return `${x === undefined ? '' : `${x},${y}`}`;
    })
    .filter(Boolean)
    .map((point, index) => `${index ? 'L' : 'M'}${point}`)
    .join(' ');

  if (!d) return null;
  return <path d={d} fill="none" stroke={color} strokeDasharray="5 4" strokeWidth="2" vectorEffect="non-scaling-stroke" />;
}

export default function StockChart({
  historyData = [],
  period = '1y',
  onPeriodChange,
  loading = false,
  error = null,
  onRetry,
  currency = '₹',
}) {
  const [showMA20, setShowMA20] = useState(true);

  const candles = useMemo(() => {
    if (!Array.isArray(historyData)) return [];
    return historyData.map((item) => {
      const date = new Date(item?.date);
      const open = numberValue(item?.open);
      const high = numberValue(item?.high);
      const low = numberValue(item?.low);
      const close = numberValue(item?.close);
      if (!item?.date || Number.isNaN(date.getTime()) || !open || !high || !low || !close) return null;
      if (high < Math.max(open, close) || low > Math.min(open, close) || high < low) return null;
      const ma20 = numberValue(item?.ma20);
      return { ...item, date, open, high, low, close, ma20, volume: Number(item?.volume) || 0 };
    }).filter(Boolean).sort((a, b) => a.date - b.date);
  }, [historyData]);

  const hasMA20 = candles.some((point) => point.ma20 !== null);

  const tooltipRows = (point) => [
    { color: '#69766e', label: 'Open', value: formatPrice(point.open, currency) },
    { color: '#117a56', label: 'High', value: formatPrice(point.high, currency) },
    { color: '#bd5147', label: 'Low', value: formatPrice(point.low, currency) },
    { color: '#151c18', label: 'Close', value: formatPrice(point.close, currency) },
    ...(point.ma20 !== null ? [{ color: '#7a8d37', label: '20-day average', value: formatPrice(point.ma20, currency) }] : []),
    { color: '#89938c', label: 'Volume', value: Number(point.volume).toLocaleString('en-IN') },
  ];

  return (
    <section className="fintech-card chart-card" aria-labelledby="price-history-title">
      <div className="chart-toolbar">
        <div className="chart-heading">
          <span className="section-index">01 / MARKET SERIES</span>
          <h2 id="price-history-title">Price history</h2>
          <p>Daily OHLC data · NSE sessions</p>
        </div>
        <div className="chart-controls">
          <div className="chart-periods" role="group" aria-label="Chart period">
            {PERIODS.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`period-button ${period === item.key ? 'is-active' : ''}`}
                aria-pressed={period === item.key}
                onClick={() => {
                  triggerHaptic();
                  onPeriodChange?.(item.key);
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            className={`indicator-toggle ${showMA20 && hasMA20 ? 'is-active' : ''}`}
            aria-pressed={showMA20 && hasMA20}
            disabled={!hasMA20}
            title={hasMA20 ? 'Toggle the API supplied 20-day moving average' : 'The selected history response has no valid MA20 values'}
            onClick={() => {
              triggerHaptic();
              setShowMA20((value) => !value);
            }}
          >
            <span className="indicator-swatch" />
            <span>MA20</span>
          </button>
        </div>
      </div>

      <div className="chart-legend" aria-live="polite">
        <span><i className="legend-candle-up" /> Up session</span>
        <span><i className="legend-candle-down" /> Down session</span>
        {hasMA20 && <span><i className="legend-average" /> 20-day average</span>}
        {!hasMA20 && <span className="legend-note">MA20 is not included in this API response.</span>}
      </div>

      <div className="chart-stage" aria-busy={loading}>
        {loading ? (
          <div className="chart-state" role="status">
            <div className="chart-skeleton" />
            <span>Loading historical sessions…</span>
          </div>
        ) : error ? (
          <div className="chart-state chart-error" role="alert">
            <AlertCircle size={22} />
            <strong>History unavailable</strong>
            <span>{error}</span>
            {onRetry && <button type="button" className="btn-secondary" onClick={onRetry}><RefreshCw size={14} /> Retry</button>}
          </div>
        ) : candles.length === 0 ? (
          <div className="chart-state">
            <Activity size={22} />
            <strong>No valid OHLC sessions for {period.toUpperCase()}</strong>
            <span>Choose another period or retry the request.</span>
            {onRetry && <button type="button" className="btn-secondary" onClick={onRetry}><RefreshCw size={14} /> Retry</button>}
          </div>
        ) : (
          <CandlestickChart
            data={candles}
            xDataKey="date"
            aspectRatio="2.35 / 1"
            animationDuration={700}
            revealSignature={`${period}-${candles[0].date.getTime()}-${candles.at(-1).date.getTime()}`}
            margin={{ top: 14, right: 58, bottom: 38, left: 12 }}
            className="ohlc-chart"
          >
            <Grid horizontal vertical={false} numTicksRows={5} fadeHorizontal={false} stroke="var(--chart-grid)" strokeDasharray="2 5" />
            <Candlestick animate positiveFill="var(--chart-positive)" negativeFill="var(--chart-negative)" />
            {showMA20 && hasMA20 && <MovingAverageOverlay />}
            <XAxis numTicks={6} tickerHalfWidth={34} />
            <YAxis orientation="right" numTicks={5} formatValue={(value) => Number(value).toLocaleString('en-IN', { maximumFractionDigits: 0 })} />
            <ChartTooltip
              rows={tooltipRows}
              showDatePill
              showCrosshair
              showDots={false}
              className="ohlc-tooltip"
              panelStyle={{ borderColor: 'var(--border-default)', backgroundColor: '#fff' }}
            />
          </CandlestickChart>
        )}
      </div>
      <p className="chart-caption">Candles are mapped from the selected symbol’s API history. No sample or generated price data is used.</p>
    </section>
  );
}
