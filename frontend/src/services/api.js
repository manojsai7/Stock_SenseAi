const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '') + '/api';

export async function searchStocks(query = '') {
  const res = await fetch(`${API_BASE}/stocks/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) throw new Error('Failed to search stocks');
  return res.json();
}

export async function getStockInfo(symbol) {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}/info`);
  if (!res.ok) throw new Error(`Failed to load info for ${symbol}`);
  return res.json();
}

export async function getStockHistory(symbol, period = '1y') {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}/history?period=${period}`);
  if (!res.ok) throw new Error(`Failed to load history for ${symbol}`);
  return res.json();
}

export async function getPrediction(symbol) {
  const res = await fetch(`${API_BASE}/predictions/${encodeURIComponent(symbol)}/latest`);
  if (!res.ok) throw new Error(`Prediction failed for ${symbol}`);
  return res.json();
}

export async function getModelMetrics(symbol) {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(symbol)}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch model metrics for ${symbol}`);
  return res.json();
}

export async function getStockQuote(symbol) {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}/quote`);
  if (!res.ok) throw new Error(`Failed to load quote for ${symbol}`);
  return res.json();
}

export async function getAvailableModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to list models');
  return res.json();
}

export async function getSystemHealth() {
  const res = await fetch(`${API_BASE}/system/health`);
  if (!res.ok) throw new Error('Failed to fetch system health');
  return res.json();
}

