const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '') + '/api';

/**
 * Safely extracts error details from an HTTP response.
 */
async function parseError(res, fallbackMessage) {
  try {
    const data = await res.json();
    if (data && data.detail) {
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    }
    if (data && data.message) {
      return data.message;
    }
  } catch (_) {
    // If response body is not JSON, ignore
  }
  return fallbackMessage || `Request failed with status ${res.status}`;
}

export async function searchStocks(query = '') {
  const res = await fetch(`${API_BASE}/stocks/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) {
    const errorMsg = await parseError(res, 'Unable to search equities');
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getStockInfo(symbol) {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}/info`);
  if (!res.ok) {
    const errorMsg = await parseError(res, `Failed to load company profile for ${symbol}`);
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getStockHistory(symbol, period = '1y') {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}/history?period=${period}`);
  if (!res.ok) {
    const errorMsg = await parseError(res, `Failed to load historical data for ${symbol}`);
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getPrediction(symbol, pipelineVersion = 'stationary-return-v1') {
  const queryParam = pipelineVersion ? `?pipeline_version=${encodeURIComponent(pipelineVersion)}` : '';
  const res = await fetch(`${API_BASE}/predictions/${encodeURIComponent(symbol)}/latest${queryParam}`);
  if (!res.ok) {
    const errorMsg = await parseError(res, `Inference not available for ${symbol}`);
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getModelMetrics(symbol) {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(symbol)}/metrics`);
  if (!res.ok) {
    const errorMsg = await parseError(res, `Evaluation metrics not available for ${symbol}`);
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getStockQuote(symbol) {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}/quote`);
  if (!res.ok) {
    const errorMsg = await parseError(res, `Failed to fetch quote for ${symbol}`);
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getAvailableModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    const errorMsg = await parseError(res, 'Failed to fetch available models');
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getSystemHealth() {
  const res = await fetch(`${API_BASE}/system/health`);
  if (!res.ok) {
    const errorMsg = await parseError(res, 'System health check failed');
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getFeatureSensitivity(symbol) {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(symbol)}/explainability`);
  if (!res.ok) {
    const errorMsg = await parseError(res, `Feature sensitivity unavailable for ${symbol}`);
    throw new Error(errorMsg);
  }
  return res.json();
}
