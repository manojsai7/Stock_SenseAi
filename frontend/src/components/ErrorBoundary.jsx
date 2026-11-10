import React from 'react';
import { AlertTriangle, RotateCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ErrorBoundary caught error]:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <section className="fintech-card" style={{ padding: '24px', textAlign: 'center', margin: '16px 0' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', color: '#f59e0b', marginBottom: '12px' }}>
            <AlertTriangle size={32} />
          </div>
          <h3 style={{ color: '#f8fafc', fontSize: '1.1rem', margin: '0 0 8px' }}>
            {this.props.title || 'Component Display Error'}
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: '0 0 16px', maxWidth: '480px', marginLeft: 'auto', marginRight: 'auto' }}>
            {this.state.error?.message || 'An unexpected rendering error occurred in this module.'}
          </p>
          <button
            type="button"
            onClick={this.handleReset}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              background: '#0ea5e9',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.82rem',
              fontWeight: 600,
            }}
          >
            <RotateCw size={13} />
            <span>Reload Module</span>
          </button>
        </section>
      );
    }
    return this.props.children;
  }
}
