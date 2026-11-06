"""Same-date forecasting metrics; no trading profitability claims."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

def baseline_predictions(train_x, train_y, test_x, alpha=1.0):
    ridge = Ridge(alpha=alpha).fit(train_x.reshape(len(train_x), -1), train_y)
    return {'naive': np.zeros(len(test_x)), 'ridge': ridge.predict(test_x.reshape(len(test_x), -1))}, ridge

def metrics(actual, predicted, previous_close):
    a, p, c = map(lambda v: np.asarray(v, dtype=float).reshape(-1), (actual, predicted, previous_close))
    if not len(a) or not (len(a) == len(p) == len(c)) or not all(np.isfinite(v).all() for v in (a,p,c)):
        raise ValueError('Metrics require aligned finite nonempty arrays')
    if (c <= 0).any() or (a <= -1).any() or (p <= -1).any():
        raise ValueError('Nonpositive reconstructed price')
    actual_close, predicted_close = c*(1+a), c*(1+p)
    error = predicted_close-actual_close
    # Three classes: exact zero returns are FLAT. No test-tuned threshold.
    truth, guess = np.sign(a), np.sign(p)
    precision, recall, _, support = precision_recall_fscore_support(truth, guess, labels=[-1,0,1], zero_division=0)
    return dict(n=len(a), price_mae=float(np.abs(error).mean()), price_rmse=float(np.sqrt(np.mean(error**2))),
                mape_pct=float(np.mean(np.abs(error/actual_close))*100), return_mae_bps=float(np.abs(a-p).mean()*10000),
                directional_accuracy=float(np.mean(truth==guess)), balanced_accuracy=float(balanced_accuracy_score(truth, guess)),
                mcc=float(matthews_corrcoef(truth, guess)), class_order=['DOWN','FLAT','UP'], precision=precision.tolist(),
                recall=recall.tolist(), support=support.tolist(), confusion_matrix=confusion_matrix(truth,guess,labels=[-1,0,1]).tolist(),
                actual_return_std=float(a.std()), predicted_return_std=float(p.std()),
                volatility_ratio=float(p.std()/a.std()) if a.std() else None)
