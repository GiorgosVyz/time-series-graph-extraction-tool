import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt


def _series(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 4 or not np.isfinite(y).all():
        raise ValueError('Need a one-dimensional series of at least four finite observations.')
    return y


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def predict(values, horizon, model='Holt'):
    y = _series(values)
    _positive_integer(horizon, 'Horizon')
    if model == 'Naive':
        return np.repeat(y[-1], horizon)
    if model != 'Holt':
        raise ValueError('Unknown model')
    if np.ptp(y) == 0:
        return np.repeat(y[-1], horizon)
    result = np.asarray(Holt(y, damped_trend=True, initialization_method='estimated').fit().forecast(horizon))
    if not np.isfinite(result).all():
        raise ValueError('Model produced nonfinite forecasts. Try the Naive model.')
    return result


def backtest(values, horizon, initial, model='Holt'):
    values = _series(values)
    _positive_integer(horizon, 'Horizon')
    _positive_integer(initial, 'Initial training window')
    if initial < 4 or initial+horizon > len(values):
        raise ValueError('No complete validation fold available.')
    rows = []
    for origin in range(initial, len(values)-horizon+1):
        for h, p in enumerate(predict(values[:origin], horizon, model), 1):
            actual = values[origin+h-1]
            rows.append((origin, h, actual, p, actual-p))
    return pd.DataFrame(rows, columns=['origin','horizon','actual','prediction','error'])


def summarize(errors, tolerance):
    e = np.asarray(errors, dtype=float)
    if e.ndim != 1 or not len(e) or not np.isfinite(e).all() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid errors or tolerance.')
    return {'MAE': np.mean(abs(e)), 'RMSE': np.sqrt(np.mean(e**2)),
            'Historical failure rate': np.mean(abs(e)>tolerance), 'Count': len(e)}
