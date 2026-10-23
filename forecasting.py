import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt


def predict(values, horizon, model='Holt'):
    y = np.asarray(values, dtype=float)
    if len(y) < 4 or not np.isfinite(y).all() or horizon < 1:
        raise ValueError('Need four finite observations and a positive horizon.')
    if model == 'Naive':
        return np.repeat(y[-1], horizon)
    if model != 'Holt':
        raise ValueError('Unknown model')
    return np.asarray(Holt(y, damped_trend=True).fit().forecast(horizon))


def backtest(values, horizon, initial, model='Holt'):
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
    if not len(e) or not np.isfinite(e).all() or tolerance < 0:
        raise ValueError('Invalid errors or tolerance.')
    return {'MAE': np.mean(abs(e)), 'RMSE': np.sqrt(np.mean(e**2)),
            'Historical failure rate': np.mean(abs(e)>tolerance), 'Count': len(e)}
