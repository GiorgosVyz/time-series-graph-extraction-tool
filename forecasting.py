import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

MODELS = ('ETS', 'ARIMA', 'Naive')
ARIMA_ORDERS = ((1, 0, 0), (1, 1, 0), (0, 1, 1))
ARIMA_MIN_TRAIN = 12


def _fit_arima(y, order):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters')
        fitted = ARIMA(y, order=order, trend='c' if order[1] == 0 else 'n').fit(
            method_kwargs={'maxiter': 100})
    if not fitted.mle_retvals.get('converged', False):
        raise ValueError('ARIMA fit did not converge.')
    return fitted


def _arima_forecast(y, horizon):
    # Only data within this training prefix participate in order selection.
    holdout = min(5, max(2, len(y)//5))
    candidates = []
    for order in ARIMA_ORDERS:
        try:
            fit = _fit_arima(y[:-holdout], order)
            predicted = np.asarray(fit.forecast(holdout))
            score = float(np.mean(np.abs(y[-holdout:]-predicted)))
            if np.isfinite(score):
                candidates.append((score, order))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            continue
    for _, order in sorted(candidates):
        try:
            result = np.asarray(_fit_arima(y, order).forecast(horizon))
            if np.isfinite(result).all():
                return result
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            continue
    raise ValueError('ARIMA could not fit a reliable candidate. Try ETS or Naive.')


def _series(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 4 or not np.isfinite(y).all():
        raise ValueError('Need a one-dimensional series of at least four finite observations.')
    return y


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def predict(values, horizon, model='ETS'):
    y = _series(values)
    _positive_integer(horizon, 'Horizon')
    if model not in MODELS:
        raise ValueError('Unknown model')
    if model == 'ARIMA' and len(y) < ARIMA_MIN_TRAIN:
        raise ValueError('ARIMA needs at least 12 training observations. Use ETS or Naive for shorter series.')
    if model == 'Naive' or np.ptp(y) == 0:
        return np.repeat(y[-1], horizon)
    if model == 'ARIMA':
        return _arima_forecast(y, horizon)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            fitted = ETSModel(y, error='add', trend='add', damped_trend=True,
                              seasonal=None, initialization_method='estimated').fit(disp=False, maxiter=300)
        if not fitted.mle_retvals.get('converged', False):
            raise ValueError('ETS did not converge. Try the Naive model.')
        result = np.asarray(fitted.forecast(horizon))
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise ValueError('ETS could not fit this series. Try the Naive model.') from exc
    if not np.isfinite(result).all():
        raise ValueError('Model produced nonfinite forecasts. Try the Naive model.')
    return result


def backtest(values, horizon, initial, model='ETS'):
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
