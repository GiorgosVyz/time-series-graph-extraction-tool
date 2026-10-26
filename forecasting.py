import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

BASE_MODELS = ('ETS', 'ARIMA', 'Naive')
MODELS = (*BASE_MODELS, 'Auto')
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
    if model == 'Auto':
        raise ValueError('Auto requires an earlier selection period. Compare and select a base model first.')
    if model not in BASE_MODELS:
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


def percentage_errors(errors, actuals):
    """Signed errors relative to |actual|. At zero: exact=0, nonexact=±inf."""
    e,a = np.asarray(errors,dtype=float),np.asarray(actuals,dtype=float)
    if e.ndim!=1 or not len(e) or a.shape!=e.shape or not np.isfinite(e).all() or not np.isfinite(a).all():
        raise ValueError('Need matching finite errors and actual values.')
    result = np.zeros_like(e)
    nonzero = a!=0
    with np.errstate(over='ignore',divide='ignore'):
        result[nonzero] = (e[nonzero]/np.abs(a[nonzero]))*100
    underflow = nonzero & (e!=0) & (result==0)
    result[underflow] = np.copysign(np.nextafter(0.,1.),e[underflow])
    unbounded = ~nonzero & (e!=0)
    result[unbounded] = np.copysign(np.inf,e[unbounded])
    return result


def outside_tolerance(error_percent, tolerance):
    """Tolerance is a percentage; boundaries are inclusive, with float roundoff allowed."""
    values = np.asarray(error_percent,dtype=float)
    if values.ndim!=1 or not len(values) or np.isnan(values).any() or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Invalid percentage errors or tolerance.')
    absolute = np.abs(values)
    return (absolute>tolerance) & ~np.isclose(absolute,tolerance,rtol=1e-12,atol=0)


def summarize(errors, tolerance, actuals):
    e = np.asarray(errors, dtype=float)
    failed = outside_tolerance(percentage_errors(e,actuals),tolerance)
    return {'MAE': np.mean(abs(e)), 'RMSE': np.sqrt(np.mean(e**2)),
            'Historical failure rate': np.mean(failed), 'Count': len(e)}


def compare_models(training_values, horizon):
    """Compare candidates strictly within an earlier prefix, before outer evaluation.

    Use up to five common, recent rolling origins at the requested terminal lead.
    Returning errors separately lets tolerance change without refitting candidates.
    """
    y = _series(training_values)
    _positive_integer(horizon, 'Horizon')
    last_origin = len(y)-horizon
    if last_origin < 4:
        raise ValueError('Auto needs at least four training points plus the horizon inside its selection period.')
    include_arima = last_origin >= ARIMA_MIN_TRAIN
    minimum = ARIMA_MIN_TRAIN if include_arima else 4
    origins = np.arange(max(minimum,last_origin-4),last_origin+1)
    errors, skipped = {}, {}
    for model in BASE_MODELS:
        if model == 'ARIMA' and not include_arima:
            skipped[model] = 'Not enough earlier training points; ARIMA needs at least 12 plus the horizon.'
            continue
        terminal_errors = []
        try:
            for origin in origins:
                forecast = np.asarray(predict(y[:origin],horizon,model),dtype=float)
                if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
                    raise ValueError('Nonfinite or incomplete forecasts.')
                terminal_errors.append(float(y[origin+horizon-1]-forecast[-1]))
            errors[model] = np.asarray(terminal_errors)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            skipped[model] = str(exc)
    if not errors:
        raise ValueError('Auto could not evaluate any candidate. Try Naive or adjust the selection period.')
    return {'errors':errors, 'actuals':y[origins+horizon-1].copy(),
            'skipped':skipped, 'origins':origins, 'training_points':len(y)}


def select_best_model(comparison, tolerance):
    """Rank by failure rate, then MAE, then simplicity for an exact tie."""
    priority = {'Naive':0, 'ETS':1, 'ARIMA':2}
    rows = [{'Model':model, **summarize(errors,tolerance,comparison['actuals'])} for model,errors in comparison['errors'].items()]
    if not rows:
        raise ValueError('No models are available for Auto selection.')
    rows.sort(key=lambda row:(row['Historical failure rate'],row['MAE'],priority[row['Model']]))
    return rows[0]['Model'], pd.DataFrame(rows).set_index('Model')
