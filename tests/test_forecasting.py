import numpy as np
import pytest
import forecasting
from forecasting import predict, backtest, ARIMA_ORDERS


@pytest.mark.parametrize('model',['ETS','ARIMA','Naive'])
@pytest.mark.parametrize('kind',['constant','negative','wave'])
def test_models_return_finite_forecasts(model,kind):
    x=np.arange(20,dtype=float)
    y={'constant':np.repeat(3.,20),'negative':-20+.2*x+np.sin(x/3),'wave':10+np.sin(x/2)}[kind]
    result=predict(y,4,model)
    assert result.shape == (4,)
    assert np.isfinite(result).all()


def test_arima_short_training_and_unknown_model():
    with pytest.raises(ValueError,match='12 training'):
        predict(np.arange(11.),3,'ARIMA')
    with pytest.raises(ValueError,match='Unknown model'):
        predict(np.ones(20),3,'Unknown')


@pytest.mark.parametrize('model',['ETS','ARIMA'])
def test_model_backtest_has_no_future_leakage(model):
    y=5+np.sin(np.arange(16)/3)
    changed=y.copy()
    changed[12:]+=100
    original=backtest(y,2,12,model)
    other=backtest(changed,2,12,model)
    np.testing.assert_allclose(original[original.origin==12].prediction,
                               other[other.origin==12].prediction)
    assert len(original)==6


def test_arima_selection_uses_only_training_holdout(monkeypatch):
    seen=[]
    y=np.arange(20,dtype=float)
    class FakeFit:
        def __init__(self, values): self.last=values[-1]
        def forecast(self,horizon): return np.repeat(self.last,horizon)
    def fit(values,order):
        seen.append(np.array(values))
        return FakeFit(values)
    monkeypatch.setattr(forecasting,'_fit_arima',fit)
    predict(y,3,'ARIMA')
    assert len(seen)==len(ARIMA_ORDERS)+1
    for inner in seen[:-1]:
        np.testing.assert_array_equal(inner,y[:-4])
    np.testing.assert_array_equal(seen[-1],y)


def test_arima_failed_candidates_are_reported(monkeypatch):
    def fail(*args): raise ValueError('Fit failed')
    monkeypatch.setattr(forecasting,'_fit_arima',fail)
    with pytest.raises(ValueError,match='could not fit'):
        predict(np.arange(20.),3,'ARIMA')
