import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image
from extraction import extract, detect_plot_area
from forecasting import predict,backtest,summarize

@pytest.mark.parametrize('name',['trend','wave','shift'])
def test_extraction(name):
    root=Path(__file__).parents[1]/'sample_images'
    truth=pd.read_csv(root/f'{name}.csv')
    result=extract(np.array(Image.open(root/f'{name}.png')),(60,30,660,330),(0,59,0,100),60,(35,110,210))
    assert np.mean(abs(result.value-truth.value))<1
    assert result.time.iloc[-1]==59


def test_rolling_no_leakage():
    y=np.arange(20,dtype=float)
    a=backtest(y,3,8,'Naive')
    changed=y.copy(); changed[8:]+=100
    b=backtest(changed,3,8,'Naive')
    assert np.array_equal(a[a.origin==8].prediction,b[b.origin==8].prediction)
    assert len(a)==30
    assert np.all(a[a.origin==8].prediction==7)


def test_metrics_and_invalid():
    assert summarize([-3,4],3)['Historical failure rate']==0.5
    assert summarize([-3,4],3)['MAE']==3.5
    with pytest.raises(ValueError): predict([1,np.nan,2,3],2)
    with pytest.raises(ValueError): backtest(np.arange(10),5,8)


def test_ets():
    assert np.isfinite(predict(np.arange(20),4)).all()


@pytest.mark.parametrize('name', ['trend', 'wave', 'shift'])
def test_detect_plot_area(name):
    root = Path(__file__).parents[1]/'sample_images'
    image = np.array(Image.open(root/f'{name}.png'))
    assert detect_plot_area(image) == (60, 30, 660, 330)


def test_detect_rejects_blank_and_unframed():
    image = np.full((400, 700, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_plot_area(image)
    image[30:331, 60] = 0
    image[330, 60:661] = 0
    with pytest.raises(ValueError):
        detect_plot_area(image)

@pytest.mark.parametrize('n', [4, 8, 20])
def test_ets_short_and_constant(n):
    assert np.isfinite(predict(np.arange(n, dtype=float), 3)).all()
    assert np.array_equal(predict(np.repeat(7., n), 3), [7., 7., 7.])


@pytest.mark.parametrize('horizon', [0, -1, 1.5, True])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        predict(np.arange(10), horizon)
    with pytest.raises(ValueError):
        backtest(np.arange(10), horizon, 4)


def test_backtest_list_and_no_ets_leakage():
    y = list(np.arange(15, dtype=float))
    before = backtest(y, 2, 6)
    changed = y[:6] + [v+100 for v in y[6:]]
    after = backtest(changed, 2, 6)
    np.testing.assert_allclose(before[before.origin == 6].prediction,
                               after[after.origin == 6].prediction)
    assert len(before) == 16


@pytest.mark.parametrize('values', [[1,2,3], [[1,2],[3,4]], [1,2,np.inf,4]])
def test_invalid_series(values):
    with pytest.raises(ValueError):
        predict(values, 2)


def test_nonfinite_tolerance():
    with pytest.raises(ValueError):
        summarize([1,2], np.nan)


@pytest.mark.parametrize('name', ['trend','wave','shift'])
def test_auto_suggestions_end_to_end(name):
    from extraction import suggest_curve_color
    root = Path(__file__).parents[1]/'sample_images'
    image = np.array(Image.open(root/f'{name}.png'))
    bounds = detect_plot_area(image)
    color = suggest_curve_color(image, bounds)
    assert color == (35,110,210)
    recovered = extract(image, bounds, (0,59,0,100), 60, color)
    results = backtest(recovered.value.to_numpy(), 3, 30)
    assert len(results) == 84
    assert np.isfinite(results.prediction).all()


def test_invalid_extraction_and_no_color():
    from extraction import suggest_curve_color
    image = np.full((100,200,3),255,dtype=np.uint8)
    with pytest.raises(ValueError):
        suggest_curve_color(image,(0,0,199,99))
    with pytest.raises(ValueError):
        extract(image,(100,0,0,99),(0,10,0,100),20,(35,110,210))
    with pytest.raises(ValueError):
        extract(image,(0,0,199,99),(0,10,0,100),20.5,(35,110,210))
