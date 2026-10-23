import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image
from extraction import extract
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


def test_holt():
    assert np.isfinite(predict(np.arange(20),4)).all()
