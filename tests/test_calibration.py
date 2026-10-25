from pathlib import Path
import numpy as np
import pytest
from calibration import numeric_label, axes_from_labels, fit_axis


def box(x,y):
    return [[x-5,y-5],[x+5,y-5],[x+5,y+5],[x-5,y+5]]


def test_numeric_formats():
    assert numeric_label('7,5') == 7.5
    assert numeric_label('−2.5') == -2.5
    assert numeric_label('1e3') == 1000
    assert numeric_label('1,234.5') == 1234.5
    assert numeric_label('7,800.00') == 7800
    assert numeric_label('7.800,00') == 7800
    for text in ['Time', '1 2 3', 'nan', '12,34.5']:
        assert numeric_label(text) is None


def test_numeric_axes_and_sample_count():
    labels = [(str(i), .99, box(50+(i-1)*30,365)) for i in range(1,21)]
    labels += [(str(v).replace('.',','), .99, box(25,350-v*10)) for v in [0,7.5,15,22.5,30]]
    result = axes_from_labels(labels,(50,50,620,350))
    assert result['axes'] == [1,20,0,30]
    assert result['count'] == 20


def test_sparse_ticks_do_not_imply_sample_count():
    labels = [(str(i),.99,box(50+i*10,365)) for i in range(0,51,10)]
    result = axes_from_labels(labels,(50,50,550,350))
    assert result['axes'][:2] == [0,50]
    assert result['count'] is None
    assert result['axes'][2:] == [None,None]


def test_invalid_labels_do_not_make_up_axes():
    assert axes_from_labels([], (50,50,620,350))['axes'] == [None]*4
    with pytest.raises(ValueError):
        fit_axis([(0,1),(50,10),(100,100),(150,1000)],0,150)
    with pytest.raises(ValueError):
        fit_axis([(0,1),(10,2)],0,100)


def test_tick_outlier_is_rejected():
    ticks = [(0,0),(25,5),(50,500),(75,15),(100,20)]
    limits, values = fit_axis(ticks,0,100)
    assert limits == [0,20]
    assert len(values) == 4
