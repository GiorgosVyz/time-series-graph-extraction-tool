"""Regression checks for temporal calibration, labels, exports and summaries."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from calibration import axes_from_labels
from time_axis import parse_label, format_time, parse_bounds, series_overview, export_series


def box(x,y):
    return [[x-8,y-8],[x+8,y-8],[x+8,y+8],[x-8,y+8]]


def labelled_axis(texts):
    labels=[(text,.99,box(x,375)) for x,text in zip(np.linspace(80,880,len(texts)),texts)]
    labels += [(str(v),.99,box(40,350-v*6)) for v in [0,10,20,30,40,50]]
    return axes_from_labels(labels,(80,50,880,350))


@pytest.mark.parametrize('labels,kind,start,end',[
    (['Mon','Tue','Wen','Thu','Fri','Sat','Sun'],'weekday',0,6),
    (['Nov','Dec','Jan','Feb','Mar'],'month',10,14),
    (['22:00','23:00','00:00','01:00','02:00'],'clock',1320,1560),
    (['2026-12-29','2026-12-30','2026-12-31','2027-01-01','2027-01-02'],'date',datetime(2026,12,29).toordinal(),datetime(2027,1,2).toordinal()),
])
def test_temporal_axes(labels,kind,start,end):
    result=labelled_axis(labels)
    assert result['time_axis']['kind']==kind
    np.testing.assert_allclose(result['axes'],[start,end,0,50])
    assert result['count']==len(labels)


def test_no_invented_date_or_year():
    assert parse_label('May 12','date') is None
    assert parse_label('03/04','date') is None
    assert parse_label('2026-02-30','date') is None
    assert format_time('weekday',7,0)=='Mon (+1 week)'
    assert format_time('month',12,0)=='Jan (+1 year)'
    assert format_time('clock',1470,1380)=='00:30 (+1 day)'
    assert format_time('clock',740.5,740)=='12:20:30'


@pytest.mark.parametrize('kind,start,end,expected',[
    ('weekday','Fri','Mon',(4,7)),
    ('month','Dec','Mar (+1 year)',(11,14)),
    ('clock','23:00','01:00',(1380,1500)),
    ('date','2026-10-25','2026-10-26',(datetime(2026,10,25).toordinal(),datetime(2026,10,26).toordinal())),
])
def test_editable_bounds(kind,start,end,expected):
    assert parse_bounds(kind,start,end)==expected


def test_invalid_temporal_order_is_not_accepted():
    with pytest.raises(ValueError): parse_bounds('weekday','Mon','Mon')
    with pytest.raises(ValueError): parse_bounds('clock','25:00','26:00')
    with pytest.raises(ValueError): parse_bounds('clock','23:00','01:00 (+1 year)')
    with pytest.raises(ValueError): parse_bounds('date','2026-10-26','2026-10-25')
    result=labelled_axis(['Mon','Tue','Thu','Fri','Sun','Mon','Wed'])
    assert result['time_axis']['kind']=='numeric'
    assert result['axes'][:2]==[None,None]


def test_summary_and_exports_use_recognized_labels():
    axis={'kind':'weekday','start':0,'end':6}
    values=[2,3,9,4,1,3,5]
    text=series_overview(np.arange(7),values,axis)
    assert 'Daily series' in text
    assert 'maximum: 9 at Wed' in text
    assert 'minimum: 1 at Fri' in text
    assert '3 units higher' in text
    assert '2026' not in text
    data=export_series(pd.DataFrame({'time':[7,8],'forecast':[5,6]}),axis)
    assert list(data.time)==['Mon (+1 week)','Tue (+1 week)']
    assert list(data.forecast)==[5,6]


def test_constant_overview_does_not_claim_trend():
    text=series_overview(np.arange(5),np.repeat(4,5),{'kind':'numeric','start':0,'end':4})
    assert 'approximately constant at 4' in text
    assert 'higher' not in text
