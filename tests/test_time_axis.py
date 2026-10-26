"""Regression checks for temporal calibration, labels, exports and summaries."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from calibration import axes_from_labels
from time_axis import (parse_label, format_time, parse_bounds, series_overview, export_series,
                       display_time, forecast_labels, apply_time_axis, daily_sampling_points)


def box(x,y):
    return [[x-8,y-8],[x+8,y-8],[x+8,y+8],[x-8,y+8]]


@pytest.mark.parametrize('start,end,expected',[
    ('2026-04-27','2026-10-24',181),
    ('2026-04-27','2026-10-23',180),
    ('2026-05-01','2026-05-31',31),
    ('2024-02-01','2024-03-01',30),
    ('2025-02-01','2025-03-01',29),
    ('2026-12-29','2027-01-03',6),
])
def test_daily_sample_count_follows_each_date_range(start,end,expected):
    first,last=parse_bounds('date',start,end)
    count=daily_sampling_points('date',first,last)
    assert count==expected
    # The extracted grid actually has a one-day interval and includes both bounds.
    grid=np.linspace(first,last,count)
    np.testing.assert_allclose(np.diff(grid),1,rtol=0,atol=1e-9)
    axis={'kind':'date','start':first,'end':last,'sampled':True,'extraction_points':count,'sampling_interval':'calendar_day'}
    exported=export_series(pd.DataFrame({'time':grid,'value':np.zeros(count)}),axis)
    assert set(exported.sampling_interval)=={'calendar_day'}
    assert exported.time.iloc[0]==start and exported.time.iloc[-1]==end


@pytest.mark.parametrize('kind,start,end',[
    ('numeric',0,20),('month',3,9),('date',None,739000),
    ('date',739000,739000),('date',739001,739000),
    ('date',739000,739001.5),('date',739000,float('inf')),
])
def test_daily_count_requires_a_valid_whole_day_date_span(kind,start,end):
    assert daily_sampling_points(kind,start,end) is None


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
    axis={'kind':'weekday','start':0,'end':6,'frequency':'Daily'}
    values=[2,3,9,4,1,3,5]
    text=series_overview(np.arange(7),values,axis)
    assert 'daily chart' in text
    assert '**Estimated high:** 9, around Wed' in text
    assert '**Estimated low:** 1, around Fri' in text
    assert 'Up 3 units' in text
    assert '2026' not in text
    data=export_series(pd.DataFrame({'time':[7,8],'forecast':[5,6]}),axis)
    assert list(data.time)==['Daily observation +1','Daily observation +2']
    assert list(data.forecast)==[5,6]


def test_constant_overview_does_not_claim_trend():
    text=series_overview(np.arange(5),np.repeat(4,5),{'kind':'numeric','start':0,'end':4})
    assert 'approximately constant at **4**' in text
    assert 'higher' not in text


@pytest.mark.parametrize('frequency',['Unknown','Daily','Intraday'])
def test_month_ticks_never_invent_intraday_precision(frequency):
    start=datetime(2026,4,27).toordinal()+.75
    end=datetime(2026,10,24).toordinal()+.06
    axis={'kind':'date','start':start,'end':end,'estimated_period':True,'frequency':frequency}
    times=np.linspace(start,end,100)
    values=np.linspace(111.3,54.6,100)
    text=series_overview(times,values,axis)
    assert '27 Apr 2026 – 24 Oct 2026 (approximate dates)' in text
    assert 'Down 56.7 units' in text
    assert '17:51' not in text and '18:00' not in text and '01:26' not in text
    assert '(first occurrences)' not in text
    recovered=export_series(pd.DataFrame({'time':times,'value':values}),axis)
    assert all(len(label)==10 for label in recovered.time)
    assert len(recovered)==100  # Presentation must not aggregate or remove observations.
    np.testing.assert_array_equal(recovered.value,values)


def test_intraday_precision_requires_source_times_and_confirmation():
    value=datetime(2026,10,24).toordinal()+.5
    axis={'kind':'date','start':value,'end':value+1,'has_time_labels':True,'frequency':'Intraday'}
    assert display_time(axis,value)=='2026-10-24 12:00'
    axis['frequency']='Daily'
    assert display_time(axis,value)=='2026-10-24'
    axis['frequency']='Unknown'
    assert display_time(axis,value)=='2026-10-24'


@pytest.mark.parametrize('frequency',['Unknown','Daily','Intraday'])
def test_temporal_forecast_chart_and_csv_use_observation_steps(frequency):
    import plotly.graph_objects as go
    start=datetime(2026,10,23).toordinal()
    axis={'kind':'date','start':start,'end':start+1,'frequency':frequency}
    figure=go.Figure(go.Scatter(x=np.arange(start,start+5),y=np.arange(5)))
    apply_time_axis(figure,axis,np.arange(start,start+5),history_count=2)
    expected=forecast_labels(axis,3)
    assert list(figure.layout.xaxis.ticktext)[-1]==expected[-1]
    assert not any('2026-10-25' in label for label in figure.layout.xaxis.ticktext)
    exported=export_series(pd.DataFrame({'time':[start+2,start+3,start+4],'forecast':[2,3,4]}),axis)
    assert list(exported.time)==expected
    assert list(exported.forecast)==[2,3,4]


def test_summary_only_mentions_first_occurrence_for_ties():
    axis={'kind':'numeric','start':0,'end':4}
    text=series_overview(np.arange(5),[1,4,4,2,1],axis)
    assert '**Estimated high:** 4, at 1' in text
    assert '**Estimated low:** 1, at 0' in text
    assert 'first occurrence' in text
    assert 'Starts and ends at approximately the same value' in text


@pytest.mark.parametrize('kind',['numeric','date','weekday'])
def test_sampled_series_keeps_its_basis_in_overview_forecasts_and_exports(kind):
    start=datetime(2026,10,1).toordinal() if kind=='date' else 0
    times=np.linspace(start,start+20,100)
    values=np.linspace(2,5,100)
    axis={'kind':kind,'start':start,'end':start+20,'frequency':'Daily','sampled':True,'extraction_points':100}
    text=series_overview(times,values,axis)
    assert '**100 sampled points' in text
    assert '100 observations' not in text
    assert forecast_labels(axis,2)==['Sample +1','Sample +2']
    recovered=export_series(pd.DataFrame({'time':times,'value':values}),axis)
    assert set(recovered.sampling_basis)=={'sampled_curve'}
    assert set(recovered.extraction_points)=={100}
    np.testing.assert_array_equal(recovered.value,values)
    future=export_series(pd.DataFrame({'time':[start+21,start+22],'forecast':[5,6]}),axis)
    assert set(future.sampling_basis)=={'sampled_curve'}
    assert set(future.extraction_points)=={100}
    assert list(future.forecast)==[5,6]
    if kind!='numeric':
        assert list(future.time)==['Sample +1','Sample +2']
