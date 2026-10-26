"""Visible label formats and partial/faint area charts; no market-calendar inference."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from PIL import Image,ImageDraw
from calibration import axes_from_labels,read_calibration
from extraction import detect_plot_area,suggest_curve_color,extract
from time_axis import (parse_label,format_time,display_time,parse_bounds,calibrated_times,
                       calendar_sample_positions,daily_sampling_points,export_series,series_overview)

ROOT=Path(__file__).parent/'fixtures'/'market'


def box(x,y,width=20):
    return np.array([[x-width,y-6],[x+width,y-6],[x+width,y+6],[x-width,y+6]],dtype=float)


@pytest.mark.parametrize('text,minutes',[
    ('9:30 AM',570),('12:00 AM',0),('12:00 PM',720),('1:00 PM',780),
    ('1:00PM',780),('01:00 p.m.',780),('11:59:30 PM',1439.5),('23:00',1380),
])
def test_clock_labels_keep_meridiem(text,minutes):
    assert parse_label(text,'clock')==minutes


@pytest.mark.parametrize('text',['13:00 PM','0:30 AM','12:75 PM','24:00'])
def test_invalid_clock_labels_are_rejected(text):
    assert parse_label(text,'clock') is None


@pytest.mark.parametrize('text,expected',[
    ('Oct 21','Oct 21'),('21 Oct','Oct 21'),('October21','Oct 21'),('Feb 29','Feb 29'),
])
def test_yearless_labels_never_acquire_a_calendar_year(text,expected):
    value=parse_label(text,'month_day')
    assert value is not None
    assert format_time('month_day',value)==expected
    assert parse_label(text,'date') is None
    assert parse_label('Apr 31','month_day') is None


def test_clock_display_does_not_invent_seconds():
    axis={'kind':'clock','start':569.110974,'end':866.985323}
    assert display_time(axis,axis['start'])=='09:29'
    axis['has_second_labels']=True
    assert display_time(axis,740.5)=='12:20:30'
    assert parse_bounds('clock','11:00 PM','1:00 AM')==(1380,1500)


def test_ocr_keeps_whole_clock_labels_instead_of_pm_word_fragments(monkeypatch):
    full=[(text,.99,box(x,375,31)*2) for x,text in zip(np.linspace(80,880,5),
                                                   ['10:00 AM','11:00 AM','12:00 PM','1:00 PM','2:00 PM'])]
    words=[[(text.split()[0],.99,box(x-10,375,18)*2),
            (text.split()[1],.99,box(x+23,375,7)*2)]
           for x,(text,_,_) in zip(np.linspace(80,880,5),full)]
    ylabels=[(str(v),.99,box(40,350-v*6,8)*2) for v in range(0,51,10)]
    lines=full+ylabels
    output=SimpleNamespace(word_results=words+[ylabels],txts=[t for t,_,_ in lines],
                           scores=[s for _,s,_ in lines],boxes=np.array([b for _,_,b in lines]))
    monkeypatch.setattr('calibration._engine',lambda:lambda *a,**kw:output)
    result=read_calibration(np.zeros((410,930,3),dtype=np.uint8),(80,50,880,350))
    assert result['time_axis']['kind']=='clock'
    np.testing.assert_allclose(result['axes'],[600,840,0,50])
    assert result['time_axis']['labels']==['10:00 AM','11:00 AM','12:00 PM','1:00 PM','2:00 PM']


def test_nonuniform_yearless_labels_all_remain_anchors():
    bounds=(50,50,650,350)
    pixels=[100,250,400,550]
    texts=['Oct 21','Oct 22','Oct 23','Oct 26']
    result=axes_from_labels([(t,.99,box(x,375)) for t,x in zip(texts,pixels)],bounds)
    axis=dict(result['time_axis'],sampled=True,extraction_points=4)
    assert axis['kind']=='month_day'
    assert result['count'] is None
    assert len(axis['anchors'])==4
    times=calibrated_times(pixels,axis,bounds)
    np.testing.assert_allclose(times,[parse_label(t,'month_day') for t in texts],rtol=0,atol=1e-9)
    np.testing.assert_allclose(np.diff(times),[1,1,3],rtol=0,atol=1e-9)
    exported=export_series(pd.DataFrame({'time':times,'value':[1,2,4,3]}),axis)
    assert list(exported.time)==texts
    text=series_overview(times,[1,2,4,3],axis)
    assert '2000' not in text and '2026' not in text
    assert 'approximate period' in text
    assert daily_sampling_points('month_day',times[0],times[-1]) is None


def test_daily_sampling_on_nonuniform_date_labels_uses_uneven_pixels():
    bounds=(50,50,650,350)
    ticks=['2026-10-21','2026-10-22','2026-10-23','2026-10-26']
    result=axes_from_labels([(t,.99,box(x,375,45)) for t,x in zip(ticks,[100,250,400,550])],bounds)
    axis=result['time_axis']
    assert axis['anchors'] is not None
    axis.update(start=np.floor(axis['start']),end=np.ceil(axis['end']),sampling_interval='calendar_day')
    count=daily_sampling_points('date',axis['start'],axis['end'])
    positions=calendar_sample_positions(axis,bounds,count)
    assert not np.allclose(np.diff(positions),np.diff(positions)[0])
    image=Image.new('RGB',(700,400),'white')
    ImageDraw.Draw(image).line((50,300,650,100),fill=(20,110,220),width=3)
    df=extract(np.asarray(image),bounds,(axis['start'],axis['end'],0,100),count,(20,110,220),pixel_positions=positions)
    times=calibrated_times(df.pixel_x,axis,bounds)
    np.testing.assert_allclose(np.diff(times),1,rtol=0,atol=1e-8)


CASES=[
    ('intraday_partial','clock',['10:00 AM','12:00 PM','1:00 PM','2:00 PM'],[83,259,347.25,434],50,58,56.17),
    ('month_day_month','month_day',['Oct 1','Oct 6','Oct 9','Oct 14','Oct 19','Oct 22'],[124,217.75,310.75,404,500.75,590.25],50,100,56.02),
    ('month_day_five_days','month_day',['Oct 21','Oct 22','Oct 23','Oct 26'],[175.25,316,457,602],50,65,56.02),
]


def fixture_calibration(name):
    _,kind,texts,pixels,ymin,ymax,last=next(case for case in CASES if case[0]==name)
    image=np.asarray(Image.open(ROOT/f'{name}.png').convert('RGB'))
    bounds=detect_plot_area(image)
    left,top,right,bottom=bounds
    labels=[(text,.99,box(x,bottom+13)) for text,x in zip(texts,pixels)]
    labels += [(str(value),.99,box(17,bottom-(value-ymin)*(bottom-top)/(ymax-ymin),8))
               for value in np.linspace(ymin,ymax,5)]
    return image,bounds,axes_from_labels(labels,bounds),last


@pytest.mark.parametrize('name',[case[0] for case in CASES])
def test_supplied_chart_formats_extract_available_curve(name):
    image,bounds,result,last=fixture_calibration(name)
    assert result['time_axis']['kind']==next(case[1] for case in CASES if case[0]==name)
    assert all(value is not None for value in result['axes'])
    assert 210<=bounds[1]<=220 and 375<=bounds[3]<=384
    if name=='intraday_partial':
        assert 465<=bounds[2]<=478  # Do not fabricate history in the empty afternoon area.
    data=extract(image,bounds,result['axes'],100,suggest_curve_color(image,bounds))
    assert abs(data.value.iloc[-1]-last)<.3


@pytest.mark.parametrize('name',[case[0] for case in CASES])
def test_supplied_chart_uploads_show_results_automatically(name):
    from io import BytesIO
    from unittest.mock import patch
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    _,_,recognized,_=fixture_calibration(name)
    content=(ROOT/f'{name}.png').read_bytes()
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'),default_timeout=60).run()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**kw:BytesIO(content)),patch(
        'calibration.read_calibration',return_value=recognized
    ):
        app.run()
        assert not app.exception
        assert len(app.metric)==3
        assert not app.button
        assert app.number_input(key='observations').value==100
        if name=='intraday_partial':
            assert app.selectbox(key='observation_frequency').value=='Intraday'
        else:
            assert app.selectbox(key='time_kind').value=='Month and day'
            overview=next(item.value for item in app.markdown if '**100 sampled points' in item.value)
            assert '2000' not in overview and '2026' not in overview
