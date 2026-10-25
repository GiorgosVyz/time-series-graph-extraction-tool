"""Open/filled financial chart regressions. Screenshots have no source CSV ground truth."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image,ImageDraw
from extraction import detect_plot_area,suggest_curve_color,extract
from calibration import axes_from_labels,numeric_label
from time_axis import parse_label,format_time,parse_bounds

ROOT=Path(__file__).parent/'fixtures'/'market'


def box(x,y):
    return [[x-8,y-4],[x+8,y-4],[x+8,y+4],[x-8,y+4]]


@pytest.mark.parametrize('text,value',[('7,800.00',7800),('7.800,00',7800),('7,800',7800),('7,5',7.5),('-1,234.50',-1234.5)])
def test_price_formats(text,value):
    assert numeric_label(text)==value


def test_right_axis_and_competing_scales():
    bounds=(50,30,650,230)
    right=[(f'{v:,.2f}',.99,box(700,230-(v-7100)/3.5)) for v in range(7100,7801,100)]
    result=axes_from_labels(right,bounds)
    assert result['value_axis_side']=='right'
    np.testing.assert_allclose(result['axes'][2:],[7100,7800])
    left=[(str(v),.99,box(20,230-v*2)) for v in range(0,101,20)]
    assert axes_from_labels(right+left,bounds)['axes'][2:]==[None,None]


def test_month_year_labels_and_partial_month_bounds():
    labels=[(text,.99,box(x,325)) for x,text in zip([170,275,380,485,590],['Jun 2026','Jul 2026','Aug 2026','Sep 2026','Oct 2026'])]
    result=axes_from_labels(labels,(55,149,675,313))
    assert result['time_axis']['kind']=='date'
    assert result['time_axis']['estimated_period']
    assert result['count'] is None
    axis=result['time_axis']
    parsed=parse_bounds('date',format_time('date',axis['start']),format_time('date',axis['end']))
    np.testing.assert_allclose(parsed,[axis['start'],axis['end']],rtol=0,atol=1/1440)
    assert parse_label('May 12','date') is None
    start,end=parse_bounds('month','Apr +0.8983 month','Oct +0.7770 month')
    np.testing.assert_allclose([start,end],[3.8983,9.777])


@pytest.mark.parametrize('name,side,expected_last',[
    ('area_left_axis','left',54.24),('area_right_axis','right',7623.68)
])
def test_supplied_screenshots_trace_price_curve(name,side,expected_last):
    image=np.array(Image.open(ROOT/f'{name}.png').convert('RGB'))
    bounds=detect_plot_area(image)
    left,top,right,bottom=bounds
    if side=='left':
        assert 50<=left<=58 and 670<=right<=680 and 145<=top<=152 and 310<=bottom<=316
        labels=[(str(v),.99,box(35,313-(v-40)*1.64)) for v in range(40,141,20)]
    else:
        assert left<=15 and 590<=right<=605 and 20<=top<=32 and 225<=bottom<=235
        labels=[(f'{v:,.2f}',.99,box(644,230-(v-7100)/3.5)) for v in range(7100,7801,100)]
    axes=axes_from_labels(labels,bounds)['axes']
    axes[:2]=[0,119]
    data=extract(image,bounds,axes,120,suggest_curve_color(image,bounds))
    assert data.attrs['curve_coverage']>.9
    assert abs(data.value.iloc[-1]-expected_last)<(2 if side=='left' else 25)
    # Check the detected points stay near saturated stroke pixels, above the pale fill.
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hits=0
    for x,y in zip(data.pixel_x,data.pixel_y):
        x,y=int(round(x)),int(round(y))
        hits+=bool(np.any(hsv[max(0,y-2):y+3,max(0,x-2):x+3,1]>110))
    assert hits/len(data)>.9


def test_filled_curve_accuracy_with_headers_and_volume_bars():
    image=Image.new('RGB',(760,360),'white')
    draw=ImageDraw.Draw(image)
    draw.rectangle((40,5,500,30),fill=(250,60,45))
    x=np.linspace(70,690,60)
    truth=90+45*np.sin(np.arange(60)/8)+np.arange(60)*.5
    points=list(zip(x,270-truth))
    draw.polygon([(70,270)]+points+[(690,270)],fill=(255,225,225))
    for y in range(70,271,40):
        draw.line((70,y,690,y),fill=(220,205,205))
    for i in range(70,690,8):
        draw.rectangle((i,276+(i%19),i+3,302),fill='#cccccc')
    draw.line(points,fill=(250,60,45),width=3)
    image=np.array(image)
    bounds=detect_plot_area(image)
    assert abs(bounds[1]-70)<=2 and abs(bounds[3]-270)<=2
    data=extract(image,(70,70,690,270),(0,59,0,200),60,suggest_curve_color(image,bounds))
    assert np.mean(abs(data.value-truth))<1.5
