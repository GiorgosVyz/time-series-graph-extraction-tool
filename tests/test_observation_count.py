"""Only visible, distinct marker evidence may supply an original sample count."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from extraction import suggest_observation_count


COLOR = (25,110,220)
BOUNDS = (30,30,650,270)


def marked_chart(count=20, markers=True, irregular=False):
    image=Image.new('RGB',(680,300),'white')
    draw=ImageDraw.Draw(image)
    x=np.linspace(30,650,count)
    if irregular:
        x[2]+=8
    y=150+70*np.sin(np.arange(count)*.8)
    points=list(zip(x,y))
    draw.line(points,fill=COLOR,width=2)
    if markers:
        for px,py in points:
            draw.ellipse((round(px)-5,round(py)-5,round(px)+5,round(py)+5),fill=COLOR)
    return np.asarray(image)


@pytest.mark.parametrize('count',[5,20,40])
@pytest.mark.parametrize('framed',[False,True])
def test_visible_markers_supply_count(count,framed):
    image=Image.fromarray(marked_chart(count))
    if framed:
        ImageDraw.Draw(image).rectangle(BOUNDS,outline='black')
    assert suggest_observation_count(np.asarray(image),BOUNDS,COLOR)==count


def test_line_vertices_are_not_observations():
    assert suggest_observation_count(marked_chart(markers=False),BOUNDS,COLOR) is None


def test_irregular_or_partial_markers_do_not_guess_count():
    assert suggest_observation_count(marked_chart(irregular=True),BOUNDS,COLOR) is None
    image=marked_chart()
    assert suggest_observation_count(image,(0,30,679,270),COLOR) is None
    assert suggest_observation_count(image,(650,30,30,270),COLOR) is None


@pytest.mark.parametrize('name',['area_left_axis','area_right_axis'])
def test_financial_lines_do_not_supply_original_count(name):
    from extraction import detect_plot_area,suggest_curve_color
    path=Path(__file__).parent/'fixtures'/'market'/f'{name}.png'
    image=np.asarray(Image.open(path).convert('RGB'))
    bounds=detect_plot_area(image)
    assert suggest_observation_count(image,bounds,suggest_curve_color(image,bounds)) is None


@pytest.mark.parametrize('ocr_available',[True,False])
def test_upload_autofills_markers_and_preserves_user_correction(ocr_available):
    from io import BytesIO
    from unittest.mock import patch
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    image=Image.fromarray(marked_chart())
    ImageDraw.Draw(image).rectangle(BOUNDS,outline='black')
    content=BytesIO()
    image.save(content,format='PNG')
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'),default_timeout=30).run()
    recognized={'axes':[0.,19.,0.,100.],'count':None,'notes':[]}
    with patch('streamlit.file_uploader',side_effect=lambda *a,**kw:BytesIO(content.getvalue())), patch(
        'calibration.read_calibration',return_value=recognized,
        side_effect=None if ocr_available else RuntimeError('OCR unavailable')
    ):
        app.run()
        assert not app.exception
        assert app.number_input(key='observations').value==20
        assert not any('Original count unknown' in item.value for item in app.caption)
        if not ocr_available:
            for key,value in zip(['xmin','xmax','ymin','ymax'],recognized['axes']):
                app.number_input(key=key).set_value(value)
            app.run()
        assert not app.exception
        assert len(app.metric)==3
        assert not any('Backtest on sampled data' in item.value for item in app.info)
        assert not any('Adjust the fields above' in item.value for item in app.caption)
        app.number_input(key='observations').set_value(16).run()
        assert app.number_input(key='observations').value==16
        assert not app.exception
        assert len(app.metric)==3
        assert any('Backtest on sampled data' in item.value for item in app.info)
        app.radio(key='input_mode').set_value('Demo').run()
        assert not app.exception
        assert app.number_input(key='observations').value==60
