"""Streamlit interaction checks (requires local loopback sockets)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1]/'app.py')


def test_demo_interactions():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key='input_mode').set_value('Demo').run()
    assert not app.exception
    assert len(app.metric) == 3
    assert app.number_input(key='plot_left').value == 60
    assert app.selectbox(key='forecast_model').value == 'ETS'
    app.number_input(key='observations').set_value(20).run()
    app.selectbox(key='forecast_model').select('ARIMA').run()
    assert not app.exception
    assert len(app.metric) == 3
    app.number_input(key='observations').set_value(8).run()
    assert not app.exception
    assert len(app.metric) == 0
    app.selectbox(key='forecast_model').select('ETS').run()
    assert not app.exception
    assert len(app.metric) == 3
    app.selectbox(key='forecast_model').select('Naive').run()
    assert not app.exception
    app.number_input(key='plot_left').set_value(660).run()
    assert not app.exception
    assert app.warning
    app.number_input(key='plot_left').set_value(60).run()
    assert not app.exception
    assert len(app.metric) == 3
    app.selectbox(key='image_source').select('wave').run()
    assert not app.exception
    assert app.number_input(key='observations').value == 60
    app.selectbox(key='image_source').select('shift').run()
    assert not app.exception
    assert len(app.metric) == 3


def test_upload_empty():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert app.radio(key='input_mode').value == 'Upload'
    assert not app.exception
    assert len(app.metric) == 0
    app.radio(key='input_mode').set_value('Demo').run()
    assert not app.exception
    assert len(app.metric) == 3


def test_error_chart_preserves_failure_counts():
    import numpy as np
    from app import error_distribution
    for values, tolerance, expected in [
        ([-6, -5, 0, 5, 6], 5, [1, 3, 1]),
        ([-1, 0, 1], 0, [1, 1, 1]),
        ([0, 0, 0], 5, [0, 3, 0]),
    ]:
        chart = error_distribution(values, tolerance)
        assert [int(np.sum(trace.y)) for trace in chart.data] == expected


def test_upload_confirmation_and_changed_settings():
    from io import BytesIO
    from unittest.mock import patch
    image = (Path(APP).parent/'sample_images'/'trend.png').read_bytes()
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key='input_mode').set_value('Upload')
    with patch('streamlit.file_uploader', side_effect=lambda *a, **kw: BytesIO(image)), patch(
        'calibration.read_calibration', return_value={'axes':[0.,59.,0.,100.], 'count':60, 'notes':[]}
    ):
        app.run()
        assert not app.exception
        assert len(app.metric) == 0
        assert len(app.checkbox) == 0
        assert app.number_input(key='xmax').value == 59.
        assert app.number_input(key='observations').value == 60
        app.button(key='extract_series').click().run()
        assert not app.exception
        assert len(app.metric) == 3


def test_unreadable_axes_remain_editable():
    from io import BytesIO
    from unittest.mock import patch
    import streamlit as st
    st.cache_data.clear()
    image = (Path(APP).parent/'sample_images'/'trend.png').read_bytes()
    app = AppTest.from_file(APP, default_timeout=30).run()
    with patch('streamlit.file_uploader', side_effect=lambda *a, **kw: BytesIO(image)), patch(
        'calibration.read_calibration', return_value={'axes':[None]*4, 'count':None, 'notes':['Unreadable']}
    ):
        app.run()
        assert not app.exception
        assert app.number_input(key='xmax').value is None
        for key, value in zip(['xmin','xmax','ymin','ymax','observations'], [0.,59.,0.,100.,60]):
            app.number_input(key=key).set_value(value)
        app.run()
        app.button(key='extract_series').click().run()
        assert not app.exception
        assert len(app.metric) == 3
        app.number_input(key='observations').set_value(20).run()
        assert not app.exception
        assert len(app.metric) == 0
        app.button(key='extract_series').click().run()
        assert not app.exception
        assert len(app.metric) == 3


def test_temporal_upload_overview_and_forecast():
    from io import BytesIO
    from unittest.mock import patch
    import streamlit as st
    st.cache_data.clear()
    image = (Path(APP).parent/'tests'/'fixtures'/'time_axes'/'hours.png').read_bytes()
    recognized = {'axes':[1200.,1740.,0.,50.], 'count':10, 'notes':[],
                  'time_axis':{'kind':'clock','start':1200.,'end':1740.,'count':10}}
    app = AppTest.from_file(APP, default_timeout=40).run()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**kw: BytesIO(image)), patch(
        'calibration.read_calibration',return_value=recognized
    ):
        app.run()
        assert not app.exception
        assert app.selectbox(key='time_kind').value == 'Clock time'
        assert app.text_input(key='time_start').value == '20:00'
        assert app.text_input(key='time_end').value == '05:00 (+1 day)'
        app.button(key='extract_series').click().run()
        assert not app.exception
        assert len(app.metric)==3
        assert any('Hourly series with 10 observations' in text.value for text in app.markdown)
        assert any('05:00 (+1 day)' in text.value for text in app.markdown)
        app.text_input(key='time_end').set_value('not a time').run()
        assert not app.exception
        assert len(app.metric)==0
        assert app.warning
