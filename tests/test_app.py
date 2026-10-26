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
    assert next(item.value for item in app.number_input if item.label=='Failure tolerance (%)') == 10.
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
        chart = error_distribution(values,tolerance,np.repeat(100.,len(values)))
        assert [int(np.sum(trace.y)) for trace in chart.data] == expected


def test_upload_updates_automatically_with_calibration_changes():
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
        assert len(app.metric) == 3
        assert not app.button
        assert len(app.checkbox) == 0
        assert app.number_input(key='xmax').value == 59.
        assert app.number_input(key='observations').value == 60
        app.number_input(key='observations').set_value(20).run()
        assert not app.exception
        assert len(app.metric) == 3
        assert any('**20 sampled points' in item.value for item in app.markdown)
        # Invalid settings hide results; correcting them restores results automatically.
        app.number_input(key='ymax').set_value(0.).run()
        assert not app.exception
        assert not app.metric
        assert not app.get('plotly_chart')
        assert app.warning
        app.number_input(key='ymax').set_value(100.).run()
        assert not app.exception
        assert len(app.metric)==3
        app.number_input(key='xmax').set_value(None).run()
        assert not app.exception
        assert not app.metric
        assert not app.button
        app.number_input(key='xmax').set_value(59.).run()
        assert not app.exception
        assert len(app.metric)==3


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
        assert not app.metric
        assert not app.button
        for key, value in zip(['xmin','xmax','ymin','ymax','observations'], [0.,59.,0.,100.,60]):
            app.number_input(key=key).set_value(value)
        app.run()
        assert not app.exception
        assert len(app.metric) == 3
        app.number_input(key='observations').set_value(20).run()
        assert not app.exception
        assert len(app.metric) == 3
        assert any('**20 sampled points' in item.value for item in app.markdown)


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
        assert len(app.metric)==3
        assert app.selectbox(key='observation_frequency').value == 'Intraday'
        assert any('10 observations · intraday chart' in text.value for text in app.markdown)
        assert any('05:00 (+1 day)' in text.value for text in app.markdown)
        app.text_input(key='time_end').set_value('not a time').run()
        assert not app.exception
        assert len(app.metric)==0
        assert app.warning
        app.text_input(key='time_end').set_value('05:00 (+1 day)').run()
        assert not app.exception
        assert len(app.metric)==3


def test_open_area_chart_upload_defaults_to_labelled_sampling():
    from io import BytesIO
    from unittest.mock import patch
    from datetime import datetime
    import streamlit as st
    st.cache_data.clear()
    image=(Path(APP).parent/'tests'/'fixtures'/'market'/'area_left_axis_full.png').read_bytes()
    start=float(datetime(2026,4,27).toordinal())+.75
    end=float(datetime(2026,10,23).toordinal())+.5
    recognized={'axes':[start,end,40.,140.],'count':None,'notes':[],
                'time_axis':{'kind':'date','start':start,'end':end,'count':None,'estimated_period':True}}
    app=AppTest.from_file(APP,default_timeout=40).run()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**kw:BytesIO(image)), patch(
        'calibration.read_calibration',return_value=recognized
    ):
        app.run()
        assert not app.exception
        assert app.selectbox(key='time_kind').value=='Dates'
        assert app.text_input(key='time_start').value=='2026-04-27'
        assert app.selectbox(key='observation_frequency').value=='Daily'
        assert app.number_input(key='observations').label=='Extraction points'
        assert app.number_input(key='observations').value==180
        assert not any('calendar-day samples' in item.value for item in app.caption)
        assert len(app.metric)==3
        assert any('Backtest on sampled data' in item.value for item in app.info)
        assert any('**180 sampled points' in item.value for item in app.markdown)
        app.text_input(key='time_end').set_value('2026-10-24').run()
        assert not app.exception
        assert app.number_input(key='observations').value==181
        app.text_input(key='time_start').set_value('2026-05-01').run()
        assert not app.exception
        assert app.number_input(key='observations').value==177
        app.number_input(key='observations').set_value(40).run()
        assert not app.exception
        assert len(app.metric)==3
        assert any('approximate dates' in item.value for item in app.markdown)
        assert app.selectbox(key='observation_frequency').value=='Unknown'
        app.text_input(key='time_start').set_value('2026-04-27').run()
        assert not app.exception
        assert app.number_input(key='observations').value==40  # A manual choice is preserved.
        app.selectbox(key='observation_frequency').select('Daily').run()
        assert not app.exception
        assert len(app.metric)==3
        assert app.number_input(key='observations').value==181  # Daily resumes automatic calculation.
        overview=next(item.value for item in app.markdown if '**181 sampled points' in item.value)
        assert 'daily sampling' in overview
        assert '27 Apr 2026 – 24 Oct 2026' in overview
        assert '18:00' not in overview
        import json
        charts=[json.loads(chart.proto.spec) for chart in app.get('plotly_chart')]
        forecast=charts[1]
        assert forecast['data'][1]['customdata'][1]=='Sample +1'
        assert 'Sample +5' in forecast['layout']['xaxis']['ticktext']
        assert forecast['layout']['title']['text']=='Forecast from 181 sampled points'
        assert all(':' not in label for label in forecast['data'][0]['customdata'])
        app.text_input(key='time_end').set_value('2027-04-27').run()
        assert not app.exception
        assert app.number_input(key='observations').value==366
        assert not app.metric
        assert any('supports up to 300' in item.value for item in app.warning)
        app.text_input(key='time_end').set_value('2026-04-29').run()
        assert not app.exception
        assert app.number_input(key='observations').value==3
        assert not app.metric  # Three daily points are insufficient to forecast.
        app.text_input(key='time_end').set_value('2026-05-03').run()
        assert not app.exception
        assert app.number_input(key='observations').value==7
        assert len(app.metric)==3
        # A different image must not inherit the user's daily sampling confirmation.
        app.radio(key='input_mode').set_value('Demo').run()
        assert not app.exception
        assert app.session_state.observation_frequency=='Unknown'
        assert app.number_input(key='observations').value==60
        assert not any('Backtest on sampled data' in item.value for item in app.info)


def test_failed_ocr_still_provides_default_sampling_count():
    from io import BytesIO
    from unittest.mock import patch
    import streamlit as st
    st.cache_data.clear()
    image=(Path(APP).parent/'tests'/'fixtures'/'market'/'area_left_axis_full.png').read_bytes()
    app=AppTest.from_file(APP,default_timeout=30).run()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**kw:BytesIO(image)), patch(
        'calibration.read_calibration',side_effect=RuntimeError('OCR unavailable')
    ):
        app.run()
        assert not app.exception
        assert app.number_input(key='observations').value==100
        assert app.number_input(key='ymin').value is None
        assert any('Original count unknown' in item.value for item in app.caption)
        assert not app.metric
        assert not app.button
