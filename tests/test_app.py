"""Streamlit interaction checks (requires local loopback sockets)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1]/'app.py')


def test_demo_interactions():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 3
    assert app.number_input(key='plot_left').value == 60
    app.number_input(key='observations').set_value(8).run()
    assert not app.exception
    app.selectbox(key='forecast_model').select('Naive').run()
    assert not app.exception
    app.number_input(key='plot_left').set_value(660).run()
    assert not app.exception
    assert app.warning
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3
    app.selectbox(key='image_source').select('wave').run()
    assert not app.exception
    assert app.number_input(key='observations').value == 60


def test_upload_empty():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.selectbox(key='image_source').select('Upload').run()
    assert not app.exception
    assert len(app.metric) == 0
