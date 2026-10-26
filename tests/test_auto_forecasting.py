"""Automatic selection is scored before, and separately from, outer evaluation."""
import numpy as np
import pytest
import forecasting
from forecasting import compare_models, select_best_model


@pytest.mark.parametrize('tolerance,expected',[(5,'ARIMA'),(3,'ETS'),(6,'ETS')])
def test_auto_ranks_failure_rate_before_mae(tolerance,expected):
    comparison={'errors':{'ETS':np.array([0,0,0,6]),'ARIMA':np.repeat(4,4),'Naive':np.repeat(5,4)},'actuals':np.repeat(100.,4)}
    winner,scores=select_best_model(comparison,tolerance)
    assert winner==expected
    assert scores.index[0]==winner
    assert list(scores.Count)==[4,4,4]


def test_auto_prefers_simpler_model_on_exact_tie():
    comparison=compare_models(np.repeat(7.,30),3)
    winner,scores=select_best_model(comparison,0.)
    assert winner=='Naive'
    assert set(scores.index)=={'Naive','ETS','ARIMA'}
    assert (scores['Historical failure rate']==0).all()


def test_candidates_use_identical_origins_inside_selection_prefix(monkeypatch):
    seen=[]
    prefix=np.arange(30,dtype=float)
    def forecast(values,horizon,model):
        seen.append((model,np.array(values)))
        return np.repeat(values[-1],horizon)
    monkeypatch.setattr(forecasting,'predict',forecast)
    comparison=compare_models(prefix,3)
    assert list(comparison['origins'])==[23,24,25,26,27]
    assert comparison['training_points']==30
    np.testing.assert_array_equal(comparison['actuals'],prefix[comparison['origins']+2])
    assert comparison['skipped']=={}
    for model in forecasting.BASE_MODELS:
        calls=[values for candidate,values in seen if candidate==model]
        assert [len(values) for values in calls]==[23,24,25,26,27]
        for values in calls:
            np.testing.assert_array_equal(values,prefix[:len(values)])
        np.testing.assert_array_equal(comparison['errors'][model],np.repeat(3.,5))
    assert max(comparison['origins'])+3<=len(prefix)


def test_short_selection_excludes_arima_without_blocking_other_models(monkeypatch):
    monkeypatch.setattr(forecasting,'predict',lambda y,h,model:np.repeat(y[-1],h))
    comparison=compare_models(np.arange(8.),2)
    assert set(comparison['errors'])=={'Naive','ETS'}
    assert 'ARIMA' in comparison['skipped']
    assert list(comparison['origins'])==[4,5,6]


def test_failed_and_nonfinite_candidates_are_excluded(monkeypatch):
    def forecast(values,horizon,model):
        if model=='ETS':
            raise ValueError('Fit failed')
        return np.repeat(np.nan if model=='ARIMA' else values[-1],horizon)
    monkeypatch.setattr(forecasting,'predict',forecast)
    comparison=compare_models(np.arange(24.),3)
    assert set(comparison['errors'])=={'Naive'}
    assert set(comparison['skipped'])=={'ETS','ARIMA'}
    assert select_best_model(comparison,5)[0]=='Naive'


def test_all_failed_candidates_are_reported(monkeypatch):
    def fail(*args):
        raise ValueError('Unavailable')
    monkeypatch.setattr(forecasting,'predict',fail)
    with pytest.raises(ValueError,match='could not evaluate any candidate'):
        compare_models(np.arange(24.),3)


@pytest.mark.parametrize('values,horizon',[(np.arange(5.),2),(np.arange(20.),0),(np.arange(20.),1.5)])
def test_invalid_selection_windows_are_rejected(values,horizon):
    with pytest.raises(ValueError):
        compare_models(values,horizon)


def test_auto_upload_metadata_and_tolerance_reuse_selection_fits():
    from pathlib import Path
    from io import StringIO
    from unittest.mock import patch
    import pandas as pd
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from streamlit.delta_generator import DeltaGenerator
    st.cache_data.clear()
    root=Path(__file__).parents[1]
    prefixes,downloads=[],{}
    original_compare=compare_models
    original_download=DeltaGenerator.download_button
    def compare(values,horizon):
        prefixes.append(np.array(values))
        return original_compare(values,horizon)
    def capture(self,label,data,*args,**kwargs):
        downloads[label]=data
        return original_download(self,label,data,*args,**kwargs)
    app=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
    app.radio(key='input_mode').set_value('Demo').run()
    with patch('forecasting.compare_models',side_effect=compare),patch.object(DeltaGenerator,'download_button',capture):
        app.selectbox(key='forecast_model').select('Auto').run()
        assert not app.exception
        assert len(app.metric)==3
        assert not app.button
        banner=next(item.value for item in app.markdown if item.value.startswith('**Auto selected '))
        winner=banner.removeprefix('**Auto selected ').removesuffix('**')
        assert winner in forecasting.BASE_MODELS
        assert app.metric[2].label=='Backtest failure rate'
        selection_table=app.dataframe[0].value
        assert 'Selection failure rate' in selection_table.columns
        assert 'Historical failure rate' not in selection_table.columns
        assert (selection_table['Selection tests']==5).all()
        assert (selection_table.Failures<=selection_table['Selection tests']).all()
        assert any(item.value.startswith('Later backtest:') for item in app.caption)
        initial=next(item.value for item in app.number_input if item.label=='Initial training points')
        assert len(prefixes)==1 and len(prefixes[0])==initial
        assert 'Auto selected' in downloads['Download forecast']
        for label in ['Download forecast data (CSV)','Download backtest']:
            data=pd.read_csv(StringIO(downloads[label]))
            assert set(data.forecast_method)=={'Auto'}
            assert set(data.selected_model)=={winner}
            assert set(data.selection_points)=={initial}
        tested=pd.read_csv(StringIO(downloads['Download backtest']))
        assert tested.origin.min()==initial
        np.testing.assert_allclose(tested.error_percent,100*(tested.actual-tested.prediction)/tested.actual.abs())
        assert set(tested.tolerance_percent)=={10.}
        np.testing.assert_array_equal(tested.failed,forecasting.outside_tolerance(tested.error_percent,10.))
        # Ranking responds to tolerance without fitting the earlier candidates again.
        tolerance=next(item for item in app.number_input if item.label=='Failure tolerance (%)')
        tolerance.set_value(1.).run()
        assert not app.exception
        assert len(app.metric)==3
        assert len(prefixes)==1


def test_auto_short_series_and_manual_models_remain_available():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'),default_timeout=60).run()
    app.radio(key='input_mode').set_value('Demo').run()
    app.number_input(key='observations').set_value(8).run()
    app.selectbox(key='forecast_model').select('Auto').run()
    assert not app.exception
    assert len(app.metric)==3
    assert any('ARIMA unavailable for selection' in item.value for item in app.caption)
    app.number_input(key='observations').set_value(5).run()
    assert not app.exception
    assert not app.metric
    assert any('separate model selection from evaluation' in item.value for item in app.info)
    app.selectbox(key='forecast_model').select('Naive').run()
    assert not app.exception
    assert len(app.metric)==3


def test_month_chart_distinguishes_selection_zero_from_later_failure_rate():
    from pathlib import Path
    from io import BytesIO
    from unittest.mock import patch
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    root=Path(__file__).parents[1]
    image=(root/'tests'/'fixtures'/'market'/'month_day_month.png').read_bytes()
    # Only the axis reader is mocked; the curve values and all model fits are real.
    calibration={'axes':[0.,99.,50.,100.],'count':None,'notes':[]}
    app=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**kw:BytesIO(image)),patch(
        'calibration.read_calibration',return_value=calibration
    ):
        app.run()
        next(item for item in app.number_input if item.label=='Failure tolerance (%)').set_value(5.).run()
        app.selectbox(key='forecast_model').select('Auto').run()
        assert not app.exception
        assert any(item.value=='**Auto selected Naive**' for item in app.markdown)
        scores=app.dataframe[0].value
        assert scores.loc['Naive','Selection failure rate']==0
        assert scores.loc['Naive','Failures']==0
        assert scores.loc['Naive','Selection tests']==5
        assert scores.loc['Naive','MAE']==pytest.approx(.828406,abs=1e-5)
        assert app.metric[2].label=='Backtest failure rate'
        assert app.metric[2].value=='45.7%'
        assert any('Later backtest: 21 failures in 46 forecasts' in item.value for item in app.caption)
