"""Percentage failure scoring stays consistent across values, models and plots."""
import numpy as np
import pytest
from forecasting import percentage_errors,outside_tolerance,summarize,select_best_model
from app import error_distribution


def test_five_percent_of_sixty_is_three_units():
    actual=np.repeat(60.,4)
    errors=actual-np.array([57.,63.,56.99,63.01])
    np.testing.assert_array_equal(outside_tolerance(percentage_errors(errors,actual),5),[False,False,True,True])
    score=summarize(errors,5,actual)
    assert score['Historical failure rate']==.5
    assert score['MAE']==pytest.approx(np.mean(np.abs(errors)))
    assert score['RMSE']==pytest.approx(np.sqrt(np.mean(errors**2)))


def test_each_prediction_uses_its_own_actual_value():
    np.testing.assert_allclose(percentage_errors([2,2],[20,200]),[10,1])
    assert summarize([2,2],5,[20,200])['Historical failure rate']==.5


@pytest.mark.parametrize('scale',[.01,1.,1000.])
def test_failure_rate_is_invariant_to_unit_scaling(scale):
    errors=np.array([3.,3.1,-20.,.5])
    actual=np.array([60.,60.,200.,20.])
    base=summarize(errors,5,actual)
    scaled=summarize(errors*scale,5,actual*scale)
    assert scaled['Historical failure rate']==base['Historical failure rate']
    assert scaled['MAE']==pytest.approx(base['MAE']*scale)
    assert scaled['RMSE']==pytest.approx(base['RMSE']*scale)


def test_negative_actuals_and_zero_actuals_have_explicit_behavior():
    percentages=percentage_errors([-3,5,0,.01],[-60,-100,0,0])
    np.testing.assert_allclose(percentages[:3],[-5,5,0])
    assert percentages[3]==np.inf
    np.testing.assert_array_equal(outside_tolerance(percentages,5),[False,False,False,True])
    assert summarize([-3,5,0,.01],5,[-60,-100,0,0])['Historical failure rate']==.25


def test_zero_tolerance_only_accepts_exact_matches():
    percentages=percentage_errors([0,1e-300,-1,0],[0,1e300,0,4])
    np.testing.assert_array_equal(outside_tolerance(percentages,0),[False,True,True,False])
    assert not np.isnan(percentages).any()


def test_roundoff_at_exact_tolerance_is_within_but_a_real_excess_fails():
    error=1.26-1.2
    assert not outside_tolerance(percentage_errors([error],[1.2]),5)[0]
    assert outside_tolerance([5.00000001],5)[0]


@pytest.mark.parametrize('errors,actuals',[
    ([1,2],[100]),([1],[np.nan]),([np.inf],[100]),([] ,[]),([[1]],[100]),
])
def test_invalid_actuals_or_errors_are_rejected(errors,actuals):
    with pytest.raises(ValueError):
        summarize(errors,5,actuals)


@pytest.mark.parametrize('tolerance',[-1,np.inf,np.nan])
def test_invalid_percentage_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        summarize([1],tolerance,[100])


def test_auto_uses_relative_errors_instead_of_absolute_unit_errors():
    comparison={'errors':{'ETS':np.array([2.,0.]),'ARIMA':np.array([0.,4.]),'Naive':np.array([0.,10.])},
                'actuals':np.array([10.,1000.])}
    winner,scores=select_best_model(comparison,5)
    assert winner=='ARIMA'
    assert scores.loc['ETS','Historical failure rate']==.5
    assert scores.loc['ARIMA','Historical failure rate']==0
    assert scores.loc['ETS','MAE']<scores.loc['ARIMA','MAE']


def test_histogram_and_metrics_share_percentage_classification():
    errors=[3,2,-20]
    actual=[60,20,200]
    fig=error_distribution(errors,5,actual)
    assert [int(np.sum(trace.y)) for trace in fig.data]==[1,1,1]
    assert fig.layout.xaxis.title.text=='Forecast error (%)'
    within=fig.data[1]
    assert np.all(np.abs(np.asarray(within.x)[np.asarray(within.y)>0])<=5)
    assert summarize(errors,5,actual)['Historical failure rate']==pytest.approx(2/3)


@pytest.mark.parametrize('errors,expected_count',[([0,1,-1],1),([1,-1],0)])
def test_unbounded_errors_count_as_failures_without_breaking_histogram(errors,expected_count):
    actual=np.zeros(len(errors))
    fig=error_distribution(errors,5,actual)
    assert sum(int(np.sum(trace.y)) for trace in fig.data)==expected_count
    assert 'unbounded percentage errors' in fig.layout.annotations[0].text
    rate=summarize(errors,5,actual)['Historical failure rate']
    assert rate==pytest.approx((len(errors)-expected_count)/len(errors))
