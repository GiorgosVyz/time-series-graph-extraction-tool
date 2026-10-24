"""Local, guided graph digitization and forecast evaluation."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from extraction import extract, detect_plot_area, suggest_curve_color
from forecasting import predict, backtest, summarize

ROOT = Path(__file__).parent
BOUND_KEYS = ['plot_left', 'plot_top', 'plot_right', 'plot_bottom']


@st.cache_data(show_spinner=False, max_entries=24)
def evaluate(values, horizon, initial, model):
    return backtest(values, horizon, initial, model)


def metric_table(results, tolerance):
    return pd.DataFrame([
        {'Lead time': h, **summarize(group.error, tolerance)}
        for h, group in results.groupby('horizon')
    ]).set_index('Lead time')


def main():
    st.set_page_config(page_title='Time Series Digitizer & Forecaster', page_icon='📈', layout='wide')
    st.title('Time Series Digitizer & Forecaster')
    st.caption('From chart pixels to a forecast you can evaluate. Runs locally, without external AI services.')
    with st.sidebar:
        st.header('Start here')
        source = st.selectbox('Image source', ['trend', 'wave', 'shift', 'Upload'], key='image_source',
                              format_func=lambda x: {'trend':'Demo · trend', 'wave':'Demo · wave', 'shift':'Demo · level shift', 'Upload':'Upload your chart'}[x])
        upload = st.file_uploader('PNG or JPG', type=['png', 'jpg', 'jpeg']) if source == 'Upload' else None
        st.info('Best results: one continuous colored line, linear axes, and evenly spaced observations.')
        st.caption('1. Check calibration\n\n2. Inspect recovered points\n\n3. Evaluate the forecast')
    path = upload if source == 'Upload' else ROOT/'sample_images'/f'{source}.png'
    if path is None:
        st.info('Upload a chart to begin, or choose a built-in demo in the sidebar.')
        return
    try:
        with Image.open(path) as original:
            if original.width * original.height > 12_000_000:
                st.error('Please use an image with at most 12 million pixels.')
                return
            im = np.array(ImageOps.exif_transpose(original).convert('RGB'))
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        st.error('This image could not be opened. Please upload a valid PNG or JPG.')
        return
    height, width = im.shape[:2]
    if min(height, width) < 20:
        st.error('The image is too small to digitize. Use a larger chart.')
        return
    image_id = hashlib.sha256(im.tobytes() + str(im.shape).encode() + source.encode()).hexdigest()
    if st.session_state.get('calibration_image') != image_id:
        st.session_state.calibration_image = image_id
        defaults = (0, 0, width-1, height-1)
        try:
            defaults = detect_plot_area(im)
            message = 'Plot area suggested automatically. Check the green border.'
        except ValueError:
            message = 'Plot borders were not detected. Adjust the rectangle below.'
        for key, value in zip(BOUND_KEYS, defaults):
            st.session_state[key] = int(value)
        try:
            rgb = suggest_curve_color(im, defaults)
            color = '#%02x%02x%02x' % rgb
        except ValueError:
            color = '#236ed2'
        for key, value in zip(['xmin', 'xmax', 'ymin', 'ymax'], [0., 59., 0., 100.]):
            st.session_state[key] = value
        st.session_state.curve_color = color
        st.session_state.observations = 60
        st.session_state.confirmed = source != 'Upload'
        st.session_state.detection_message = message

    st.subheader('1. Check calibration')
    st.caption('Green = plot boundary. Numerical axis values cannot be inferred from the border alone.')
    if st.button('Auto-detect plot area'):
        try:
            for key, value in zip(BOUND_KEYS, detect_plot_area(im)):
                st.session_state[key] = int(value)
            st.session_state.detection_message = 'Plot borders detected. Check the green rectangle.'
        except ValueError as exc:
            st.session_state.detection_message = str(exc)
    st.info(st.session_state.detection_message)
    with st.expander('Adjust plot rectangle (pixels)', expanded=source == 'Upload'):
        st.caption(f'Image: {width} × {height} pixels. Origin is the top-left corner; exclude labels and legends.')
        cols = st.columns(4)
        bounds = tuple(cols[i].number_input(label, 0, int(limit-1), key=key)
                       for i, (label, key, limit) in enumerate(zip(['Left','Top','Right','Bottom'], BOUND_KEYS, [width,height,width,height])))
    cols = st.columns(4)
    axes = tuple(cols[i].number_input(label, key=key) for i, (label,key) in enumerate(zip(
        ['Time minimum', 'Time maximum', 'Value minimum', 'Value maximum'], ['xmin','xmax','ymin','ymax'])))
    cols = st.columns([2, 1])
    count = cols[0].number_input('Number of original observations', 8, 300, key='observations',
                                help='Use the original number of equally spaced samples, not the image width. Extra samples create interpolated observations.')
    color = cols[1].color_picker('Curve color', key='curve_color')
    with st.expander('Advanced extraction settings'):
        distance = st.slider('Color tolerance', 10, 200, 90)
    rgb = tuple(int(color[i:i+2], 16) for i in (1,3,5))
    preview = Image.fromarray(im)
    draw = ImageDraw.Draw(preview)
    if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
        draw.rectangle(bounds, outline='#00a050', width=2)
    try:
        df = extract(im, bounds, axes, count, rgb, distance)
        for x, y in zip(df.pixel_x, df.pixel_y):
            draw.ellipse((x-2,y-2,x+2,y+2), fill='#e53935')
        extraction_error = None
    except ValueError as exc:
        df, extraction_error = None, str(exc)
    st.image(preview, caption='Green: plot boundary · Red: recovered observations', width='stretch')
    confirmed = st.checkbox('Axis limits and observation count match the original chart', key='confirmed')
    if extraction_error:
        st.warning(extraction_error)
        return
    if not confirmed:
        st.info('Confirm the calibration to continue. Uploaded images start with example axis values; replace them with your chart’s values.')
        return

    st.subheader('2. Inspect the recovered series')
    history = go.Figure(go.Scatter(x=df.time, y=df.value, name='Recovered series', mode='lines+markers', marker_size=4))
    history.update_layout(xaxis_title='Time', yaxis_title='Value', height=300, margin=dict(t=20,b=20))
    st.plotly_chart(history, width='stretch')
    if source != 'Upload':
        truth = pd.read_csv(ROOT/'sample_images'/f'{source}.csv')
        if count == len(truth) and np.allclose(axes, [0,59,0,100]):
            st.caption(f'Sample extraction MAE vs. known data: {np.mean(abs(df.value-truth.value)):.3f} value units.')
    st.download_button('Download recovered data', df[['time','value']].to_csv(index=False), 'extracted.csv', 'text/csv')

    st.subheader('3. Forecast and evaluate')
    cols = st.columns(3)
    horizon = cols[0].number_input('Forecast horizon', 1, min(20,count//3), min(5,count//3))
    initial = cols[1].number_input('Initial training observations', 4, count-horizon, max(4,count//2))
    model = cols[2].selectbox('Forecast model', ['Holt','Naive'], key='forecast_model')
    tolerance = st.number_input('Failure tolerance (value units)', min_value=0., value=5., help='A prediction fails when its absolute error is greater than this value.')
    y = df.value.to_numpy()
    try:
        with st.spinner('Evaluating historical forecasts…'):
            results = evaluate(y, horizon, initial, model)
            baseline = evaluate(y, horizon, initial, 'Naive')
            future = predict(y, horizon, model)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        st.error(f'Forecast could not be calculated: {exc}')
        return
    terminal = results.loc[results.horizon == horizon, 'error']
    selected = summarize(terminal, tolerance)
    naive = summarize(baseline.loc[baseline.horizon == horizon, 'error'], tolerance)
    cols = st.columns(3)
    cols[0].metric(f'MAE · {horizon}-step ahead', f'{selected["MAE"]:.3f}')
    cols[1].metric(f'RMSE · {horizon}-step ahead', f'{selected["RMSE"]:.3f}')
    cols[2].metric('Historical failure rate', f'{selected["Historical failure rate"]:.1%}')
    st.caption(f'Based on {len(terminal)} rolling origins at lead time {horizon}. Failure = absolute error > {tolerance:g}.')
    if len(terminal) < 20:
        st.warning('Fewer than 20 validation origins: the failure rate is based on limited evidence.')
    if model == 'Holt' and selected['MAE'] >= naive['MAE']:
        st.info('The naive baseline matches or beats Holt at this lead time. A more complex model does not always forecast better.')
    times = df.time.iloc[-1] + np.arange(1,horizon+1)*(axes[1]-axes[0])/(count-1)
    forecast = pd.DataFrame({'time':times, 'forecast':future})
    fig = go.Figure(go.Scatter(x=df.time,y=y,name='Recovered history'))
    fig.add_trace(go.Scatter(x=np.r_[df.time.iloc[-1],times], y=np.r_[y[-1],future], name=model+' forecast', line=dict(dash='dash'), mode='lines+markers'))
    fig.add_vline(x=float(df.time.iloc[-1]), line_dash='dot', line_color='gray')
    fig.update_layout(xaxis_title='Time', yaxis_title='Value', height=380, margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    comparison = pd.DataFrame([{'Model':model, **selected}, {'Model':'Naive baseline', **naive}]) if model != 'Naive' else pd.DataFrame([{'Model':'Naive', **selected}])
    st.dataframe(comparison.set_index('Model').style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', 'Historical failure rate':'{:.1%}'}), width='stretch')
    with st.expander('Errors by lead time and validation details'):
        st.caption('Expanding-window backtesting: each fit uses only observations preceding its forecast origin. All models use the same origins.')
        st.dataframe(metric_table(results,tolerance).style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', 'Historical failure rate':'{:.1%}'}), width='stretch')
        errors = go.Figure(go.Histogram(x=terminal, name='Historical errors'))
        errors.add_vline(x=-tolerance, line_dash='dash', line_color='red')
        errors.add_vline(x=tolerance, line_dash='dash', line_color='red')
        errors.update_layout(xaxis_title='Actual minus predicted (selected lead time)', yaxis_title='Count', height=270)
        st.plotly_chart(errors, width='stretch')
    cols = st.columns(2)
    cols[0].download_button('Download forecast', forecast.to_csv(index=False), 'forecast.csv', 'text/csv')
    cols[1].download_button('Download backtest', results.to_csv(index=False), 'backtest.csv', 'text/csv')
    st.info('Historical failure rates estimate past performance; they are not exact future probabilities. Overlapping validation folds are dependent. Changing conditions and extraction errors can reduce reliability. No prediction interval is claimed.')


if __name__ == '__main__':
    main()
