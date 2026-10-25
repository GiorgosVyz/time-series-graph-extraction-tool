"""Local, guided graph digitization and forecast evaluation."""
from pathlib import Path
import hashlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from extraction import extract, detect_plot_area, suggest_curve_color
from forecasting import predict, backtest, summarize, MODELS, ARIMA_MIN_TRAIN
from calibration import read_calibration
from time_axis import (KINDS, KEYS, REVERSE_KEYS, format_time, parse_bounds,
                       apply_time_axis, export_series, series_overview)

ROOT = Path(__file__).parent
BOUND_KEYS = ['plot_left', 'plot_top', 'plot_right', 'plot_bottom']
AXIS_KEYS = ['xmin', 'xmax', 'ymin', 'ymax']


@st.cache_data(show_spinner=False, max_entries=24)
def detect_axis_values(image, bounds):
    return read_calibration(image, bounds)


def fill_axis_fields(image, bounds):
    """Optional OCR failure must not prevent manual extraction."""
    try:
        with st.spinner('Reading chart axes…'):
            result = detect_axis_values(image, tuple(bounds))
        for key, value in zip(AXIS_KEYS, result['axes']):
            st.session_state[key] = value
        st.session_state.observations = result['count']
        axis = result.get('time_axis', {'kind':'numeric', 'start':result['axes'][0], 'end':result['axes'][1]})
        st.session_state.detected_time_axis = axis
        st.session_state.time_kind = KEYS[axis['kind']]
        st.session_state.axis_editor_kind = KEYS[axis['kind']]
        if axis['kind'] != 'numeric':
            st.session_state.time_start = format_time(axis['kind'],axis['start'],axis['start'])
            st.session_state.time_end = format_time(axis['kind'],axis['end'],axis['start'])
        if any(value is None for value in result['axes']):
            return 'Some axis labels could not be read reliably. Fill in the missing values below.'
        if result['count'] is None:
            return 'Axis values detected. Enter the original observation count below.'
        return 'Axis values detected. Observation count suggested from visible tick labels; check it matches your chart.'
    except Exception:
        logging.getLogger(__name__).warning('Automatic axis reading failed', exc_info=True)
        return 'Automatic axis reading is unavailable. You can enter the values below.'


@st.cache_data(show_spinner=False, max_entries=24)
def evaluate(values, horizon, initial, model):
    return backtest(values, horizon, initial, model)


def metric_table(results, tolerance):
    return pd.DataFrame([
        {'Lead time': h, **summarize(group.error, tolerance)}
        for h, group in results.groupby('horizon')
    ]).set_index('Lead time')


def error_distribution(errors, tolerance):
    """Use common bins so colored counts add up to the observed errors exactly."""
    values = np.asarray(errors, dtype=float)
    edges = np.histogram_bin_edges(values, bins=min(16, max(6, int(np.sqrt(len(values))))))
    # Put tolerance boundaries on bin edges instead of coloring mixed bins by center.
    cuts = [v for v in (-tolerance, 0, tolerance) if edges[0] < v < edges[-1]]
    edges = np.unique(np.r_[edges, cuts])
    centers, widths = (edges[:-1]+edges[1:])/2, np.diff(edges)
    fig = go.Figure()
    groups = [
        ('Overestimate', values < -tolerance, '#6366F1'),
        ('Within tolerance', np.abs(values) <= tolerance, '#0D9488'),
        ('Underestimate', values > tolerance, '#EA580C'),
    ]
    for label, mask, color in groups:
        counts, _ = np.histogram(values[mask], bins=edges)
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=widths*.94, name=label,
            marker=dict(color=color, line=dict(color='white', width=1)),
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate=(label+'<br>Error range: %{customdata[0]:.2f} to %{customdata[1]:.2f}'
                           '<br>Past forecasts: %{y}<extra></extra>'),
        ))
    fig.add_vrect(x0=-tolerance, x1=tolerance, fillcolor='#0D9488', opacity=.07, line_width=0, layer='below')
    for boundary in sorted(set([-tolerance, tolerance])):
        fig.add_vline(x=boundary, line_dash='dash', line_color='#64748B', line_width=1.5)
    fig.add_vline(x=0, line_color='#334155', line_width=1)
    low, high = min(float(values.min()), 0), max(float(values.max()), 0)
    padding = max((high-low)*.08, .5)
    fig.update_layout(
        barmode='stack', template='plotly_white', height=360,
        margin=dict(t=15, b=65, l=20, r=20),
        legend=dict(orientation='h', yanchor='top', y=-.27, x=0),
        xaxis=dict(title='Forecast error (actual − predicted)', zeroline=False,
                   range=[min(low-padding, float(edges[0])), max(high+padding, float(edges[-1]))]),
        yaxis=dict(title='Number of past forecasts', dtick=1 if len(values)<15 else None, rangemode='tozero'),
        hoverlabel=dict(bgcolor='white', font_size=13),
    )
    return fig


def main():
    st.set_page_config(page_title='Time Series Digitizer & Forecaster', page_icon='📈', layout='wide')
    st.title('Time Series Digitizer & Forecaster')
    mode = st.radio('Chart source', ['Upload', 'Demo'], index=0, horizontal=True, key='input_mode')
    if mode == 'Demo':
        source = st.selectbox('Demo chart', ['trend', 'wave', 'shift'], key='image_source',
                              format_func=lambda x: {'trend':'Trend', 'wave':'Wave', 'shift':'Level shift'}[x])
        upload = None
    else:
        source = 'Upload'
        upload = st.file_uploader('Upload your chart', type=['png', 'jpg', 'jpeg'],
                                  help='Use a PNG or JPG with one continuous colored line, linear axes, and evenly spaced observations.')
    path = upload if source == 'Upload' else ROOT/'sample_images'/f'{source}.png'
    if path is None:
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
    image_id = hashlib.sha256(im.tobytes() + str(im.shape).encode() + source.encode() + b'time-axis-v1').hexdigest()
    calibration_keys = BOUND_KEYS + ['ymin', 'ymax', 'observations', 'curve_color', 'time_kind']
    if (st.session_state.get('calibration_image') != image_id
            or any(key not in st.session_state for key in calibration_keys)):
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
        axis_defaults = [None]*4 if source == 'Upload' else [0., 59., 0., 100.]
        for key, value in zip(['xmin', 'xmax', 'ymin', 'ymax'], axis_defaults):
            st.session_state[key] = value
        st.session_state.curve_color = color
        st.session_state.observations = None if source == 'Upload' else 60
        st.session_state.time_kind = 'Numeric'
        st.session_state.axis_editor_kind = 'Numeric'
        st.session_state.detected_time_axis = {'kind':'numeric','start':axis_defaults[0],'end':axis_defaults[1]}
        if source == 'Upload':
            message += ' ' + fill_axis_fields(im, defaults)
        st.session_state.approved_calibration = None
        st.session_state.detection_message = message

    st.subheader('1. Check calibration')
    st.info(st.session_state.detection_message)
    with st.expander('Adjust plot rectangle (pixels)', expanded=False):
        st.caption(f'Image: {width} × {height} pixels. Origin is the top-left corner; exclude labels and legends.')
        cols = st.columns(4)
        bounds = tuple(cols[i].number_input(label, 0, int(limit-1), key=key)
                       for i, (label, key, limit) in enumerate(zip(['Left','Top','Right','Bottom'], BOUND_KEYS, [width,height,width,height])))
    time_kind = st.selectbox('Time axis', KINDS, key='time_kind',
                             help='Recognized from the labels. Weekdays and months keep their names without assigning a calendar year.')
    kind = REVERSE_KEYS[time_kind]
    if st.session_state.axis_editor_kind != time_kind:
        st.session_state.axis_editor_kind = time_kind
        st.session_state.time_start = ''
        st.session_state.time_end = ''
        if kind == 'numeric':
            st.session_state.xmin = None
            st.session_state.xmax = None
    cols = st.columns(4)
    time_error = None
    if kind == 'numeric':
        xmin = cols[0].number_input('Time minimum', value=None, key='xmin', placeholder='From your chart')
        xmax = cols[1].number_input('Time maximum', value=None, key='xmax', placeholder='From your chart')
    else:
        example = {'clock':'12:20','weekday':'Mon','month':'Jan','date':'2026-10-25'}[kind]
        start_text = cols[0].text_input('Start time', key='time_start', placeholder=example)
        end_text = cols[1].text_input('End time', key='time_end', placeholder=example,
                                     help='For a repeated cycle, append (+1 day), (+1 week), or (+1 year). Dates require an explicit year.')
        try:
            xmin,xmax = parse_bounds(kind,start_text,end_text)
        except ValueError as exc:
            xmin,xmax = None,None
            time_error = str(exc)
    ymin = cols[2].number_input('Value minimum', value=None, key='ymin', placeholder='From your chart')
    ymax = cols[3].number_input('Value maximum', value=None, key='ymax', placeholder='From your chart')
    axes = (xmin,xmax,ymin,ymax)
    time_axis = {'kind':kind,'start':xmin,'end':xmax}
    cols = st.columns([2, 1])
    count = cols[0].number_input('Number of original observations', 5, 300, value=None, key='observations', placeholder='From your chart',
                                help='Suggested from dense, regularly spaced labels. Check the original sample count; tick labels do not always mark every observation.')
    color = cols[1].color_picker('Curve color', key='curve_color')
    with st.expander('Advanced extraction settings'):
        distance = st.slider('Color tolerance', 10, 200, 90)
    rgb = tuple(int(color[i:i+2], 16) for i in (1,3,5))
    preview = Image.fromarray(im)
    draw = ImageDraw.Draw(preview)
    if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
        draw.rectangle(bounds, outline='#00a050', width=2)
    missing_values = any(value is None for value in axes) or count is None
    try:
        df = None if missing_values else extract(im, bounds, axes, count, rgb, distance)
        if df is not None:
            for x, y in zip(df.pixel_x, df.pixel_y):
                draw.ellipse((x-2,y-2,x+2,y+2), fill='#e53935')
        extraction_error = None
    except ValueError as exc:
        df, extraction_error = None, str(exc)
    st.image(preview, width=750)
    st.markdown(
        '<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.875rem;margin:4px 0 16px;">'
        '<span><span style="display:inline-block;width:14px;height:10px;border:2px solid #00a050;'
        'margin-right:7px;" aria-hidden="true"></span>Plot boundary</span>'
        '<span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        'background:#e53935;margin-right:7px;" aria-hidden="true"></span>Recovered observations</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if missing_values:
        if time_error and (start_text or end_text):
            st.warning(time_error)
        st.button('Extract series', type='primary', disabled=True,
                  help='Enter the four axis limits and original observation count above to continue.')
        return
    if extraction_error:
        st.warning(extraction_error)
        return
    if source == 'Upload':
        st.caption(f'Time: {format_time(kind,xmin,xmin)} to {format_time(kind,xmax,xmin)} · Values: {axes[2]:g} to {axes[3]:g} · {count} observations. '
                   'Adjust the fields above to match your chart.')
        calibration = (image_id, bounds, axes, kind, count, color, distance)
        if st.button('Extract series', type='primary', key='extract_series'):
            st.session_state.approved_calibration = calibration
        if st.session_state.approved_calibration != calibration:
            return

    st.subheader('2. Inspect the recovered series')
    history = go.Figure(go.Scatter(x=df.time, y=df.value, name='Recovered series', mode='lines+markers', marker_size=4))
    history.update_layout(xaxis_title='Time', yaxis_title='Value',
                          xaxis_range=[axes[0], axes[1]], yaxis_range=[axes[2], axes[3]],
                          height=380, margin=dict(t=20))
    display_labels = [format_time(kind,t,xmin) for t in df.time]
    if kind != 'numeric':
        history.data[0].update(customdata=display_labels,
                               hovertemplate='%{customdata}<br>Value: %{y:.3f}<extra></extra>')
    apply_time_axis(history,time_axis,df.time)
    st.plotly_chart(history, width=850)
    st.markdown('**Series overview**')
    st.write(series_overview(df.time,df.value,time_axis))
    if source != 'Upload':
        truth = pd.read_csv(ROOT/'sample_images'/f'{source}.csv')
        if count == len(truth) and np.allclose(axes, [0,59,0,100]):
            st.caption(f'Average extraction error on this demo (MAE): {np.mean(abs(df.value-truth.value)):.3f} value units.')
    st.download_button('Download recovered data', export_series(df,time_axis).to_csv(index=False), 'extracted.csv', 'text/csv')

    st.subheader('3. Forecast and evaluate')
    cols = st.columns(3)
    horizon = cols[0].number_input('Forecast horizon', 1, min(20,count//3), min(5,count//3))
    if st.session_state.get('forecast_model') not in MODELS:
        st.session_state.forecast_model = 'ETS'
    model = cols[2].selectbox('Forecast model', MODELS, key='forecast_model',
                            help='ETS: damped trend smoothing. ARIMA: selects among three small models using training data only. Naive: repeats the last value.')
    minimum = ARIMA_MIN_TRAIN if model == 'ARIMA' else 4
    if count-horizon < minimum:
        st.info('ARIMA needs at least 12 training observations plus the forecast horizon. Choose ETS or Naive, or reduce the horizon.')
        return
    initial = cols[1].number_input('Initial training observations', minimum, count-horizon, max(minimum,count//2))
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
    st.caption(f'Based on {len(terminal)} rolling origins at lead time {horizon}.')
    if len(terminal) < 20:
        st.warning('Fewer than 20 validation origins: the failure rate is based on limited evidence.')
    if model != 'Naive' and selected['MAE'] >= naive['MAE']:
        st.info(f'The naive baseline matches or beats {model} at this lead time. A more complex model does not always forecast better.')
    times = df.time.iloc[-1] + np.arange(1,horizon+1)*(axes[1]-axes[0])/(count-1)
    forecast = pd.DataFrame({'time':times, 'forecast':future})
    fig = go.Figure(go.Scatter(x=df.time,y=y,name='History'))
    fig.add_trace(go.Scatter(x=np.r_[df.time.iloc[-1],times], y=np.r_[y[-1],future], name=model+' forecast', line=dict(dash='dash'), mode='lines+markers'))
    fig.add_vline(x=float(df.time.iloc[-1]), line_dash='dot', line_color='gray')
    fig.update_layout(xaxis_title='Time', yaxis_title='Value', height=380, margin=dict(t=20))
    if kind != 'numeric':
        fig.data[0].update(customdata=display_labels,
                            hovertemplate='%{customdata}<br>Value: %{y:.3f}<extra></extra>')
        fig.data[1].update(customdata=[format_time(kind,t,xmin) for t in np.r_[df.time.iloc[-1],times]],
                            hovertemplate='%{customdata}<br>Forecast: %{y:.3f}<extra></extra>')
    apply_time_axis(fig,time_axis,np.r_[df.time.to_numpy(),times])
    st.plotly_chart(fig, width='stretch')
    comparison = pd.DataFrame([{'Model':model, **selected}, {'Model':'Naive baseline', **naive}]) if model != 'Naive' else pd.DataFrame([{'Model':'Naive', **selected}])
    st.dataframe(comparison.set_index('Model').style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', 'Historical failure rate':'{:.1%}'}), width='stretch')
    with st.expander('Understand the forecast errors'):
        st.markdown('**Historical forecast error distribution**')
        within = int((terminal.abs() <= tolerance).sum())
        st.write(f'{within} of {len(terminal)} past forecasts were within your ±{tolerance:g} tolerance '
                 f'when predicting {horizon} observations ahead.')
        st.plotly_chart(error_distribution(terminal, tolerance), width='stretch',
                        config={'displayModeBar': False})
        st.markdown('**Forecast accuracy by lead time**')
        detail = metric_table(results,tolerance).rename_axis('Observations ahead').rename(columns={'Count':'Past forecasts'})
        st.dataframe(detail.style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', 'Historical failure rate':'{:.1%}'}), width='stretch')
    cols = st.columns(3)
    cols[0].download_button('Download forecast',
                            fig.to_html(full_html=True, include_plotlyjs=True),
                            'forecast_chart.html', 'text/html',
                            help='Save the interactive history and forecast chart. Open the HTML file in a browser; it works offline.')
    cols[1].download_button('Download forecast data (CSV)', export_series(forecast,time_axis).to_csv(index=False), 'forecast.csv', 'text/csv')
    backtest_export = results.copy()
    backtest_export['forecast_origin_time'] = [display_labels[int(o)-1] for o in results.origin]
    backtest_export['target_time'] = [display_labels[int(o+h)-1] for o,h in zip(results.origin,results.horizon)]
    cols[2].download_button('Download backtest', backtest_export.to_csv(index=False), 'backtest.csv', 'text/csv')
    st.info('Models are evaluated on the same historical periods, using only data available before each forecast. '
            'Historical failure rates describe past performance and cannot provide an exact probability of future error. '
            'Overlapping tests are dependent, and changing conditions or extraction errors can affect reliability.')


if __name__ == '__main__':
    main()
