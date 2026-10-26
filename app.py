"""Local, guided graph digitization and forecast evaluation."""
from pathlib import Path
import hashlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from extraction import extract, detect_plot_area, suggest_curve_color, suggest_observation_count
from forecasting import (predict, backtest, summarize, MODELS, ARIMA_MIN_TRAIN, compare_models, select_best_model,
                         percentage_errors, outside_tolerance)
from calibration import read_calibration
from time_axis import (KINDS, KEYS, REVERSE_KEYS, FREQUENCIES, display_time, forecast_labels, parse_bounds, daily_sampling_points,
                       apply_time_axis, export_series, series_overview, calibrated_times, calendar_sample_positions)

ROOT = Path(__file__).parent
CALIBRATION_VERSION = 'chart-formats-v8'
BOUND_KEYS = ['plot_left', 'plot_top', 'plot_right', 'plot_bottom']
AXIS_KEYS = ['xmin', 'xmax', 'ymin', 'ymax']
DEFAULT_EXTRACTION_POINTS = 100
MAX_EXTRACTION_POINTS = 300


def style_chart(fig):
    """Keep the on-screen charts and downloaded forecast visually consistent."""
    fig.update_layout(
        template='plotly_white', paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family='Segoe UI, Arial, sans-serif',color='#20343C',size=13),
        hoverlabel=dict(bgcolor='#FFFFFF',font_family='Segoe UI, Arial, sans-serif'),
        colorway=['#0F766E','#D97706'],
    )
    fig.update_xaxes(gridcolor='#E8EEEB',zerolinecolor='#D8E1DD',automargin=True)
    fig.update_yaxes(gridcolor='#E8EEEB',zerolinecolor='#D8E1DD',automargin=True)
    return fig


def custom_extraction_count():
    st.session_state.manual_extraction_points = True
    if (st.session_state.detected_observation_count is None and st.session_state.time_kind == 'Dates'
            and st.session_state.observation_frequency == 'Daily'):
        st.session_state.observation_frequency = 'Unknown'


def change_sampling_frequency():
    if (st.session_state.observation_frequency == 'Daily' and st.session_state.time_kind == 'Dates'
            and st.session_state.detected_observation_count is None):
        st.session_state.manual_extraction_points = False


@st.cache_data(show_spinner=False, max_entries=24)
def detect_axis_values(image, bounds, calibration_version):
    """Separate cached OCR results when supported label formats change."""
    return read_calibration(image, bounds)


@st.cache_data(show_spinner=False, max_entries=24)
def detect_point_count(image, bounds):
    try:
        return suggest_observation_count(image,bounds,suggest_curve_color(image,bounds))
    except ValueError:
        return None


def fill_axis_fields(image, bounds):
    """Optional OCR failure must not prevent manual extraction."""
    marker_count = detect_point_count(image,tuple(bounds))
    try:
        with st.spinner('Reading chart axes…'):
            result = detect_axis_values(image, tuple(bounds), CALIBRATION_VERSION)
        for key, value in zip(AXIS_KEYS, result['axes']):
            st.session_state[key] = value
        detected_count = marker_count if marker_count is not None else result['count']
        st.session_state.detected_observation_count = detected_count
        st.session_state.observations = detected_count if detected_count is not None else DEFAULT_EXTRACTION_POINTS
        axis = result.get('time_axis', {'kind':'numeric', 'start':result['axes'][0], 'end':result['axes'][1]})
        st.session_state.detected_time_axis = axis
        st.session_state.time_kind = KEYS[axis['kind']]
        st.session_state.axis_editor_kind = KEYS[axis['kind']]
        st.session_state.observation_frequency = 'Intraday' if axis.get('has_time_labels') or axis['kind']=='clock' else 'Unknown'
        if axis['kind']=='date' and not axis.get('has_time_labels') and detected_count is None:
            st.session_state.observation_frequency = 'Daily'
        if axis['kind'] != 'numeric':
            display_axis = dict(axis,frequency=st.session_state.observation_frequency)
            st.session_state.time_start = display_time(display_axis,axis['start'])
            st.session_state.time_end = display_time(display_axis,axis['end'])
        if any(value is None for value in result['axes']):
            return 'Some axis labels could not be read reliably. Fill in the missing values below.'
        if axis['kind']=='month_day':
            period_note = ' Dates are approximate; no calendar year is shown.'
        elif axis.get('anchors') is not None:
            period_note = ' Calendar positions are interpolated between the visible date labels.'
        elif axis.get('estimated_clock_range'):
            period_note = ' Start and end times are approximate.'
        else:
            period_note = ' The period is estimated from month/year ticks.' if axis.get('estimated_period') else ''
        if detected_count is None:
            return 'Axis values detected.' + period_note
        count_source = 'visible point markers' if marker_count is not None else 'axis labels'
        return f'Axis values detected. Point count suggested from {count_source}; check it matches your chart.' + period_note
    except Exception:
        logging.getLogger(__name__).warning('Automatic axis reading failed', exc_info=True)
        st.session_state.detected_observation_count = marker_count
        st.session_state.observations = marker_count if marker_count is not None else DEFAULT_EXTRACTION_POINTS
        return 'Automatic axis reading is unavailable. You can enter the values below.'


@st.cache_data(show_spinner=False, max_entries=24)
def evaluate(values, horizon, initial, model):
    return backtest(values, horizon, initial, model)


@st.cache_data(show_spinner=False, max_entries=24)
def compare_auto_candidates(training_values, horizon):
    """Cache matching errors and actuals for percentage-based selection."""
    return compare_models(training_values,horizon)


def metric_table(results, tolerance):
    return pd.DataFrame([
        {'Lead time': h, **summarize(group.error, tolerance,group.actual)}
        for h, group in results.groupby('horizon')
    ]).set_index('Lead time')


def error_distribution(errors, tolerance, actuals):
    """Use common bins so colored counts add up to the observed errors exactly."""
    percentages = percentage_errors(errors,actuals)
    failed = outside_tolerance(percentages,tolerance)
    finite = np.isfinite(percentages)
    values = percentages[finite].copy()
    failed = failed[finite]
    # Put float-roundoff boundary matches exactly on their displayed boundary.
    near = ~failed & (np.abs(values)>tolerance)
    values[near] = np.sign(values[near])*tolerance
    edges = np.histogram_bin_edges(values if len(values) else [0.], bins=min(16, max(6, int(np.sqrt(len(values))))))
    # Put tolerance boundaries on bin edges instead of coloring mixed bins by center.
    cuts = [v for v in (-tolerance, 0, tolerance) if edges[0] < v < edges[-1]]
    edges = np.unique(np.r_[edges, cuts])
    centers, widths = (edges[:-1]+edges[1:])/2, np.diff(edges)
    fig = go.Figure()
    groups = [
        ('Overestimate', failed & (values<0), '#6366F1'),
        ('Within tolerance', ~failed, '#0D9488'),
        ('Underestimate', failed & (values>0), '#EA580C'),
    ]
    for label, mask, color in groups:
        group_values = values[mask].copy()
        if label=='Within tolerance' and tolerance>0:
            group_values[group_values==tolerance] = np.nextafter(tolerance,-np.inf)
        counts, _ = np.histogram(group_values, bins=edges)
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=widths*.94, name=label,
            marker=dict(color=color, line=dict(color='white', width=1)),
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate=(label+'<br>Error range: %{customdata[0]:.2f}% to %{customdata[1]:.2f}%'
                           '<br>Past forecasts: %{y}<extra></extra>'),
        ))
    fig.add_vrect(x0=-tolerance, x1=tolerance, fillcolor='#0D9488', opacity=.07, line_width=0, layer='below')
    for boundary in sorted(set([-tolerance, tolerance])):
        fig.add_vline(x=boundary, line_dash='dash', line_color='#64748B', line_width=1.5)
    fig.add_vline(x=0, line_color='#334155', line_width=1)
    low, high = min(float(values.min()) if len(values) else 0, -tolerance), max(float(values.max()) if len(values) else 0, tolerance)
    padding = max((high-low)*.08, .5)
    fig.update_layout(
        barmode='stack', template='plotly_white', height=360,
        margin=dict(t=15, b=65, l=20, r=20),
        legend=dict(orientation='h', yanchor='top', y=-.27, x=0),
        xaxis=dict(title='Forecast error (%)', zeroline=False,
                   range=[min(low-padding, float(edges[0])), max(high+padding, float(edges[-1]))]),
        yaxis=dict(title='Number of past forecasts', dtick=1 if len(values)<15 else None, rangemode='tozero'),
        hoverlabel=dict(bgcolor='white', font_size=13),
    )
    unbounded = int((~finite).sum())
    if unbounded:
        fig.add_annotation(text=f'{unbounded} unbounded percentage errors are not plotted; all count as failures.',
                           x=.5,y=1.12,xref='paper',yref='paper',showarrow=False)
        fig.update_layout(margin=dict(t=55,b=65,l=20,r=20))
    return style_chart(fig)


def main():
    st.set_page_config(page_title='Time Series Digitizer & Forecaster', page_icon='📈', layout='wide')
    st.html('<style>' + (ROOT/'ui.css').read_text(encoding='utf-8') + '</style>')
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
    image_id = hashlib.sha256(im.tobytes() + str(im.shape).encode() + source.encode() + CALIBRATION_VERSION.encode()).hexdigest()
    calibration_keys = BOUND_KEYS + ['ymin', 'ymax', 'observations', 'curve_color', 'time_kind', 'observation_frequency', 'detected_observation_count', 'manual_extraction_points']
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
        st.session_state.observations = DEFAULT_EXTRACTION_POINTS if source == 'Upload' else 60
        st.session_state.detected_observation_count = None if source == 'Upload' else 60
        st.session_state.manual_extraction_points = False
        st.session_state.time_kind = 'Numeric'
        st.session_state.observation_frequency = 'Unknown'
        st.session_state.axis_editor_kind = 'Numeric'
        st.session_state.detected_time_axis = {'kind':'numeric','start':axis_defaults[0],'end':axis_defaults[1]}
        if source == 'Upload':
            message += ' ' + fill_axis_fields(im, defaults)
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
        st.session_state.observation_frequency = 'Unknown'
        if kind == 'date' and st.session_state.detected_observation_count is None:
            st.session_state.observation_frequency = 'Daily'
            st.session_state.manual_extraction_points = False
        if kind == 'numeric':
            st.session_state.xmin = None
            st.session_state.xmax = None
    cols = st.columns(4)
    time_error = None
    if kind == 'numeric':
        xmin = cols[0].number_input('Time minimum', value=None, key='xmin', placeholder='From your chart')
        xmax = cols[1].number_input('Time maximum', value=None, key='xmax', placeholder='From your chart')
    else:
        example = {'clock':'12:20 or 1:00 PM','weekday':'Mon','month':'Jan','month_day':'Oct 21','date':'2026-10-25'}[kind]
        start_text = cols[0].text_input('Start time', key='time_start', placeholder=example)
        end_text = cols[1].text_input('End time', key='time_end', placeholder=example,
                                     help='For a repeated cycle, append (+1 day), (+1 week), or (+1 year). Full dates require a year; month/day labels keep it unspecified.')
        try:
            xmin,xmax = parse_bounds(kind,start_text,end_text)
            detected = st.session_state.detected_time_axis
            if (kind=='month_day' and detected.get('kind')==kind
                    and start_text==display_time(detected,detected['start'])
                    and end_text==display_time(detected,detected['end'])):
                xmin,xmax = detected['start'],detected['end']
        except ValueError as exc:
            xmin,xmax = None,None
            time_error = str(exc)
    ymin = cols[2].number_input('Value minimum', value=None, key='ymin', placeholder='From your chart')
    ymax = cols[3].number_input('Value maximum', value=None, key='ymax', placeholder='From your chart')
    axes = (xmin,xmax,ymin,ymax)
    time_axis = dict(st.session_state.detected_time_axis) if st.session_state.detected_time_axis.get('kind') == kind else {}
    frequency = st.selectbox('Sampling frequency', FREQUENCIES, key='observation_frequency', on_change=change_sampling_frequency,
                            help='For full dates with an unknown original count, Daily creates one interpolated point per calendar day. Yearless labels and clock charts retain their detected count or use custom sampling.') if kind != 'numeric' else 'Unknown'
    time_axis.update(kind=kind,start=xmin,end=xmax,frequency=frequency)
    if kind == 'date':
        time_axis['has_time_labels'] = bool(time_axis.get('has_time_labels') or ':' in start_text or ':' in end_text)
    if kind=='clock':
        time_axis['has_second_labels'] = bool(time_axis.get('has_second_labels') or start_text.count(':')>=2 or end_text.count(':')>=2)
    detected_count = st.session_state.detected_observation_count
    daily_count = daily_sampling_points(kind,xmin,xmax) if frequency=='Daily' and detected_count is None else None
    if detected_count is None and not st.session_state.manual_extraction_points:
        st.session_state.observations = daily_count if daily_count is not None else DEFAULT_EXTRACTION_POINTS
    cols = st.columns([2, 1])
    count = cols[0].number_input('Extraction points', 2, max(MAX_EXTRACTION_POINTS,st.session_state.observations or 0), key='observations', on_change=custom_extraction_count,
                                help='Uses a detected original count when available. Date-only charts otherwise use the number of calendar days, including both endpoints. Editing this count switches to custom sampling. Other unknown counts start at 100 samples.')
    color = cols[1].color_picker('Curve color', key='curve_color')
    sampled = detected_count is None or count != detected_count
    time_axis.update(sampled=sampled,extraction_points=count)
    daily_sampling = daily_count is not None and count==daily_count
    if daily_sampling:
        time_axis['sampling_interval'] = 'calendar_day'
    elif detected_count is None and count is not None:
        st.caption(f'Original count unknown. Using {count} evenly spaced samples.')
    elif sampled:
        st.caption(f'Using {count} evenly spaced samples; the suggested original count was {detected_count}.')
    with st.expander('Advanced extraction settings'):
        distance = st.slider('Color tolerance', 10, 200, 90)
    rgb = tuple(int(color[i:i+2], 16) for i in (1,3,5))
    point_color = '#2563eb' if np.linalg.norm(np.array(rgb)-[229,57,53]) < 120 else '#e53935'
    preview = Image.fromarray(im)
    draw = ImageDraw.Draw(preview)
    if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
        draw.rectangle(bounds, outline='#00a050', width=2)
    missing_values = any(value is None for value in axes) or count is None
    if count is not None and count > MAX_EXTRACTION_POINTS:
        st.warning(f'This selection creates {count} points. Forecasting supports up to {MAX_EXTRACTION_POINTS} extraction points; shorten the period or choose a smaller custom count.')
        return
    try:
        positions = None if missing_values else calendar_sample_positions(time_axis,bounds,count)
        df = None if missing_values else extract(im, bounds, axes, count, rgb, distance,pixel_positions=positions)
        if df is not None and time_axis.get('anchors') is not None:
            df['time'] = calibrated_times(df.pixel_x,time_axis,bounds)
        if df is not None:
            for x, y in zip(df.pixel_x, df.pixel_y):
                draw.ellipse((x-2,y-2,x+2,y+2), fill=point_color)
        extraction_error = None
    except ValueError as exc:
        df, extraction_error = None, str(exc)
    if df is None:
        try:
            # A trace preview remains useful while unreadable axes await correction.
            trace = extract(im,bounds,(0.,1.,0.,1.),min(150,width),rgb,distance)
            for x,y in zip(trace.pixel_x,trace.pixel_y):
                draw.ellipse((x-1,y-1,x+1,y+1),fill=point_color)
        except ValueError:
            pass
    st.image(preview, width=680)
    st.markdown(
        '<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.875rem;margin:4px 0 16px;">'
        '<span><span style="display:inline-block;width:14px;height:10px;border:2px solid #00a050;'
        'margin-right:7px;" aria-hidden="true"></span>Plot boundary</span>'
        '<span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{point_color};margin-right:7px;" aria-hidden="true"></span>Extracted points</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if missing_values:
        if time_error and (start_text or end_text):
            st.warning(time_error)
        st.caption('Fill in the missing axis limits to show the series and forecast.')
        return
    if extraction_error:
        st.warning(extraction_error)
        return
    st.subheader('2. Inspect the recovered series')
    history = go.Figure(go.Scatter(x=df.time, y=df.value, name='Recovered series', mode='lines+markers', marker_size=4))
    history.update_traces(line=dict(color='#0F766E',width=2.5),marker_color='#0F766E')
    history.update_layout(xaxis_title='Time', yaxis_title='Value',
                          xaxis_range=[axes[0], axes[1]], yaxis_range=[axes[2], axes[3]],
                          height=380, margin=dict(t=20))
    display_labels = [display_time(time_axis,t) for t in df.time]
    if kind != 'numeric':
        history.data[0].update(customdata=display_labels,
                               hovertemplate='%{customdata}<br>Value: %{y:.3f}<extra></extra>')
    apply_time_axis(history,time_axis,df.time)
    st.plotly_chart(style_chart(history), width=850, theme=None)
    st.markdown('**Series overview**')
    st.write(series_overview(df.time,df.value,time_axis))
    if source != 'Upload':
        truth = pd.read_csv(ROOT/'sample_images'/f'{source}.csv')
        if count == len(truth) and np.allclose(axes, [0,59,0,100]):
            st.caption(f'Average extraction error on this demo (MAE): {np.mean(abs(df.value-truth.value)):.3f} value units.')
    st.download_button('Download recovered data', export_series(df,time_axis).to_csv(index=False), 'extracted.csv', 'text/csv')

    st.subheader('3. Forecast and evaluate')
    if sampled:
        st.info('Backtest on sampled data: results evaluate the sampled curve. Interpolation can make errors appear smaller.')
    cols = st.columns(3)
    horizon = cols[0].number_input('Forecast horizon', 1, min(20,count//3), min(5,count//3),
                                  help='Number of future steps in the extracted series. For sampled data, each step is one sample. Exact future dates and trading sessions cannot be recovered from the image.')
    if st.session_state.get('forecast_model') not in MODELS:
        st.session_state.forecast_model = 'ETS'
    method = cols[2].selectbox('Forecast model', MODELS, key='forecast_model',
                            help='Auto: compares ETS, ARIMA and Naive on earlier data, using failure rate then MAE. ETS: damped trend. ARIMA: restricted order selection. Naive: repeats the last value.')
    minimum = horizon+4 if method=='Auto' else (ARIMA_MIN_TRAIN if method == 'ARIMA' else 4)
    if count-horizon < minimum:
        if method=='Auto':
            st.info(f'Auto needs at least {2*horizon+4} points to separate model selection from evaluation. Reduce the horizon or choose a model directly.')
        else:
            st.info('ARIMA needs at least 12 training observations plus the forecast horizon. Choose ETS or Naive, or reduce the horizon.')
        return
    initial = cols[1].number_input('Initial training points', minimum, count-horizon, max(minimum,count//2),
                                   help='In Auto, these earlier points are used for model selection. The following points evaluate the selected model.')
    tolerance = st.number_input('Failure tolerance (%)', min_value=0., value=10.,step=.5,
                                help='Allowed deviation from each actual value. At 10%, an actual value of 60 allows predictions from 54 to 66. At zero, only an exact match is within tolerance.')
    y = df.value.to_numpy()
    selection = None
    model = method
    try:
        with st.spinner('Evaluating historical forecasts…'):
            if method=='Auto':
                selection = compare_auto_candidates(y[:initial],horizon)
                model, selection_scores = select_best_model(selection,tolerance)
            results = evaluate(y, horizon, initial, model)
            baseline = evaluate(y, horizon, initial, 'Naive')
            future = predict(y, horizon, model)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        context = f'Auto selected {model}, but forecasting failed' if method=='Auto' and model!='Auto' else 'Forecast could not be calculated'
        st.error(f'{context}: {exc}')
        return
    if selection is not None:
        st.markdown(f'**Auto selected {model}**')
        st.caption('Earlier data selects the model; the later backtest measures its performance.')
        with st.expander('Auto selection scores (earlier data)'):
            st.caption(f'Selection: first {initial} points, {len(selection["origins"])} common rolling origins at lead time {horizon}. '
                       'Lowest failure rate wins; ties use MAE, then the simpler model. The winner stays fixed during evaluation.')
            selection_table = selection_scores.rename(columns={'Historical failure rate':'Selection failure rate','Count':'Selection tests'}).copy()
            selection_table['Failures'] = [int(outside_tolerance(percentage_errors(selection['errors'][candidate],selection['actuals']),tolerance).sum()) for candidate in selection_table.index]
            selection_table = selection_table[['Selection failure rate','Failures','Selection tests','MAE','RMSE']]
            st.dataframe(selection_table.style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', 'Selection failure rate':'{:.1%}'}),width='stretch')
            for candidate,reason in selection['skipped'].items():
                st.caption(f'{candidate} unavailable for selection: {reason}')
    terminal_rows = results.loc[results.horizon==horizon]
    terminal = terminal_rows.error
    terminal_percent = percentage_errors(terminal,terminal_rows.actual)
    terminal_failed = outside_tolerance(terminal_percent,tolerance)
    selected = summarize(terminal,tolerance,terminal_rows.actual)
    baseline_rows = baseline.loc[baseline.horizon==horizon]
    naive = summarize(baseline_rows.error,tolerance,baseline_rows.actual)
    cols = st.columns(3)
    cols[0].metric(f'MAE (units) · {horizon}-step ahead', f'{selected["MAE"]:.3f}')
    cols[1].metric(f'RMSE (units) · {horizon}-step ahead', f'{selected["RMSE"]:.3f}')
    failure_label = 'Backtest failure rate' if selection is not None else 'Historical failure rate'
    cols[2].metric(failure_label, f'{selected["Historical failure rate"]:.1%}')
    if selection is not None:
        failures = int(terminal_failed.sum())
        st.caption(f'Later backtest: {failures} failures in {len(terminal)} forecasts at lead time {horizon}.')
    else:
        st.caption(f'Based on {len(terminal)} rolling origins at lead time {horizon}.')
    if len(terminal) < 20:
        st.warning('Fewer than 20 validation origins: the failure rate is based on limited evidence.')
    if model != 'Naive' and selected['MAE'] >= naive['MAE']:
        st.info(f'The naive baseline matches or beats {model} at this lead time. A more complex model does not always forecast better.')
    times = df.time.iloc[-1] + np.arange(1,horizon+1)*(axes[1]-axes[0])/(count-1)
    forecast = pd.DataFrame({'time':times, 'forecast':future})
    future_labels = forecast_labels(time_axis,horizon)
    fig = go.Figure(go.Scatter(x=df.time,y=y,name='History',line=dict(color='#0F766E',width=2.5)))
    fig.add_trace(go.Scatter(x=np.r_[df.time.iloc[-1],times], y=np.r_[y[-1],future], name=model+' forecast', line=dict(dash='dash',color='#D97706',width=2.5), mode='lines+markers'))
    fig.add_vline(x=float(df.time.iloc[-1]), line_dash='dot', line_color='gray')
    fig.update_layout(xaxis_title='Time', yaxis_title='Value', height=380, margin=dict(t=20))
    if sampled:
        fig.update_layout(title=f'Forecast from {count} sampled points',margin=dict(t=55))
    if method=='Auto':
        title = f'Forecast from {count} sampled points · Auto: {model}' if sampled else f'Auto selected {model}'
        fig.update_layout(title=title,margin=dict(t=55))
    if kind != 'numeric':
        fig.data[0].update(customdata=display_labels,
                            hovertemplate='%{customdata}<br>Value: %{y:.3f}<extra></extra>')
        fig.data[1].update(customdata=[display_labels[-1]]+future_labels,
                            hovertemplate='%{customdata}<br>Forecast: %{y:.3f}<extra></extra>')
    apply_time_axis(fig,time_axis,np.r_[df.time.to_numpy(),times],history_count=count)
    if kind != 'numeric':
        st.caption('Forecasts use future steps in the extracted series.')
    st.plotly_chart(style_chart(fig), width='stretch', theme=None)
    comparison = pd.DataFrame([{'Model':model, **selected}, {'Model':'Naive baseline', **naive}]) if model != 'Naive' else pd.DataFrame([{'Model':'Naive', **selected}])
    st.dataframe(comparison.set_index('Model').rename(columns={'Historical failure rate':failure_label}).style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', failure_label:'{:.1%}'}), width='stretch')
    with st.expander('Understand the forecast errors'):
        st.markdown('**Historical forecast error distribution**')
        within = int((~terminal_failed).sum())
        step_unit = 'samples' if sampled else 'observations'
        st.write(f'{within} of {len(terminal)} past forecasts were within your ±{tolerance:g}% tolerance '
                 f'when predicting {horizon} {step_unit} ahead.')
        st.plotly_chart(error_distribution(terminal,tolerance,terminal_rows.actual), width='stretch',
                        config={'displayModeBar': False}, theme=None)
        st.markdown('**Forecast accuracy by lead time**')
        detail = metric_table(results,tolerance).rename_axis('Samples ahead' if sampled else 'Observations ahead').rename(columns={'Count':'Past forecasts','Historical failure rate':failure_label})
        st.dataframe(detail.style.format({'MAE':'{:.3f}', 'RMSE':'{:.3f}', failure_label:'{:.1%}'}), width='stretch')
    cols = st.columns(3)
    cols[0].download_button('Download forecast',
                            fig.to_html(full_html=True, include_plotlyjs=True),
                            'forecast_chart.html', 'text/html',
                            type='primary',
                            help='Save the interactive history and forecast chart. Open the HTML file in a browser; it works offline.')
    forecast_export = export_series(forecast,time_axis)
    backtest_export = results.copy()
    backtest_export['error_percent'] = percentage_errors(results.error,results.actual)
    backtest_export['failed'] = outside_tolerance(backtest_export.error_percent,tolerance)
    backtest_export['tolerance_percent'] = tolerance
    backtest_export['forecast_origin_time'] = [display_labels[int(o)-1] for o in results.origin]
    backtest_export['target_time'] = [display_labels[int(o+h)-1] for o,h in zip(results.origin,results.horizon)]
    if sampled:
        backtest_export['sampling_basis'] = 'sampled_curve'
        backtest_export['extraction_points'] = count
        if daily_sampling:
            backtest_export['sampling_interval'] = 'calendar_day'
    if method=='Auto':
        for data in (forecast_export,backtest_export):
            data['forecast_method'] = 'Auto'
            data['selected_model'] = model
            data['selection_points'] = initial
            data['selection_tolerance_percent'] = tolerance
    cols[1].download_button('Download forecast data (CSV)', forecast_export.to_csv(index=False), 'forecast.csv', 'text/csv')
    cols[2].download_button('Download backtest', backtest_export.to_csv(index=False), 'backtest.csv', 'text/csv')
    st.info('Models are evaluated on the same historical periods, using only data available before each forecast. '
            'Historical failure rates describe past performance and cannot provide an exact probability of future error. '
            'Overlapping tests are dependent, and changing conditions or extraction errors can affect reliability.')


if __name__ == '__main__':
    main()
