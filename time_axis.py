"""Temporal axis recognition and display without invented calendar years."""
import re
from datetime import datetime, timedelta
import numpy as np

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
KINDS = ['Numeric', 'Clock time', 'Weekdays', 'Months', 'Dates']
KEYS = {'numeric':'Numeric', 'clock':'Clock time', 'weekday':'Weekdays', 'month':'Months', 'date':'Dates'}
REVERSE_KEYS = {v:k for k,v in KEYS.items()}
FREQUENCIES = ['Unknown', 'Daily', 'Intraday']


def parse_label(text, kind):
    text = text.strip().rstrip('.')
    if kind == 'clock':
        match = re.fullmatch(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', text)
        if match:
            h,m,s = (int(v or 0) for v in match.groups())
            if h < 24 and m < 60 and s < 60:
                return h*60+m+s/60
    elif kind == 'weekday':
        names = {'monday':0,'tuesday':1,'wednesday':2,'thursday':3,'friday':4,'saturday':5,'sunday':6,
                 'mon':0,'tue':1,'tues':1,'wed':2,'wen':2,'weds':2,'thu':3,'thur':3,'thurs':3,'fri':4,'sat':5,'sun':6}
        return names.get(text.lower())
    elif kind == 'month':
        for i, name in enumerate(MONTHS):
            if text.lower() == name.lower() or text.lower() == datetime(2000,i+1,1).strftime('%B').lower():
                return i
    elif kind == 'date':
        # Ambiguous numeric dates such as 03/04 and yearless dates are excluded.
        formats = ['%Y-%m-%d','%Y-%m-%d %H:%M','%Y/%m/%d','%d %b %Y','%d %B %Y','%b %d %Y','%B %d %Y','%b %Y','%B %Y']
        for fmt in formats:
            try:
                if not re.search(r'\b\d{4}\b', text):
                    continue
                parsed = datetime.strptime(text,fmt)
                return float(parsed.toordinal())+(parsed.hour*3600+parsed.minute*60+parsed.second)/86400
            except ValueError:
                pass
    return None


def unwrap(values, kind):
    period = {'clock':1440,'weekday':7,'month':12}.get(kind)
    if not period:
        return values
    result, offset = [], 0
    for value in values:
        if result and np.isclose(value, result[-1] % period):
            raise ValueError('Repeated adjacent time labels are ambiguous.')
        if result and value+offset < result[-1]:
            offset += period
        result.append(value+offset)
    return result


def detect_time_axis(labels, bounds, fit_axis):
    left,top,right,bottom = bounds
    width,height = right-left,bottom-top
    candidates = []
    for kind in ['date','clock','weekday','month']:
        ticks = []
        for text,score,box in labels:
            if score < .7:
                continue
            points = np.asarray(box)
            x,y = points.mean(axis=0)
            if not (left-.03*width <= x <= right+.03*width and bottom < y < bottom+max(45,.18*height)):
                continue
            value = parse_label(text,kind)
            if value is not None:
                ticks.append((float(x),float(value),text))
        ticks.sort()
        # Prefer one label at each pixel position (word and line boxes may duplicate it).
        unique = []
        for tick in ticks:
            if not unique or tick[0]-unique[-1][0] > 5:
                unique.append(tick)
        ticks = unique
        if len(ticks) < 3:
            continue
        try:
            values = unwrap([t[1] for t in ticks],kind)
            endpoints, accepted = fit_axis([(t[0],v) for t,v in zip(ticks,values)],left,right)
        except ValueError:
            continue
        if endpoints[1] <= endpoints[0]:
            continue
        count = None
        p = np.array([t[0] for t,v in zip(ticks,values) if any(np.isclose(v,a,rtol=0,atol=1e-5) for a in accepted)])
        if (len(accepted) >= 5 and p[0]-left < .035*width and right-p[-1] < .035*width
                and np.allclose(np.diff(accepted),np.median(np.diff(accepted)),rtol=.02,atol=1e-5)):
            count = len(accepted)
        month_ticks = kind == 'date' and any(re.fullmatch(r'[A-Za-z]+\s+\d{4}',t[2].strip()) for t in ticks)
        candidates.append((len(accepted), {'kind':kind,'start':endpoints[0],'end':endpoints[1],
                                           'labels':[t[2] for t in ticks], 'count':count,
                                           'estimated_period':month_ticks,
                                           'has_time_labels':kind == 'clock' or any(re.search(r'\d{1,2}:\d{2}',t[2]) for t in ticks)}))
    if not candidates:
        return None
    candidates.sort(key=lambda c:c[0],reverse=True)
    return candidates[0][1]


def format_time(kind, value, origin=None):
    value = float(value)
    origin = value if origin is None else float(origin)
    if kind == 'numeric':
        return f'{value:g}'
    if kind == 'date':
        date = datetime.fromordinal(int(np.floor(value)))+timedelta(days=value % 1)
        # Avoid floating point noise around exact days and minutes.
        date = date + timedelta(seconds=30)
        return date.strftime('%Y-%m-%d') if date.hour == date.minute == 0 else date.strftime('%Y-%m-%d %H:%M')
    period,unit = {'clock':(1440,'day'),'weekday':(7,'week'),'month':(12,'year')}[kind]
    cycles = int(np.floor(value/period)-np.floor(origin/period))
    if kind == 'clock':
        total = int(round(value*60))
        label = f'{(total//3600)%24:02d}:{(total//60)%60:02d}'
        if total % 60:
            label += f':{total%60:02d}'
        cycles = total//86400-int(np.floor(origin/1440))
    else:
        base = int(np.floor(value+1e-8))
        label = (DAYS if kind == 'weekday' else MONTHS)[base % period]
        fraction = value-base
        if not np.isclose(fraction,0,atol=1e-7):
            label += f' +{fraction:.4f} {"day" if kind == "weekday" else "month"}'
    if cycles:
        label += f' ({cycles:+d} {unit}{"s" if abs(cycles)!=1 else ""})'
    return label


def parse_bounds(kind, start, end):
    values = []
    period = {'clock':1440,'weekday':7,'month':12}.get(kind)
    for text in (start,end):
        match = re.fullmatch(r'(.*?)\s*\(([+-]?\d+) (day|week|year)s?\)',text.strip())
        base = match.group(1) if match else text
        fraction = re.fullmatch(r'(.*?)\s*\+(\d+(?:\.\d+)?)\s+(day|month)',base.strip())
        fractional_value = 0.
        if fraction:
            if kind not in ('weekday','month') or fraction.group(3) != ('day' if kind=='weekday' else 'month'):
                raise ValueError('The fractional time unit must match the selected axis.')
            base = fraction.group(1)
            fractional_value = float(fraction.group(2))
            if fractional_value >= 1:
                raise ValueError('Fractional offsets must be less than one day or month.')
        value = parse_label(base,kind)
        if value is None:
            raise ValueError('Use valid time labels, such as 12:20, Mon, Jan, or 2026-10-25.')
        if match:
            if period is None:
                raise ValueError('Cycle offsets apply only to clock times, weekdays and months.')
            if match.group(3) != {'clock':'day','weekday':'week','month':'year'}[kind]:
                raise ValueError('Use day offsets for clock times, week offsets for weekdays, and year offsets for months.')
            value += int(match.group(2))*period
        values.append(float(value)+fractional_value)
    if values[1] < values[0] and period and '(' not in end:
        values[1] += period
    if values[1] <= values[0]:
        raise ValueError('End time must follow start time. Use a cycle offset for repeated labels.')
    return tuple(values)


def display_time(axis, value, friendly=False):
    """Present calendar dates without implying unsupported intraday precision."""
    kind = axis['kind']
    if kind == 'date':
        date = datetime.fromordinal(int(np.floor(value)))+timedelta(days=float(value) % 1)
        show_clock = axis.get('frequency') == 'Intraday' and axis.get('has_time_labels', False)
        if friendly:
            label = f'{date.day} {MONTHS[date.month-1]} {date.year}'
            return label+date.strftime(' %H:%M') if show_clock else label
        return format_time(kind,value,axis['start']) if show_clock else date.strftime('%Y-%m-%d')
    if friendly and kind in ('month','weekday'):
        # A summary needs the month/day name, rather than a fractional tick offset.
        return format_time(kind,np.floor(value),np.floor(axis['start']))
    return format_time(kind,value,axis['start'])


def forecast_labels(axis, count):
    """A raster chart cannot establish future trading dates or sampling times."""
    unit = 'Daily observation' if axis.get('frequency') == 'Daily' else 'Observation'
    return [f'{unit} +{step}' for step in range(1,count+1)]


def apply_time_axis(figure, axis, positions, history_count=None):
    if axis['kind'] == 'numeric':
        return
    positions = np.asarray(positions,dtype=float)
    if history_count is not None:
        historical = positions[:history_count]
        indices = np.unique(np.linspace(0,len(historical)-1,min(6,len(historical))).round().astype(int))
        future = positions[history_count:]
        # A short forecast occupies little width; keep its label clear of the final history tick.
        if len(future) and len(indices)>1:
            indices = indices[:-1]
        future_indices = [len(future)-1] if len(future) else []
        ticks = np.r_[historical[indices], future[future_indices]]
        future_text = forecast_labels(axis,len(future))
        labels = [display_time(axis,p) for p in historical[indices]]+[future_text[i] for i in future_indices]
    elif axis['kind'] in ('month','weekday'):
        ticks = np.arange(np.ceil(positions.min()),np.floor(positions.max())+1)
    else:
        ticks = positions
    if history_count is None:
        if not len(ticks):
            ticks = positions
        indices = np.unique(np.linspace(0,len(ticks)-1,min(8,len(ticks))).round().astype(int))
        ticks = ticks[indices]
        labels = [display_time(axis,p) for p in ticks]
    figure.update_xaxes(tickmode='array',tickvals=ticks,ticktext=labels,
                        title=KEYS[axis['kind']],tickangle=-25 if axis['kind']=='date' else 0)


def export_series(frame, axis):
    result = frame[['time', 'value' if 'value' in frame else 'forecast']].copy()
    if axis['kind'] != 'numeric':
        result['time'] = (forecast_labels(axis,len(result)) if 'forecast' in frame else
                          [display_time(axis,v) for v in result.time])
    return result


def series_overview(times, values, axis):
    times,values = np.asarray(times),np.asarray(values)
    labels = [display_time(axis,t,friendly=True) for t in times]
    frequency = axis.get('frequency','Unknown')
    title = f' · {frequency.lower()} chart' if frequency != 'Unknown' else ''
    prefix = f'**{len(values)} observations{title}**  \n{labels[0]} – {labels[-1]}'
    approximate = axis.get('estimated_period',False) or (axis['kind'] in ('date','month','weekday')
                  and not (frequency == 'Intraday' and axis.get('has_time_labels'))
                  and np.any(~np.isclose(times,np.round(times),rtol=0,atol=1e-7)))
    if approximate:
        prefix += ' (approximate dates)' if axis['kind']=='date' else ' (approximate period)'
    peak, low = int(np.argmax(values)), int(np.argmin(values))
    if np.ptp(values) < 1e-9*max(1,float(np.max(abs(values)))):
        return prefix+f'\n\nValues stay approximately constant at **{values[0]:.3g}**.'
    when = 'around' if approximate or axis['kind'] in ('month','weekday') else 'at'
    extrema = (f'\n\n- **Estimated high:** {values[peak]:.3g}, {when} {labels[peak]}'
               f'\n- **Estimated low:** {values[low]:.3g}, {when} {labels[low]}')
    note = ('\n\nRepeated highs or lows show their first occurrence.'
            if np.count_nonzero(values == values[peak]) > 1 or np.count_nonzero(values == values[low]) > 1 else '')
    change = values[-1]-values[0]
    ending = ('\n- **Overall change:** Starts and ends at approximately the same value.' if np.isclose(change,0) else
              f'\n- **Overall change:** {"Up" if change>0 else "Down"} {abs(change):.3g} units from the first value.')
    return prefix+extrema+ending+note
