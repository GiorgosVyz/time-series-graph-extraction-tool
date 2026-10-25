"""Temporal axis recognition and display without invented calendar years."""
import re
from datetime import datetime, timedelta
import numpy as np

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
KINDS = ['Numeric', 'Clock time', 'Weekdays', 'Months', 'Dates']
KEYS = {'numeric':'Numeric', 'clock':'Clock time', 'weekday':'Weekdays', 'month':'Months', 'date':'Dates'}
REVERSE_KEYS = {v:k for k,v in KEYS.items()}


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
        formats = ['%Y-%m-%d','%Y-%m-%d %H:%M','%Y/%m/%d','%d %b %Y','%d %B %Y','%b %d %Y','%B %d %Y']
        for fmt in formats:
            try:
                if not re.search(r'\b\d{4}\b', text):
                    continue
                return float(datetime.strptime(text,fmt).toordinal())
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
        p = np.array([t[0] for t,v in zip(ticks,values) if any(np.isclose(v,a) for a in accepted)])
        if (len(accepted) >= 5 and p[0]-left < .035*width and right-p[-1] < .035*width
                and np.allclose(np.diff(accepted),np.median(np.diff(accepted)),rtol=.02,atol=1e-5)):
            count = len(accepted)
        candidates.append((len(accepted), {'kind':kind,'start':endpoints[0],'end':endpoints[1],
                                           'labels':[t[2] for t in ticks], 'count':count}))
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
            label += f' +{fraction:.2g} {"day" if kind == "weekday" else "month"}'
    if cycles:
        label += f' ({cycles:+d} {unit}{"s" if abs(cycles)!=1 else ""})'
    return label


def parse_bounds(kind, start, end):
    values = []
    period = {'clock':1440,'weekday':7,'month':12}.get(kind)
    for text in (start,end):
        match = re.fullmatch(r'(.*?)\s*\(([+-]?\d+) (day|week|year)s?\)',text.strip())
        base = match.group(1) if match else text
        value = parse_label(base,kind)
        if value is None:
            raise ValueError('Use valid time labels, such as 12:20, Mon, Jan, or 2026-10-25.')
        if match:
            if period is None:
                raise ValueError('Cycle offsets apply only to clock times, weekdays and months.')
            if match.group(3) != {'clock':'day','weekday':'week','month':'year'}[kind]:
                raise ValueError('Use day offsets for clock times, week offsets for weekdays, and year offsets for months.')
            value += int(match.group(2))*period
        values.append(float(value))
    if values[1] < values[0] and period and '(' not in end:
        values[1] += period
    if values[1] <= values[0]:
        raise ValueError('End time must follow start time. Use a cycle offset for repeated labels.')
    return tuple(values)


def apply_time_axis(figure, axis, positions):
    if axis['kind'] == 'numeric':
        return
    positions = np.asarray(positions,dtype=float)
    indices = np.unique(np.linspace(0,len(positions)-1,min(8,len(positions))).round().astype(int))
    labels = [format_time(axis['kind'],p,axis['start']) for p in positions[indices]]
    figure.update_xaxes(tickmode='array',tickvals=positions[indices],ticktext=labels,
                        title=KEYS[axis['kind']],tickangle=-25 if axis['kind']=='date' else 0)


def export_series(frame, axis):
    result = frame[['time', 'value' if 'value' in frame else 'forecast']].copy()
    if axis['kind'] != 'numeric':
        result['time'] = [format_time(axis['kind'],v,axis['start']) for v in result.time]
    return result


def series_overview(times, values, axis):
    times,values = np.asarray(times),np.asarray(values)
    labels = [format_time(axis['kind'],t,axis['start']) for t in times]
    step = (times[-1]-times[0])/(len(times)-1)
    title = {'numeric':'Numeric-axis','clock':'Time-of-day','weekday':'Weekday-based','month':'Month-based','date':'Date-based'}[axis['kind']]
    if axis['kind'] == 'clock' and np.isclose(step,60): title='Hourly'
    if axis['kind'] in ('weekday','date') and np.isclose(step,1): title='Daily'
    if axis['kind'] == 'month' and np.isclose(step,1): title='Monthly'
    prefix = f'{title} series with {len(values)} observations, from {labels[0]} to {labels[-1]}.'
    peak, low = int(np.argmax(values)), int(np.argmin(values))
    if np.ptp(values) < 1e-9*max(1,float(np.max(abs(values)))):
        return prefix+f' The extracted values remain approximately constant at {values[0]:.3g}.'
    extrema = (f' Estimated maximum: {values[peak]:.3g} at {labels[peak]}; '
               f'estimated minimum: {values[low]:.3g} at {labels[low]} (first occurrences).')
    change = values[-1]-values[0]
    ending = (' The first and last values are approximately equal.' if np.isclose(change,0) else
              f' The final value is {abs(change):.3g} units {"higher" if change>0 else "lower"} than the first.')
    return prefix+extrema+ending
