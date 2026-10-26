"""Local OCR calibration for clean, linear numeric axes."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import re
import cv2
import numpy as np
from time_axis import detect_time_axis, parse_label

_LOCK = Lock()


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={
        'Global.model_root_dir': str(Path(__file__).parent/'work'/'ocr_models'),
        'Global.use_cls': False,
        'Global.log_level': 'critical',
        'EngineConfig.onnxruntime.intra_op_num_threads': 2,
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
    })


def numeric_label(text):
    text = text.strip().replace('−', '-').replace('–', '-')
    if ',' in text and '.' in text:
        if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',text):
            text = text.replace(',','')
        elif re.fullmatch(r'[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?',text):
            text = text.replace('.','').replace(',','.')
        else:
            return None
    elif re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+',text):
        text = text.replace(',','')
    else:
        text = text.replace(',','.')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
        return None
    number = float(text)
    return number if np.isfinite(number) else None


def fit_axis(ticks, start, end, increasing=True):
    """Fit pixel-to-value using consensus; do not accept inconsistent/log axes."""
    if len(ticks) < 3:
        raise ValueError('Not enough numeric tick labels were read.')
    ticks = np.array(sorted(ticks), dtype=float)
    # Duplicated labels can occur if OCR word and line boxes overlap.
    _, indices = np.unique(np.round(ticks[:, 0], 1), return_index=True)
    ticks = ticks[np.sort(indices)]
    p, v = ticks.T
    best = None
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if p[j]-p[i] < (end-start)*.3:
                continue
            slope = (v[j]-v[i])/(p[j]-p[i])
            if slope == 0 or (slope > 0) != increasing:
                continue
            intercept = v[i]-slope*p[i]
            residual = np.abs((v-(slope*p+intercept))/slope)
            good = residual <= max(3., (end-start)*.012)
            rank = (int(good.sum()), -float(residual[good].sum()))
            if best is None or rank > best[0]:
                best = (rank, good)
    if best is None or best[0][0] < max(3, int(np.ceil(.75*len(ticks)))):
        raise ValueError('Tick labels do not form a consistent linear scale.')
    p, v = ticks[best[1]].T
    if np.ptp(p) < .5*(end-start):
        raise ValueError('Recognized labels cover too little of the axis.')
    slope, intercept = np.polyfit(p, v, 1)
    values = np.array([slope*start+intercept, slope*end+intercept])
    # Snap only to the observed tick lattice, allowing small OCR box offsets.
    step = float(np.median(np.diff(np.sort(v))))
    for i, endpoint in enumerate((start, end)):
        nearest = np.argmin(abs(p-endpoint))
        snap_distance = min(.025*(end-start),max(2.,.08*float(np.median(np.diff(p)))))
        if abs(p[nearest]-endpoint) <= snap_distance:
            values[i] = v[nearest]
        elif step > 0:
            snapped = v[0]+round((values[i]-v[0])/step)*step
            if abs(snapped-values[i]) < .1*step:
                values[i] = snapped
    return [float(round(n, 6)) for n in values], v


def axes_from_labels(labels, bounds):
    left, top, right, bottom = bounds
    width, height = right-left, bottom-top
    xticks, yticks, right_ticks = [], [], []
    for text, score, box in labels:
        value = numeric_label(text)
        if value is None or score < .7:
            continue
        box = np.asarray(box)
        x, y = np.mean(box, axis=0)
        if left-.03*width <= x <= right+.03*width and bottom < y < bottom+max(35,.12*height):
            xticks.append((x, value))
        if left-max(85,.18*width) < x < left-3 and top-.03*height <= y <= bottom+.03*height:
            yticks.append((y, value))
        if right+3 < x < right+max(130,.25*width) and top-.03*height <= y <= bottom+.03*height:
            right_ticks.append((y,value))
    result = {'axes': [None]*4, 'count': None, 'notes': []}
    try:
        result['axes'][:2], xvalues = fit_axis(xticks, left, right)
        xmin, xmax = result['axes'][:2]
        # A suggestion only for densely labelled integer sample indices.
        count = round(xmax-xmin)+1
        if (5 <= count <= 300 and len(xvalues) >= .8*count
                and np.allclose(xvalues, np.round(xvalues))
                and np.isclose(xmin, round(xmin)) and np.isclose(xmax, round(xmax))
                and np.mean(np.isclose(np.diff(np.sort(xvalues)), 1)) >= .8):
            result['count'] = count
    except ValueError as exc:
        result['notes'].append('Time axis: '+str(exc))
    scales = []
    for side,ticks in [('left',yticks),('right',right_ticks)]:
        try:
            ends, accepted = fit_axis(ticks, top, bottom, increasing=False)
            scales.append((side,ends,len(accepted)))
        except ValueError:
            pass
    if len(scales)==2 and not np.allclose(scales[0][1],scales[1][1],rtol=.01,atol=.01):
        result['notes'].append('Value axis: two different scales detected; enter the intended scale manually.')
    elif scales:
        side,ends,_ = max(scales,key=lambda item:item[2])
        result['axes'][2:] = ends[::-1]
        result['value_axis_side'] = side
    else:
        result['notes'].append('Value axis: not enough consistent tick labels were read on either side.')
    temporal = detect_time_axis(labels, bounds, fit_axis)
    if temporal:
        result['notes'] = [note for note in result['notes'] if not note.startswith('Time axis:')]
        result['time_axis'] = temporal
        result['axes'][:2] = [temporal['start'], temporal['end']]
        result['count'] = temporal['count']
    else:
        result['time_axis'] = {'kind':'numeric', 'start':result['axes'][0], 'end':result['axes'][1]}
    return result


def read_calibration(image, bounds):
    # Upscaling improves small tick labels; OCR boxes are mapped back to source pixels.
    scale = min(2., 2400/max(image.shape[:2]))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with _LOCK:
        output = _engine()(cv2.resize(bgr, None, fx=scale, fy=scale), return_word_box=True)
    full_lines = [] if output.txts is None else [
        (text,score,np.asarray(box)/scale) for text,score,box in zip(output.txts,output.scores,output.boxes)]
    # Keep complete temporal labels: dropping PM turns 1:00 PM into 01:00.
    temporal_lines = [line for line in full_lines
                      if any(parse_label(line[0],kind) is not None for kind in ('date','clock','month_day'))]
    temporal_boxes = [(box.min(axis=0),box.max(axis=0)) for _,_,box in temporal_lines]
    labels = []
    for line in (output.word_results or ()):
        for text, score, box in line:
            if box is not None:
                word_box = np.asarray(box)/scale
                center = word_box.mean(axis=0)
                if any(np.all(center >= low-1) and np.all(center <= high+1) for low,high in temporal_boxes):
                    continue
                labels.append((text, score, word_box))
    labels = labels+temporal_lines if labels else full_lines
    return axes_from_labels(labels, bounds)
