"""Local OCR calibration for clean, linear numeric axes."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import re
import cv2
import numpy as np

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
    text = text.strip().replace('−', '-').replace('–', '-').replace(',', '.')
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
        if abs(p[nearest]-endpoint) <= .025*(end-start):
            values[i] = v[nearest]
        elif step > 0:
            snapped = v[0]+round((values[i]-v[0])/step)*step
            if abs(snapped-values[i]) < .1*step:
                values[i] = snapped
    return [float(round(n, 6)) for n in values], v


def axes_from_labels(labels, bounds):
    left, top, right, bottom = bounds
    width, height = right-left, bottom-top
    xticks, yticks = [], []
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
    result = {'axes': [None]*4, 'count': None, 'notes': []}
    try:
        result['axes'][:2], xvalues = fit_axis(xticks, left, right)
        xmin, xmax = result['axes'][:2]
        # A suggestion only for densely labelled integer sample indices.
        count = round(xmax-xmin)+1
        if (8 <= count <= 300 and len(xvalues) >= .8*count
                and np.allclose(xvalues, np.round(xvalues))
                and np.isclose(xmin, round(xmin)) and np.isclose(xmax, round(xmax))
                and np.mean(np.isclose(np.diff(np.sort(xvalues)), 1)) >= .8):
            result['count'] = count
    except ValueError as exc:
        result['notes'].append('Time axis: '+str(exc))
    try:
        ends, _ = fit_axis(yticks, top, bottom, increasing=False)
        result['axes'][2:] = ends[::-1]
    except ValueError as exc:
        result['notes'].append('Value axis: '+str(exc))
    return result


def read_calibration(image, bounds):
    # Upscaling improves small tick labels; OCR boxes are mapped back to source pixels.
    scale = min(2., 2400/max(image.shape[:2]))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with _LOCK:
        output = _engine()(cv2.resize(bgr, None, fx=scale, fy=scale), return_word_box=True)
    labels = []
    for line in (output.word_results or ()):
        for text, score, box in line:
            if box is not None:
                labels.append((text, score, np.asarray(box)/scale))
    if not labels and output.txts is not None:
        labels = [(text, score, box/scale) for text, score, box in zip(output.txts, output.scores, output.boxes)]
    return axes_from_labels(labels, bounds)
