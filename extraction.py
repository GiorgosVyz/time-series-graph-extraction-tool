import cv2
import numpy as np
import pandas as pd


def suggest_curve_color(image, bounds):
    """Suggest the most common saturated color within the plot, excluding gray axes."""
    left, top, right, bottom = bounds
    crop = image[top:bottom+1, left:right+1, :3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    pixels = crop[(hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 50)]
    if len(pixels) < 20:
        raise ValueError('No clear colored curve found. Choose its color manually.')
    bins = pixels // 16
    colors, counts = np.unique(bins, axis=0, return_counts=True)
    selected = pixels[np.all(bins == colors[np.argmax(counts)], axis=1)]
    return tuple(int(v) for v in np.median(selected, axis=0))


def detect_plot_area(image):
    """Suggest a framed plot/grid rectangle; reject charts without enough evidence.

    Long dark horizontal and vertical lines must form all four borders.
    Axis values cannot be inferred from these pixel coordinates.
    """
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    dark = (gray < 210).astype('uint8') * 255
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                                 np.ones((1, max(15, width // 4) | 1), np.uint8))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                               np.ones((max(15, height // 4) | 1, 1), np.uint8))
    lines = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        left, top, w, h = cv2.boundingRect(contour)
        if w < width * .3 or h < height * .25:
            continue
        right, bottom = left+w-1, top+h-1
        # Require continuous support along each border, allowing thick strokes.
        border_scores = [
            np.mean(np.any(horizontal[max(0, y-2):min(height, y+3), left:right+1], axis=0))
            for y in (top, bottom)
        ] + [
            np.mean(np.any(vertical[top:bottom+1, max(0, x-2):min(width, x+3)], axis=1))
            for x in (left, right)
        ]
        if min(border_scores) >= .8:
            candidates.append((w*h, (left, top, right, bottom)))
    if not candidates:
        raise ValueError('Could not identify four plot borders. Enter the rectangle manually.')
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[1][0] > .7*candidates[0][0]:
        raise ValueError('Multiple possible plot areas detected. Enter the rectangle manually.')
    return candidates[0][1]


def extract(image, bounds, axes, count, color, distance=90):
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('Use an RGB chart image.')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError('Observation count must be an integer.')
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('Color tolerance must be positive and finite.')
    left, top, right, bottom = map(int, bounds)
    xmin, xmax, ymin, ymax = axes
    if not (0 <= left < right < image.shape[1] and 0 <= top < bottom < image.shape[0]):
        raise ValueError('Invalid plot rectangle.')
    if not np.isfinite(axes).all() or xmax <= xmin or ymax <= ymin or count < 5:
        raise ValueError('Use increasing finite axis limits and at least five observations.')
    crop = image[top:bottom+1,left:right+1,:3]
    mask = (np.linalg.norm(crop.astype(float)-np.array(color),axis=2)<distance).astype('uint8')
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    if n < 2:
        raise ValueError('No curve detected. Adjust color or rectangle.')
    component = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    yy,xx = np.where(labels==component)
    columns = np.unique(xx)
    if len(columns)<0.8*(right-left+1) or columns[0]>2 or columns[-1]<right-left-2:
        raise ValueError('Curve must cover the selected time range. Adjust rectangle or color.')
    rows = [np.median(yy[xx==x]) for x in columns]
    px = np.linspace(0,right-left,count)
    py = np.interp(px,columns,rows)+top
    return pd.DataFrame({'time':np.linspace(xmin,xmax,count), 'value':ymax-(py-top)/(bottom-top)*(ymax-ymin), 'pixel_x':px+left,'pixel_y':py})
