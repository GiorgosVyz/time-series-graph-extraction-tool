import cv2
import numpy as np
import pandas as pd


def extract(image, bounds, axes, count, color, distance=90):
    left, top, right, bottom = map(int, bounds)
    xmin, xmax, ymin, ymax = axes
    if not (0 <= left < right < image.shape[1] and 0 <= top < bottom < image.shape[0]):
        raise ValueError('Invalid plot rectangle.')
    if not np.isfinite(axes).all() or xmax <= xmin or ymax <= ymin or count < 8:
        raise ValueError('Use increasing finite axis limits and at least eight observations.')
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
