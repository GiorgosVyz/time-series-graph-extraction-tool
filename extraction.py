"""Curve extraction for framed line charts and simple open price/area charts."""
import cv2
import numpy as np
import pandas as pd


def suggest_curve_color(image, bounds):
    """Favor a hue spanning the plot width, then its saturated stroke over pale fill."""
    left, top, right, bottom = bounds
    crop = image[top:bottom+1, left:right+1, :3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    vivid = (hsv[:,:,1] > 90) & (hsv[:,:,2] > 40)
    best = None
    for hue in range(0,180,10):
        delta = abs(hsv[:,:,0].astype(float)-hue)
        match = vivid & (np.minimum(delta,180-delta) <= 7)
        coverage = np.any(match,axis=0).sum()
        if coverage and (best is None or coverage > best[0]):
            best = (coverage,match)
    if best is None or best[0] < max(10,.15*crop.shape[1]):
        raise ValueError('No clear colored curve found. Choose its color manually.')
    selected = best[1] & (hsv[:,:,1] >= np.quantile(hsv[:,:,1][best[1]],.8))
    return tuple(int(v) for v in np.median(crop[selected],axis=0))


def _color_mask(crop,color,distance):
    error = np.linalg.norm(crop.astype(float)-np.asarray(color),axis=2)
    hsv = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    saturation = cv2.cvtColor(np.array([[color]],dtype=np.uint8),cv2.COLOR_RGB2HSV)[0,0,1]
    return (error < distance) & (hsv[:,:,1] >= max(50,float(saturation)*.75)), error


def suggest_observation_count(image, bounds, color):
    """Count clearly separated, regularly spaced filled markers, never line bends."""
    left,top,right,bottom = map(int,bounds)
    if not (0 <= left < right < image.shape[1] and 0 <= top < bottom < image.shape[0]):
        return None
    crop = image[top:bottom+1,left:right+1,:3]
    mask,_ = _color_mask(crop,color,60)
    if np.mean(np.any(mask,axis=0)) < .85:
        return None
    radii = cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
    stroke_radius = float(np.median(radii.max(axis=0)))
    # Marker centers must be substantially thicker than the connecting stroke.
    centers = (radii >= max(2.8,1.8*stroke_radius)).astype('uint8')
    # At a border only half a marker is visible, so its remaining core is smaller.
    centers[:,:4] = radii[:,:4] >= 2.8
    centers[:,-4:] = radii[:,-4:] >= 2.8
    _,_,stats,centroids = cv2.connectedComponentsWithStats(centers,8)
    if not 5 <= len(stats)-1 <= 300:
        return None
    sizes = []
    width = crop.shape[1]
    positions = centroids[1:,0].copy()
    for i,(x,y,w,h,area) in enumerate(stats[1:]):
        # A plot border can cover the outermost pixels of an endpoint marker.
        edge = x <= 3 or x+w >= width-3
        if h < (2 if edge else 3) or w < (1 if edge else 3) or area < (2 if edge else 5):
            return None
        if not edge and not .65 <= w/h <= 1.5:
            return None
        if edge:
            positions[i] = 0 if x <= 3 else width-1
        else:
            sizes.append(max(w,h))
    if not sizes or max(sizes) > 1.5*min(sizes):
        return None
    positions.sort()
    spacing = float(np.median(np.diff(positions)))
    if spacing < 2.5*max(sizes) or not np.allclose(np.diff(positions),spacing,rtol=.05,atol=1.5):
        return None
    if positions[0] > max(3,.2*spacing) or width-1-positions[-1] > max(3,.2*spacing):
        return None
    return len(positions)


def _leading_extension(crop, first_x, first_y):
    """Only accept another color if it physically meets the main stroke at its edge."""
    hsv = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    region = (hsv[:,:,1]>100)&(hsv[:,:,2]>50)
    yy,xx = np.indices(region.shape)
    region &= (xx >= first_x-.035*crop.shape[1]) & (xx <= first_x+3)
    region &= abs(yy-first_y) < max(12,.15*crop.shape[0])
    _,labels = cv2.connectedComponents(cv2.morphologyEx(region.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),8)
    anchor = labels[max(0,int(first_y)-3):min(crop.shape[0],int(first_y)+4),first_x:first_x+4]
    components = np.unique(anchor[anchor>0])
    y,x = np.where(np.isin(labels,components)&region&(xx<first_x))
    return x,y


def _open_plot_area(image):
    height,width = image.shape[:2]
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,5,20)
    joined = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((1,9),np.uint8))
    horizontal = cv2.morphologyEx(joined,cv2.MORPH_OPEN,np.ones((1,max(31,width//4)|1),np.uint8))
    _,_,stats,_ = cv2.connectedComponentsWithStats(horizontal,8)
    segments = [s for s in stats[1:] if s[2] >= .5*width and s[3] <= 4]
    if segments:
        right_edge = float(np.median([s[0]+s[2] for s in segments]))
        segments = [s for s in segments if abs(s[0]+s[2]-right_edge) <= .12*width]
    rows = []
    for s in sorted(segments,key=lambda s:s[1]):
        y = float(s[1]+(s[3]-1)/2)
        if rows and y-rows[-1] <= 3:
            rows[-1] = (rows[-1]+y)/2
        else:
            rows.append(y)
    best = None
    rows = np.asarray(rows)
    for anchor in rows:
        for other in rows:
            for multiple in range(1,5):
                spacing = abs(other-anchor)/multiple
                if spacing < max(8,.025*height) or spacing > .3*height:
                    continue
                slots = np.round((rows-anchor)/spacing)
                keep = abs(rows-(anchor+slots*spacing)) <= 2
                selected = rows[keep]
                if len(selected) < 3 or np.ptp(selected) < .25*height:
                    continue
                occupancy = len(selected)/(np.ptp(slots[keep])+1)
                if occupancy < .65:
                    continue
                rank = (len(selected),float(np.ptp(selected)),occupancy)
                if best is None or rank > best[0]:
                    best = (rank,selected)
    if best is None:
        raise ValueError('Could not identify plot borders or a consistent grid. Adjust the rectangle manually.')
    top,bottom = float(best[1][0]),float(best[1][-1])
    spacing = float(np.median(np.diff(best[1])))
    next_row = int(round(bottom+spacing))
    if next_row+2 < height and np.any(horizontal[max(0,next_row-2):next_row+3],axis=0).sum() >= .5*width:
        bottom = float(next_row)
    span = bottom-top
    search = (0,max(0,int(top-.08*span)),width-1,min(height-1,int(bottom+.12*span)))
    color = suggest_curve_color(image,search)
    crop = image[search[1]:search[3]+1,:,:3]
    mask,_ = _color_mask(crop,color,45)
    yy,xx = np.where(mask)
    if not len(xx) or np.ptp(xx) < .5*width:
        raise ValueError('No continuous price curve found inside the grid. Adjust the rectangle manually.')
    left,right = int(xx.min()),int(xx.max())
    # A short differently colored edge segment may be part of the same price curve.
    first_y = np.median(yy[xx <= left+3])
    extension_x,extension_y = _leading_extension(crop,left,first_y)
    if len(extension_x):
        left = int(extension_x.min())
        yy = np.r_[yy,extension_y]
    top = min(top,float(yy.min()+search[1]))
    bottom = max(bottom,float(yy.max()+search[1]))
    return left,max(0,int(np.floor(top))),right,min(height-1,int(np.ceil(bottom)))


def detect_plot_area(image):
    """Prefer four supported borders; otherwise seek a coherent open grid and curve."""
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
        return _open_plot_area(image)
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
    mask,error = _color_mask(crop,color,distance)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    if n < 2:
        raise ValueError('No curve detected. Adjust color or rectangle.')
    component = 1+np.argmax(stats[1:,cv2.CC_STAT_WIDTH])
    seed = labels == component
    seed_columns = np.any(seed,axis=0)
    # Retain short detached portions near markers without admitting interior labels.
    mask[:,seed_columns] = seed[:,seed_columns]
    columns = np.flatnonzero(np.any(mask,axis=0))
    width = right-left+1
    gaps = np.diff(columns)-1
    if (len(columns)<.85*width or columns[0]>.035*width or columns[-1]<width-1-.035*width
            or (len(gaps) and gaps.max()>.04*width)):
        raise ValueError('Curve must cover the selected time range. Adjust rectangle or color.')
    rows = []
    for x in columns:
        candidates = np.flatnonzero(mask[:,x])
        closest = error[candidates,x].min()
        rows.append(float(np.median(candidates[error[candidates,x] <= closest+6])))
    # Complete only a short leading edge that changes color close to the main stroke.
    if columns[0] > 0:
        candidates_x,candidates_y = _leading_extension(crop,int(columns[0]),rows[0])
        extra_x,extra_y = [],[]
        for x in range(int(columns[0])):
            candidates = candidates_y[candidates_x==x]
            if len(candidates):
                extra_x.append(x)
                extra_y.append(float(candidates[np.argmin(abs(candidates-rows[0]))]))
        columns = np.r_[extra_x,columns]
        rows = extra_y+rows
    px = np.linspace(0,right-left,count)
    py = np.interp(px,columns,rows)+top
    result = pd.DataFrame({'time':np.linspace(xmin,xmax,count), 'value':ymax-(py-top)/(bottom-top)*(ymax-ymin), 'pixel_x':px+left,'pixel_y':py})
    result.attrs['curve_coverage'] = len(columns)/width
    return result
