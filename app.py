from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from extraction import extract
from forecasting import predict, backtest, summarize

st.set_page_config(page_title='Time Series Digitizer & Forecaster', layout='wide')
st.title('Time Series Digitizer & Forecaster')
st.caption('Recover a colored curve, calibrate its axes, and measure historical forecast reliability.')
source = st.selectbox('Image source', ['Upload','trend','wave','shift'])
upload = st.file_uploader('PNG or JPG',type=['png','jpg','jpeg']) if source=='Upload' else None
path = upload if source=='Upload' else Path(__file__).parent/'sample_images'/f'{source}.png'
if not path:
    st.info('Upload a clean single-series chart, or select a sample above.')
    st.stop()
try:
    im = np.array(Image.open(path).convert('RGB'))
    height,width = im.shape[:2]
    st.image(im,caption=f'Image size: {width} × {height} pixels')
    st.subheader('Calibrate the plot')
    st.caption('Enter the pixel coordinates of the plotting rectangle, excluding labels. Pixel origin is the top left. Samples use left=60, top=30, right=660, bottom=330; axes 0–59 and 0–100; 60 observations.')
    cols=st.columns(4)
    bounds=[cols[i].number_input(label,0, int(width-1 if i%2==0 else height-1),value=min(default,int(width-1 if i%2==0 else height-1))) for i,(label,default) in enumerate(zip(['Left','Top','Right','Bottom'],[60,30,660,330]))]
    cols=st.columns(4)
    axes=[cols[i].number_input(label,value=float(default)) for i,(label,default) in enumerate(zip(['Time minimum','Time maximum','Value minimum','Value maximum'],[0,59,0,100]))]
    count=st.number_input('Number of equally spaced original observations',8,300,60)
    color=st.color_picker('Curve color','#236ed2')
    distance=st.slider('Color tolerance',10,200,90)
    rgb=tuple(int(color[i:i+2],16) for i in (1,3,5))
    df=extract(im,bounds,axes,count,rgb,distance)
    overlay=go.Figure(go.Image(z=im))
    overlay.add_trace(go.Scatter(x=df.pixel_x,y=df.pixel_y,mode='markers',marker=dict(size=4,color='red'),name='Extracted points'))
    st.plotly_chart(overlay,use_container_width=True)
    st.download_button('Download extracted CSV',df[['time','value']].to_csv(index=False),'extracted.csv','text/csv')
    st.subheader('Forecast and rolling validation')
    horizon=st.slider('Forecast horizon (observations)',1,min(20,count//3),5)
    initial=st.slider('Initial training observations',4,count-horizon,count//2)
    tolerance=st.number_input('Failure tolerance (value units)',min_value=0.0,value=5.0)
    model=st.selectbox('Forecast model',['Holt','Naive'])
    y=df.value.to_numpy()
    results=backtest(y,horizon,initial,model)
    baseline=backtest(y,horizon,initial,'Naive')
    st.caption('Each row below measures one lead time across rolling origins. Failure means absolute error strictly exceeds the tolerance.')
    st.dataframe(results.groupby('horizon').error.apply(lambda e: __import__('pandas').Series(summarize(e,tolerance))).unstack())
    st.write('Naive baseline (all lead times)',summarize(baseline.error,tolerance))
    future=predict(y,horizon,model)
    times=df.time.iloc[-1]+np.arange(1,horizon+1)*(axes[1]-axes[0])/(count-1)
    fig=go.Figure(go.Scatter(x=df.time,y=y,name='Extracted history'))
    fig.add_trace(go.Scatter(x=times,y=future,name=model+' forecast'))
    st.plotly_chart(fig,use_container_width=True)
    st.bar_chart(results.groupby('horizon').error.mean())
    st.download_button('Download backtest CSV',results.to_csv(index=False),'backtest.csv','text/csv')
    st.warning('Historical failure rates are estimates, not exact future probabilities. Overlapping folds are dependent. Regime changes and digitization errors may invalidate historical reliability. Choosing more samples than the original chart contains creates artificial observations.')
except (ValueError,OSError) as exc:
    st.error(str(exc))
