from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

root=Path(__file__).parent/'sample_images'
root.mkdir(exist_ok=True)
x=np.arange(60)
series={'trend':20+0.8*x+3*np.sin(x/4),'wave':50+22*np.sin(x/7),'shift':np.where(x<30,30,65)+4*np.cos(x/3)}
for name,y in series.items():
    im=Image.new('RGB',(720,390),'white')
    d=ImageDraw.Draw(im)
    for v in range(0,101,20):
        py=330-3*v
        d.line((60,py,660,py),fill='#dddddd')
        d.text((25,py-5),str(v),fill='black')
    d.rectangle((60,30,660,330),outline='black')
    for t in range(0,60,10):
        d.text((60+t/59*600,342),str(t),fill='black')
    points=[(60+t/59*600,330-v*3) for t,v in zip(x,y)]
    d.line(points,fill=(35,110,210),width=3)
    im.save(root/f'{name}.png')
    pd.DataFrame({'time':x,'value':y}).to_csv(root/f'{name}.csv',index=False)
