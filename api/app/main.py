import os,tempfile,csv,io
from pathlib import Path
from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image
import httpx

class S(BaseSettings):
    cors_origins:str='http://localhost:3000'; fire_model_repo:str='rabahdev/fire-smoke-yolov8n'; fire_model_file:str='best.pt'; fire_confidence:float=.25; nasa_firms_map_key:str=''; nasa_firms_source:str='VIIRS_NOAA21_NRT'
    class Config: env_file='.env'; extra='ignore'
s=S(); model=None
app=FastAPI(title='FireWatch AI API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in s.cors_origins.split(',')],allow_methods=['*'],allow_headers=['*'])

def get_model():
    global model
    if model is None:
        p=hf_hub_download(repo_id=s.fire_model_repo,filename=s.fire_model_file,local_dir='models')
        model=YOLO(p)
    return model

def detect(path):
    im=Image.open(path).convert('RGB'); m=get_model(); r=m.predict(source=im,conf=s.fire_confidence,verbose=False)[0]; ds=[]
    for b in r.boxes:
        c=float(b.conf[0]); k=int(b.cls[0]); ds.append({'label':str(m.names[k]),'confidence':c,'box':[float(x) for x in b.xyxy[0].tolist()]})
    fire=any('fire' in d['label'].lower() or 'flame' in d['label'].lower() for d in ds); mc=max([d['confidence'] for d in ds],default=0); sev='CRITICAL' if fire and mc>=.8 else 'HIGH' if fire else 'WATCH' if ds else 'CLEAR'
    return {'summary':{'fire_detected':fire,'detection_count':len(ds),'max_confidence':mc,'severity':sev},'detections':ds,'image_width':im.width,'image_height':im.height,'model':s.fire_model_repo}

@app.get('/health')
def health(): return {'status':'ok','model':s.fire_model_repo}
@app.post('/api/v1/detect/image')
async def image(file:UploadFile=File(...)):
    if not (file.content_type or '').startswith('image/'): raise HTTPException(415,'Expected image')
    data=await file.read();
    if len(data)>40*1024*1024: raise HTTPException(413,'File too large')
    fd,p=tempfile.mkstemp(suffix=Path(file.filename or '.jpg').suffix or '.jpg');os.close(fd);Path(p).write_bytes(data)
    try:return detect(p)
    finally:Path(p).unlink(missing_ok=True)

@app.get('/api/v1/weather')
async def weather(latitude:float=Query(...,ge=-90,le=90),longitude:float=Query(...,ge=-180,le=180)):
    params={'latitude':latitude,'longitude':longitude,'current':'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation','timezone':'auto','forecast_days':2}
    async with httpx.AsyncClient(timeout=15) as c:r=await c.get('https://api.open-meteo.com/v1/forecast',params=params);r.raise_for_status()
    return r.json()

@app.get('/api/v1/fires')
async def fires():
    if not s.nasa_firms_map_key:return {'configured':False,'fires':[],'message':'NASA FIRMS MAP_KEY is not configured; no synthetic satellite observations are returned.'}
    u=f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{s.nasa_firms_map_key}/{s.nasa_firms_source}/world/1'
    async with httpx.AsyncClient(timeout=30) as c:r=await c.get(u);r.raise_for_status()
    out=[]
    for x in csv.DictReader(io.StringIO(r.text)):
        try:out.append({'latitude':float(x['latitude']),'longitude':float(x['longitude']),'confidence':x.get('confidence'),'frp':float(x['frp']) if x.get('frp') else None,'acq_date':x.get('acq_date'),'satellite':x.get('satellite'),'label':'NASA FIRMS active fire'})
        except:pass
    return {'configured':True,'source':s.nasa_firms_source,'fires':out}

@app.get('/api/v1/risk')
async def risk(latitude:float,longitude:float):
    params={'latitude':latitude,'longitude':longitude,'current':'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation','timezone':'auto'}
    async with httpx.AsyncClient(timeout=15) as c:r=await c.get('https://api.open-meteo.com/v1/forecast',params=params);r.raise_for_status()
    w=r.json().get('current',{});temp=float(w.get('temperature_2m') or 20);hum=float(w.get('relative_humidity_2m') or 60);wind=float(w.get('wind_speed_10m') or 5);prec=float(w.get('precipitation') or 0);score=max(0,min(100,int(20+max(0,min(28,(temp-20)*1.2))+max(0,min(25,(60-hum)*.6))+max(0,min(22,wind*.8))-min(20,prec*8))))
    level='EXTREME' if score>=80 else 'HIGH' if score>=60 else 'MODERATE' if score>=35 else 'LOW';drivers=[]
    if temp>=32:drivers.append('Elevated temperature')
    if hum<=35:drivers.append('Low relative humidity')
    if wind>=25:drivers.append('Strong surface wind')
    if prec>.5:drivers.append('Recent precipitation')
    return {'score':score,'level':level,'drivers':drivers or ['No dominant weather risk driver'],'note':'Explainable screening model using live weather; not a certified fire-behavior forecast.','weather':w}
