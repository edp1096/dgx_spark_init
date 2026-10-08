"""HTTP bridge to the qualified single-process QWIM/MiniMax H3 worker."""
import asyncio
import base64
import json
import os
from pathlib import Path
import secrets
import re
from generation_progress import append_event, read_events, prune
import signal
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(os.getenv('JOB_DIR', '/job'))
app = FastAPI()
lock = asyncio.Lock()
active = 0
queued = 0
quiescing = False
last_used = time.monotonic()

class Generate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1, max_length=8192)
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**63), ge=0, lt=2**63)
    model: str | None = None
    size: str = '1024x1024'
    response_format: str = 'b64_json'
    n: int = 1
    output_format: str = 'png'


def clean_job(case):
    for folder,suffix in [('output','.mp4'),('output','.png'),('results','.json'),('requests','.running')]:
        (ROOT/folder/(case+suffix)).unlink(missing_ok=True)

def ready():
    return (ROOT / 'ready.json').exists() and not (ROOT / 'stop').exists()

@app.get('/health')
def health():
    if not ready(): raise HTTPException(503, 'Models are loading')
    return {'status': 'ok', 'features':{'progress':True,'eta':True}, 'image_model': 'qwen-image-2.1-uc-nvfp4', 'video_model': 'minimax-h3-nvfp4', 'video': {'width':864,'height':480,'frames':124,'fps':24,'audio':True}}

@app.get('/v1/runtime/memory')
def memory():
    return {'status':'ok','busy':bool(active or queued),'active':active,'queued':queued,'quiescing':quiescing,'idle_for_seconds':time.monotonic()-last_used,'core_ready':ready(),'memory_gib':28,'workspace_gib':6,'keep_models_loaded':True}

@app.get('/v1/runtime/progress/{request_id}')
def progress(request_id: str, after: int = 0):
    if not re.fullmatch(r'[a-f0-9]{32}', request_id) or after < 0:
        raise HTTPException(400, 'Invalid progress cursor or request ID')
    return read_events(ROOT, request_id, after)

@app.post('/v1/runtime/quiesce')
def quiesce():
    global quiescing
    if active or queued: raise HTTPException(409,'Generation is active or queued')
    quiescing=True
    return {'status':'ok'}

@app.post('/v1/runtime/resume')
def resume():
    global quiescing
    quiescing=False
    return {'status':'ok'}

@app.post('/v1/runtime/prepare')
def prepare():
    if not ready(): raise HTTPException(503,'Models are loading')
    return {'status':'ok'}

@app.post('/v1/runtime/cancel')
def cancel():
    (ROOT/'stop').touch()
    pid=ROOT/'worker.pid'
    if pid.exists():
        try: os.kill(int(pid.read_text()),signal.SIGTERM)
        except ProcessLookupError: pass
    return {'status':'ok'}

async def generate(body, kind, request):
    global active,queued,last_used
    if quiescing or not ready(): raise HTTPException(503,'Generation service is unavailable')
    if not body.prompt.strip(): raise HTTPException(400,'prompt cannot be blank')
    if kind=='qwim' and (body.size!='1024x1024' or body.n!=1 or body.response_format!='b64_json' or body.output_format!='png'):
        raise HTTPException(400,'This joint set supports one 1024x1024 text-generated image')
    progress_id = request.headers.get('X-SparkTalk-Request-ID', secrets.token_hex(16))
    if not re.fullmatch(r'[a-f0-9]{32}', progress_id): raise HTTPException(400, 'Invalid request ID')
    prune(ROOT)
    append_event(ROOT, progress_id, {'event':'queued', 'stage':'queued', 'kind':kind, 'time':time.time(), 'elapsed_seconds':0, 'eta_seconds':None, 'eta_scope':None})
    queued+=1
    entered=False
    try:
        async with lock:
            queued-=1;entered=True
            if quiescing or not ready(): raise HTTPException(503,'Generation service is unavailable')
            active+=1
            case=secrets.token_hex(16)
            payload={'case':case,'kind':kind,'prompt':body.prompt,'seed':body.seed,'progress_id':progress_id}
            temp=ROOT/'requests'/(case+'.tmp');temp.write_text(json.dumps(payload));temp.replace(temp.with_suffix('.json'))
            deadline=time.monotonic()+1800
            result_path=ROOT/'results'/(case+'.json')
            try:
                while not result_path.exists():
                    if await request.is_disconnected(): cancel();raise HTTPException(499,'Generation canceled')
                    if time.monotonic()>deadline: cancel();raise HTTPException(504,'Generation timed out')
                    if not ready(): raise HTTPException(503,'Worker stopped during generation')
                    await asyncio.sleep(.2)
                result=json.loads(result_path.read_text())
                if result.get('status')!='success': raise HTTPException(502,'Video/image worker failed; inspect service logs')
                output=ROOT/'output'/(case+('.mp4' if kind=='h3' else '.png'))
                if not output.is_file() or output.stat().st_size==0:raise HTTPException(502,'Worker produced no media')
                if kind=='h3':return FileResponse(output,media_type='video/mp4',filename='MiniMax-H3-'+case+'.mp4',background=BackgroundTask(clean_job,case))
                encoded=base64.b64encode(output.read_bytes()).decode();clean_job(case)
                return {'created':int(time.time()),'seed':body.seed,'data':[{'b64_json':encoded}]}
            finally:
                active-=1;last_used=time.monotonic()
    finally:
        if not entered:queued-=1

@app.post('/v1/videos/generations')
async def video(body:Generate,request:Request):return await generate(body,'h3',request)

@app.post('/v1/images/generations')
async def image(body:Generate,request:Request):return await generate(body,'qwim',request)
