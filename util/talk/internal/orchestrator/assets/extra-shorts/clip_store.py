"""Durable, conversation-scoped image-to-video productions."""
import base64
import hashlib
import io
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

import store


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    source_image_id: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    source_image: str = Field(min_length=1, max_length=44739320)
    brief: str = Field(min_length=3, max_length=12000)
    user_request: str = Field(default='',max_length=65536)
    resolution: Literal['480p', '720p'] = '480p'
    request_key: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**62), ge=0, lt=2**63-16)


def folder(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Invalid production ID')
    return store.ROOT / '_clips' / job_id


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def initialize():
    with store.connect() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS clip_jobs (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, request_key TEXT NOT NULL,
            fingerprint TEXT NOT NULL, project TEXT NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL, progress REAL NOT NULL, error TEXT NOT NULL DEFAULT '',
            canceled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL,
            UNIQUE(session_id, request_key))''')
        c.execute("UPDATE clip_jobs SET status=CASE WHEN canceled=1 THEN 'canceled' ELSE 'queued' END, stage=CASE WHEN canceled=1 THEN '중지됨' ELSE '이어서 준비 중' END,updated=? WHERE status='running'", (time.time(),))


def decode_image(value):
    prefix, separator, encoded = value.partition(',')
    formats = {'data:image/png;base64': 'PNG', 'data:image/jpeg;base64': 'JPEG', 'data:image/webp;base64': 'WEBP'}
    if not separator or prefix not in formats:
        raise ValueError('PNG, JPEG or WebP data URL required')
    raw = base64.b64decode(encoded, validate=True)
    if not raw or len(raw) > 32*1024**2:
        raise ValueError('Source image must be at most 32 MiB')
    with Image.open(io.BytesIO(raw)) as picture:
        if picture.format != formats[prefix] or picture.width*picture.height > 16*1024**2 or getattr(picture, 'is_animated', False):
            raise ValueError('Source must be a static image of at most 16 megapixels')
        image = ImageOps.exif_transpose(picture).convert('RGB')
        output = io.BytesIO(); image.save(output, format='PNG')
    return output.getvalue(), hashlib.sha256(raw).hexdigest()


def create(request):
    source, admitted_hash = decode_image(request.source_image)
    project = request.model_dump(exclude={'source_image'})
    project.update(source_sha256=hashlib.sha256(source).hexdigest(), admitted_sha256=admitted_hash)
    # An unspecified seed is assigned once, not on each network retry.
    ignored={'seed'} | ({'brief'} if project.get('user_request') else set())
    fingerprint = hashlib.sha256(json.dumps({k:v for k,v in project.items() if k not in ignored}, sort_keys=True).encode()).hexdigest()
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT id,fingerprint FROM clip_jobs WHERE session_id=? AND request_key=?', (request.session_id,request.request_key)).fetchone()
        if row:
            if row['fingerprint'] != fingerprint:
                raise ValueError('This request key already belongs to a different production')
            job_id = row['id']
        else:
            job_id = uuid.uuid4().hex
            root = folder(job_id); root.mkdir(parents=True)
            (root/'source.png').write_bytes(source)
            atomic_json(root/'project.json', project)
            timestamp = time.time()
            c.execute('INSERT INTO clip_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                      (job_id,request.session_id,request.request_key,fingerprint,json.dumps(project,ensure_ascii=False),
                       'queued','대기 중',0,'',0,timestamp,timestamp))
    return get(job_id)


def get(job_id):
    root = folder(job_id)
    with store.connect() as c:
        row = c.execute('SELECT * FROM clip_jobs WHERE id=?',(job_id,)).fetchone()
    if not row:
        raise KeyError(job_id)
    result = dict(row)
    result['project'] = json.loads(result['project'])
    result.pop('fingerprint',None)
    result['output_ready'] = result['status']=='completed' and (root/'final.mp4').is_file()
    if (root/'report.json').exists():
        result['report'] = json.loads((root/'report.json').read_text())
    elif result['status'] in ('failed','canceled') and (root/'state.json').exists():
        state=json.loads((root/'state.json').read_text())
        plan=json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else None
        result['report']={'passed':False,'plan':plan,'source_image_id':result['project']['source_image_id'],
                          'image_attempts':state.get('images',[]),'video_attempts':state.get('videos',[]),
                          'repair_budget':{'image_attempts_max':3,'video_attempts_max':3}}
    return result


def list_jobs(session_id):
    with store.connect() as c:
        ids = [r[0] for r in c.execute('SELECT id FROM clip_jobs WHERE session_id=? ORDER BY created DESC LIMIT 50',(session_id,))]
    return [get(job_id) for job_id in ids]


def claim():
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute("SELECT id FROM clip_jobs WHERE status='queued' AND canceled=0 ORDER BY created LIMIT 1").fetchone()
        if row:
            c.execute("UPDATE clip_jobs SET status='running',stage='준비 중',updated=? WHERE id=?",(time.time(),row[0]))
            return row[0]
    return None


def update(job_id,status,stage,progress,error=''):
    folder(job_id)
    with store.connect() as c:
        if status=='completed':
            changed=c.execute("UPDATE clip_jobs SET status='completed',stage=?,progress=?,error='',updated=? WHERE id=? AND canceled=0",
                              (stage,min(100,max(0,progress)),time.time(),job_id)).rowcount
            if not changed:
                c.execute("UPDATE clip_jobs SET status='canceled',stage='중지됨',updated=? WHERE id=? AND canceled=1",(time.time(),job_id))
            return
        c.execute('UPDATE clip_jobs SET status=?,stage=?,progress=?,error=?,updated=? WHERE id=?',
                  (status,stage,min(100,max(0,progress)),str(error)[:2500],time.time(),job_id))


def action(job_id,operation,cancel_turn_key=''):
    folder(job_id)
    if cancel_turn_key and (len(cancel_turn_key)!=64 or any(c not in '0123456789abcdef' for c in cancel_turn_key)):
        raise ValueError('Invalid cancellation request key')
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        if operation=='cancel':
            row=c.execute('SELECT project FROM clip_jobs WHERE id=?',(job_id,)).fetchone()
            if not row:raise KeyError(job_id)
            project=json.loads(row['project']);project['cancel_turn_key']=cancel_turn_key
            changed=c.execute("UPDATE clip_jobs SET project=?,canceled=1,status=CASE WHEN status='queued' THEN 'canceled' ELSE status END,stage='중지 요청',updated=? WHERE id=? AND status IN ('queued','running')",(json.dumps(project,ensure_ascii=False),time.time(),job_id)).rowcount
        elif operation=='retry':
            path=folder(job_id)/'state.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            if ((len(state.get('images',[]))>=3 and not state.get('selected_image')) or
                (len(state.get('videos',[]))>=3 and not state.get('accepted_video'))):
                raise ValueError('자동 수정 횟수를 모두 사용했습니다. 같은 작업을 다시 시작하지 말고 검사 결과와 새 수정 요청을 바탕으로 새 버전을 만드세요.')
            changed=c.execute("UPDATE clip_jobs SET canceled=0,status='queued',stage='검사한 단계부터 이어서 진행',error='',updated=? WHERE id=? AND status IN ('failed','canceled')",(time.time(),job_id)).rowcount
        else:
            raise ValueError('Unsupported production action')
        if not changed:
            raise ValueError('This action is unavailable for the current state')
    return get(job_id)
