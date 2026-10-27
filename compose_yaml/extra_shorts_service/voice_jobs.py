"""Durable source preparation and standalone speech jobs; no GPU ownership here."""
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import store
import voice_catalog

TERMINAL = {'completed', 'needs_input', 'failed', 'canceled'}


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    action: Literal['prepare','generate']
    name: str = Field(default='', max_length=80)
    scope: Literal['persistent','temporary'] = 'temporary'
    voice_id: voice_catalog.VoiceID | None = None
    revision: int | None = Field(default=None, ge=1)
    target_voice_id: voice_catalog.VoiceID | None = None
    text: str = Field(default='', max_length=500)
    instructions: str = Field(default='', max_length=200)
    continuous: bool = False
    source_attachment: dict = Field(default_factory=dict)
    source_url: str = Field(default='', max_length=4096)
    source_name: str = Field(default='', max_length=255)
    source_mime: str = Field(default='', max_length=128)
    source_sha256: str = Field(default='', pattern=r'^(?:[a-f0-9]{64})?$')
    start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    track: int | None = Field(default=None, ge=0, le=64)
    channel: int | None = Field(default=None, ge=0, le=63)
    transcript: str = Field(default='', max_length=1200)
    candidate: int | None = Field(default=None, ge=0, le=9)
    source_job_id: str = Field(default='', pattern=r'^(?:[a-f0-9]{32})?$')
    idempotency_key: str = Field(min_length=1, max_length=160)

    @model_validator(mode='after')
    def coherent(self):
        if self.action=='prepare' and self.voice_id:
            raise ValueError('참조 재설정에는 target_voice_id와 원본을 지정해 주세요. voice_id는 대사 생성에 사용합니다')
        if self.source_job_id and (self.source_url or self.source_attachment):
            raise ValueError('새 원본과 이전 작업 원본을 동시에 지정할 수 없습니다')
        if (self.start is None) != (self.end is None) or (self.start is not None and (self.end <= self.start or self.end-self.start > 15)):
            raise ValueError('선택 구간은 시작·끝을 함께 지정하고 15초 이하여야 합니다')
        if self.action == 'generate' and not self.text.strip():
            raise ValueError('읽을 대사를 입력해 주세요')
        if self.action == 'generate':
            from speech import directed_text
            directed_text(self.text, self.instructions)
            if self.continuous:
                from speech import chunks
                chunks(directed_text(self.text, self.instructions), continuous=True)
        if self.action == 'prepare' and self.scope == 'persistent' and not self.name.strip():
            raise ValueError('저장할 음성 이름을 입력해 주세요')
        if any(c in self.instructions for c in '[]<>'):
            raise ValueError('전체 말투 지시는 괄호 없이 적고, 구간 태그는 대사에 넣어 주세요')
        if any(ord(c)<32 for c in self.name):
            raise ValueError('음성 이름에는 줄바꿈이나 제어문자를 사용할 수 없습니다')
        return self


def folder(job_id):
    if not re.fullmatch('[a-f0-9]{32}', job_id):
        raise ValueError('잘못된 음성 작업 ID')
    return store.ROOT / 'voice-jobs' / job_id


def initialize():
    with store.connect() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS voice_jobs (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, result TEXT NOT NULL DEFAULT "{}", error TEXT NOT NULL DEFAULT "", canceled INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1, published_version INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL, idempotency_key TEXT NOT NULL, UNIQUE(session_id,idempotency_key))')
        conn.execute('CREATE TABLE IF NOT EXISTS user_voices (id TEXT PRIMARY KEY, owner_session TEXT NOT NULL, scope TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, metadata TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL)')
        conn.execute("UPDATE voice_jobs SET status=CASE WHEN canceled=1 THEN 'canceled' ELSE 'queued' END,stage='음성 작업 복구 중',updated=? WHERE status='running'", (time.time(),))
    voice_catalog.LIBRARY.mkdir(parents=True, exist_ok=True)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def create(request, source=None):
    if source is not None:
        source.seek(0);source_hash=hashlib.file_digest(source,'sha256').hexdigest();source.seek(0)
        request=request.model_copy(update={'source_sha256':source_hash})
    existing = None
    with store.connect() as conn:
        existing = conn.execute('SELECT id FROM voice_jobs WHERE session_id=? AND idempotency_key=?', (request.session_id,request.idempotency_key)).fetchone()
    if existing:
        original=get(existing[0], request.session_id)
        keys=('action','name','scope','voice_id','target_voice_id','text','instructions','continuous','source_url','source_sha256','start','end','track','channel','transcript','candidate','source_job_id')
        if any(original['request'].get(k, False if k=='continuous' else None)!=request.model_dump().get(k) for k in keys):
            raise ValueError('같은 요청 키로 다른 음성 작업을 만들 수 없습니다')
        return original
    if request.target_voice_id:
        target=voice_catalog.get_voice(request.target_voice_id,request.session_id)
        if target['kind']!='custom' or request.action!='prepare':raise ValueError('사용자 음성의 참조만 업데이트할 수 있습니다')
    if request.voice_id:
        item=voice_catalog.get_voice(request.voice_id,request.session_id,request.revision)
        request=request.model_copy(update={'revision':item['revision']})
    elif source is None and not request.source_job_id:
        raise ValueError('참조할 음성 파일이나 등록된 음성을 지정해 주세요')
    if request.source_job_id:
        old=get(request.source_job_id,request.session_id)
        if not (folder(old['id'])/'source.bin').is_file():
            raise ValueError('재선택할 원본 파일이 없습니다')
    job_id=uuid.uuid4().hex;root=folder(job_id);root.mkdir(parents=True)
    if source is not None:
        import shutil
        with (root/'source.bin').open('wb') as dest:
            shutil.copyfileobj(source,dest,1024*1024)
        if not (root/'source.bin').stat().st_size:
            raise ValueError('빈 음성 파일입니다')
    elif request.source_job_id:
        import shutil
        source_path=folder(request.source_job_id)/'source.bin'
        with source_path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if old['request'].get('source_sha256') and actual!=old['request']['source_sha256']:raise ValueError('재사용할 원본 파일 해시가 다릅니다')
        shutil.copyfile(source_path,root/'source.bin')
    stamp=time.time()
    try:
        with store.connect() as conn:
            conn.execute('INSERT INTO voice_jobs(id,session_id,request,status,stage,created,updated,idempotency_key) VALUES(?,?,?,?,?,?,?,?)',(job_id,request.session_id,request.model_dump_json(),'queued','음성 작업 대기',stamp,stamp,request.idempotency_key))
    except Exception:
        import shutil
        shutil.rmtree(root)
        with store.connect() as conn:
            duplicate=conn.execute('SELECT id FROM voice_jobs WHERE session_id=? AND idempotency_key=?',(request.session_id,request.idempotency_key)).fetchone()
        if duplicate:return create(request, source)
        raise
    return get(job_id,request.session_id)


def get(job_id, session_id=None):
    folder(job_id)
    with store.connect() as conn:
        row=conn.execute('SELECT * FROM voice_jobs WHERE id=?',(job_id,)).fetchone()
    if not row or (session_id is not None and row['session_id']!=session_id):
        raise ValueError('이 대화의 음성 작업을 찾을 수 없습니다')
    value=dict(row);value['request']=json.loads(value['request']);value['result']=json.loads(value['result'])
    value['delivery_pending']=value['status'] in TERMINAL and value['published_version']<value['version']
    return value


def list_jobs(session_id):
    with store.connect() as conn:
        ids=[r[0] for r in conn.execute('SELECT id FROM voice_jobs WHERE session_id=? ORDER BY created DESC LIMIT 50',(session_id,))]
    return [get(x,session_id) for x in ids]


def claim():
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute("SELECT id FROM voice_jobs WHERE status='queued' AND canceled=0 ORDER BY created LIMIT 1").fetchone()
        if not row:return None
        conn.execute("UPDATE voice_jobs SET status='running',stage='음성 준비 중',updated=? WHERE id=?",(time.time(),row[0]))
    return row[0]


def update(job_id,status,stage,progress,result=None,error=''):
    with store.connect() as conn:
        if result is None:
            conn.execute('UPDATE voice_jobs SET status=?,stage=?,progress=?,error=?,updated=? WHERE id=?',(status,stage,max(0,min(100,progress)),error[:2000],time.time(),job_id))
        else:
            previous=conn.execute('SELECT result FROM voice_jobs WHERE id=?',(job_id,)).fetchone()
            if previous:
                old=json.loads(previous[0])
                for key in ('committed_voice_id','committed_revision'):
                    if key in old:result={key:old[key],**result}
            conn.execute('UPDATE voice_jobs SET status=?,stage=?,progress=?,result=?,error=?,updated=? WHERE id=?',(status,stage,max(0,min(100,progress)),json.dumps(result,ensure_ascii=False),error[:2000],time.time(),job_id))


def action(job_id,session_id,operation,changes=None):
    job=get(job_id,session_id)
    with store.connect() as conn:
        if operation=='cancel' and job['status'] in ('queued','running','needs_input'):
            conn.execute("UPDATE voice_jobs SET canceled=1,version=CASE WHEN published_version>=version THEN version+1 ELSE version END,status=CASE WHEN status='running' THEN status ELSE 'canceled' END,stage='음성 작업 중지 요청',updated=? WHERE id=?",(time.time(),job_id))
        elif operation in ('retry','choose') and job['status'] in ('failed','canceled','needs_input'):
            request=job['request']
            if changes:
                if job['result'].get('committed_voice_id'):
                    raise ValueError('음성 등록은 이미 끝났습니다. 해당 음성의 참조 변경 작업을 사용해 주세요')
                allowed={'start','end','track','channel','transcript','candidate','name','scope'}
                if set(changes)-allowed:raise ValueError('수정할 수 없는 음성 작업 항목입니다')
                request.update(changes)
            validated=Request.model_validate(request)
            conn.execute("UPDATE voice_jobs SET request=?,status='queued',stage='음성 작업 다시 준비',error='',canceled=0,version=version+1,updated=? WHERE id=?",(validated.model_dump_json(),time.time(),job_id))
        else:raise ValueError('현재 상태에서는 이 음성 작업을 수행할 수 없습니다')
    return get(job_id,session_id)


def acknowledge(job_id,version):
    with store.connect() as conn:
        conn.execute('UPDATE voice_jobs SET published_version=? WHERE id=? AND version=?',(version,job_id,version))


def pending_deliveries():
    with store.connect() as conn:
        ids=[r[0] for r in conn.execute("SELECT id FROM voice_jobs WHERE status IN ('completed','needs_input','failed','canceled') AND published_version<version ORDER BY updated LIMIT 30")]
    return [get(x) for x in ids]


def cancel_session(session_id):
    with store.connect() as conn:
        conn.execute("UPDATE voice_jobs SET canceled=1,status=CASE WHEN status='queued' THEN 'canceled' ELSE status END,stage='대화 삭제로 취소',version=version+1,updated=? WHERE session_id=? AND status IN ('queued','running')",(time.time(),session_id))
        rows=conn.execute("SELECT metadata FROM user_voices").fetchall()
    import voice_library
    for row in rows:
        current=json.loads(row[0])
        if current.get('scope')=='temporary' and current.get('owner_session')==session_id:
            current['status']='deleted';current['expires_at']=time.time();voice_library._record(current)
    voice_library.cleanup()
    return {'status':'ok'}
