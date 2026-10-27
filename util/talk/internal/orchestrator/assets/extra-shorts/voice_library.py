"""The CPU worker is the sole writer of user voice manifests and immutable revisions."""
import hashlib
import json
import shutil
import time
import uuid
import wave
from pathlib import Path

import store
import voice_catalog
from voice_jobs import atomic_json

TEMP_LIFETIME_SECONDS = 24 * 60 * 60


def _record(current, job_id=None):
    stamp=time.time()
    with store.connect() as conn:
        conn.execute('INSERT INTO user_voices(id,owner_session,scope,name,status,revision,metadata,created,updated) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET scope=excluded.scope,name=excluded.name,status=excluded.status,revision=excluded.revision,metadata=excluded.metadata,updated=excluded.updated',
                     (current['id'],current['owner_session'],current['scope'],current['name'],current['status'],current['revision'],json.dumps(current,ensure_ascii=False),current.get('created',stamp),stamp))
        if job_id:
            conn.execute("UPDATE voice_jobs SET result=json_set(result,'$.committed_voice_id',?,'$.committed_revision',?) WHERE id=?",
                         (current['id'],current['revision'],job_id))
    atomic_json(voice_catalog.LIBRARY/current['id']/'current.json',current)


def repair_manifests():
    # A crash after the DB commit but before atomic publication is recoverable.
    with store.connect() as conn:
        rows=conn.execute('SELECT metadata FROM user_voices').fetchall()
    for row in rows:
        value=json.loads(row[0]);path=voice_catalog.LIBRARY/value['id']/'current.json'
        try: existing=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError): existing=None
        if existing!=value:
            atomic_json(path,value)


def stage(reference, transcript, session_id, name, metadata, voice_id=None):
    voice_id=voice_id or 'voice_'+uuid.uuid4().hex
    if not voice_catalog.is_custom(voice_id):raise ValueError('잘못된 사용자 음성 ID')
    parent=voice_catalog.LIBRARY/voice_id
    if (parent/'current.json').exists():
        current=voice_catalog._current(voice_id)
        if current['owner_session']!=session_id or current['status']!='preparing':raise ValueError('다른 음성의 준비 파일을 덮어쓸 수 없습니다')
    raw=reference.read_bytes();sha=hashlib.sha256(raw).hexdigest();text_sha=hashlib.sha256(transcript.encode()).hexdigest();dest=parent/'r1';dest.mkdir(parents=True,exist_ok=True)
    manifest={'id':voice_id,'revision':1,'kind':'custom','name':name or '임시 음성','description':'사용자 참조 음성','native_language':metadata.get('language') or '자동 인식',
              'reference_text':transcript,'reference_text_sha256':text_sha,'reference_sha256':sha,'reference_source':'사용자가 제공한 음성 샘플',
              'reference_duration_seconds':metadata['duration_seconds'],'source':metadata,'preview_text':'','preview_sha256':None,'warnings':metadata.get('warnings',[])}
    if (dest/'manifest.json').exists():
        old=json.loads((dest/'manifest.json').read_text())
        if old['reference_sha256']!=sha or old['reference_text_sha256']!=text_sha:
            raise ValueError('준비된 음성의 참조를 바꾸려면 새 준비 작업이 필요합니다')
    (dest/'reference.wav').write_bytes(raw);atomic_json(dest/'manifest.json',manifest)
    current={'id':voice_id,'kind':'custom','revision':1,'name':manifest['name'],'description':manifest['description'],'scope':'temporary','owner_session':session_id,'status':'preparing','expires_at':time.time()+TEMP_LIFETIME_SECONDS,'created':time.time()}
    _record(current)
    return voice_catalog.get_voice(voice_id,session_id,allow_pending=True)


def finish(voice_id, session_id, preview, preview_text, validation, scope, name='', target_id=None):
    if target_id:
        committed = voice_catalog.get_voice(target_id, session_id)
        if committed.get('staged_voice_id') == voice_id:
            return committed
    staged=voice_catalog.get_voice(voice_id,session_id,allow_pending=True)
    if staged['status']!='preparing':
        return staged
    source=voice_catalog.LIBRARY/voice_id/'r1'
    if target_id:
        old=voice_catalog.get_voice(target_id,session_id)
        if old['kind']!='custom':raise ValueError('기본 음성을 덮어쓸 수 없습니다')
        new_id=target_id;revision=old['revision']+1;scope=old['scope'];display=name or old['name'];owner=old['owner_session'];created=voice_catalog._current(target_id)['created']
    else:
        new_id=voice_id;revision=1;display=name or staged['name'];owner=session_id;created=voice_catalog._current(voice_id)['created']
    dest=voice_catalog.LIBRARY/new_id/f'r{revision}';dest.mkdir(parents=True,exist_ok=True)
    if target_id:
        if (dest/'manifest.json').exists():
            unfinished=json.loads((dest/'manifest.json').read_text())
            if unfinished.get('staged_voice_id')!=voice_id:raise ValueError('음성 버전이 이미 존재합니다')
        shutil.copy2(source/'reference.wav',dest/'reference.wav')
    data=preview.read_bytes();(dest/'preview.wav').write_bytes(data)
    with wave.open(str(preview)) as wav: preview_duration = wav.getnframes()/wav.getframerate()
    manifest={**json.loads((source/'manifest.json').read_text()),'id':new_id,'revision':revision,'name':display,'staged_voice_id':voice_id,'preview_text':preview_text,'preview_duration_seconds':preview_duration,'preview_sha256':hashlib.sha256(data).hexdigest(),'preview_validation':validation}
    atomic_json(dest/'manifest.json',manifest)
    current={'id':new_id,'kind':'custom','revision':revision,'name':display,'description':manifest['description'],'scope':scope,'owner_session':owner,'status':'ready','expires_at':time.time()+TEMP_LIFETIME_SECONDS if scope=='temporary' else None,'created':created}
    _record(current, manifest.get('source',{}).get('job_id'))
    if target_id:
        drop=voice_catalog._current(voice_id);drop['status']='deleted';drop['expires_at']=time.time();_record(drop)
    return voice_catalog.get_voice(new_id,session_id)


def manage(session_id,voice_id,action,name=''):
    voice=voice_catalog.get_voice(voice_id,session_id)
    if voice['kind']!='custom':raise ValueError('기본 음성은 이름 변경·등록 해제 대상이 아닙니다')
    current=voice_catalog._current(voice_id)
    if action=='rename' or (action=='promote' and name):
        if not name.strip() or len(name)>80 or any(ord(c)<32 for c in name):raise ValueError('음성 이름은 줄바꿈 없이 1~80자로 입력해 주세요')
        current['name']=name.strip()
    if action=='promote':
        current['scope']='persistent';current['expires_at']=None
    elif action=='delete':
        current['status']='deleted';current['expires_at']=time.time()
        with store.connect() as conn:conn.execute('DELETE FROM voice_preferences WHERE voice_id=?',(voice_id,))
    elif action!='rename':raise ValueError('지원하지 않는 음성 관리 작업')
    _record(current)
    return {'voice_id':voice_id,'name':current['name'],'scope':current['scope'],'status':current['status'],'existing_outputs_preserved':True}


def active_reference(voice_id):
    with store.connect() as conn:
        for table,field in [('voice_jobs','request'),('jobs','request')]:
            rows=conn.execute(f"SELECT {field} FROM {table} WHERE status IN ('queued','running','preview_ready','needs_input')").fetchall()
            if any(voice_id in row[0] for row in rows):return True
        rows=conn.execute("SELECT result FROM voice_jobs WHERE status IN ('queued','running')").fetchall()
        return any(voice_id in row[0] for row in rows)


def cleanup():
    now=time.time()
    with store.connect() as conn:rows=conn.execute('SELECT metadata FROM user_voices').fetchall()
    removed=[]
    for row in rows:
        current=json.loads(row[0]);expires=current.get('expires_at')
        if current['status']!='deleted' and not (current['scope']=='temporary' and expires and expires<=now):continue
        if active_reference(current['id']):continue
        current['status']='deleted';_record(current)
        for path in (voice_catalog.LIBRARY/current['id']).glob('r*'):
            if path.is_dir() and path.name[1:].isdigit():
                for audio in path.glob('*.wav'):audio.unlink()
        removed.append(current['id'])
    return removed
