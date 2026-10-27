"""Built-in voices and immutable, session-checked user reference revisions."""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Annotated

from pydantic import Field

BUILTIN_IDS = ('sohee', 'uncle_fu', 'vivian', 'serena', 'dylan', 'eric', 'ryan', 'aiden', 'ono_anna')
VOICE_ID_PATTERN = r'^(?:' + '|'.join(BUILTIN_IDS) + r'|voice_[a-f0-9]{32})$'
VoiceID = Annotated[str, Field(pattern=VOICE_ID_PATTERN)]
ROOT = Path(__file__).resolve().parent
LIBRARY = Path(os.environ.get('VOICE_LIBRARY_DIR', '/voices'))
LEGACY_REFERENCES = {'sohee': '58bfb593012f0e53cb498a0f48c726b553b6b9df502df3fede6f602453641604', 'uncle_fu': 'a9b3a7fc902802d5f121f36fa64d6c4bd590b822d6e401d16eaea2022414c803'}


def valid_id(voice_id):
    return isinstance(voice_id, str) and re.fullmatch(VOICE_ID_PATTERN, voice_id) is not None


def is_custom(voice_id):
    return valid_id(voice_id) and voice_id.startswith('voice_')


def builtin_catalog():
    return json.loads((ROOT / 'voices/catalog.json').read_text())['voices']


def _current(voice_id):
    if not is_custom(voice_id):
        raise ValueError('잘못된 사용자 음성 ID')
    path = LIBRARY / voice_id / 'current.json'
    try:
        item = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError('등록되지 않았거나 복구가 필요한 음성입니다: ' + voice_id) from error
    if item.get('id') != voice_id or item.get('kind') != 'custom':
        raise ValueError('음성 등록 정보가 일치하지 않습니다')
    return item


def get_voice(voice_id, session_id=None, revision=None, *, allow_pending=False, allow_history=False):
    if not valid_id(voice_id):
        raise ValueError('등록되지 않은 음성입니다: ' + str(voice_id))
    if not is_custom(voice_id):
        if revision not in (None, 1):
            raise ValueError('기본 음성 참조 버전은 1입니다')
        for item in builtin_catalog():
            if item['id'] == voice_id:
                return {**item, 'kind': 'builtin', 'scope': 'builtin', 'status': 'ready', 'revision': 1}
    current = _current(voice_id)
    private = current.get('scope') == 'temporary' or current.get('status') == 'preparing'
    if private and (not session_id or current.get('owner_session') != session_id):
        raise ValueError('이 대화에서 사용할 수 없는 임시 음성입니다')
    expired = current.get('expires_at') is not None and current['expires_at'] <= time.time()
    if (current.get('status') == 'deleted' or expired) and not (allow_history and revision):
        raise ValueError('삭제되었거나 만료된 음성입니다')
    if current.get('status') not in ('ready', 'deleted') and not (allow_pending and current.get('status') == 'preparing'):
        raise ValueError('아직 사용할 준비가 끝나지 않은 음성입니다')
    number = revision if revision is not None else current['revision']
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise ValueError('잘못된 음성 참조 버전')
    try:
        reference = json.loads((LIBRARY / voice_id / f'r{number}' / 'manifest.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError('음성 참조 버전을 읽을 수 없습니다') from error
    if reference.get('id') != voice_id or reference.get('revision') != number:
        raise ValueError('음성 참조 버전이 일치하지 않습니다')
    return {**reference, **{key: current.get(key) for key in ('name','description','scope','owner_session','status','expires_at')},
            'id': voice_id, 'kind': 'custom', 'revision': number}


def catalog(session_id=None):
    voices = [get_voice(item['id']) for item in builtin_catalog()]
    if LIBRARY.is_dir():
        for folder in sorted(LIBRARY.glob('voice_*')):
            try:
                item = get_voice(folder.name, session_id)
                if item['status'] == 'ready':
                    voices.append(item)
            except ValueError:
                continue
    return voices


def reference(voice_id, session_id=None, revision=None, *, allow_pending=False, allow_history=False):
    item = get_voice(voice_id, session_id, revision, allow_pending=allow_pending, allow_history=allow_history)
    path = LIBRARY / voice_id / f"r{item['revision']}" / 'reference.wav' if item['kind'] == 'custom' else ROOT / item['reference_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['reference_sha256']:
        raise ValueError('음성 참조 파일 검증 실패: ' + voice_id)
    transcript = item['reference_text']
    if item['kind'] == 'custom' and hashlib.sha256(transcript.encode()).hexdigest() != item['reference_text_sha256']:
        raise ValueError('음성 참조 대사 검증 실패: ' + voice_id)
    return path, transcript


def preview(voice_id, session_id=None, revision=None):
    item = get_voice(voice_id, session_id, revision)
    path = LIBRARY / voice_id / f"r{item['revision']}" / 'preview.wav' if item['kind'] == 'custom' else ROOT / item['preview_file']
    if not item.get('preview_sha256') or hashlib.sha256(path.read_bytes()).hexdigest() != item['preview_sha256']:
        raise ValueError('음성 미리 듣기 검증 실패: ' + voice_id)
    return path


def public_catalog(session_id=None, include_temporary=False):
    fields = {'id','name','description','native_language','kind','scope','status','revision','reference_sha256',
              'preview_text','preview_sha256','preview_duration_seconds','reference_duration_seconds','reference_source','warnings','expires_at'}
    return [{k:v for k,v in item.items() if k in fields} for item in catalog(session_id) if include_temporary or item.get('scope')!='temporary']
