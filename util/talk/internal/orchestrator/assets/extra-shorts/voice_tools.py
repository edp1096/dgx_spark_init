"""Conversation-scoped selection and deterministic voice-only revisions."""
import json
from copy import deepcopy

import store
import voice_catalog
from spec import Plan


def execute(session_id, action, voice_id='', job_id=''):
    if action == 'list':
        voices=voice_catalog.public_catalog(session_id)
        return {'voices': voices, 'count': len(voices),
                'selected_voice': store.selected_voice(session_id), 'training': False,
                'usage': 'preview plays saved Fish audio; select sets this conversation default; apply creates a new version of an existing video.'}
    if not voice_id and action in ('preview', 'apply'):
        voice_id = store.selected_voice(session_id) or ''
    item = voice_catalog.get_voice(voice_id, session_id)
    if action == 'preview':
        voice_catalog.preview(voice_id, session_id)
        return {'voice': item['id'], 'name': item['name'], 'engine': 'Fish', 'text': item['preview_text'], 'preview_sha256': item['preview_sha256']}
    if action == 'select':
        store.select_voice(session_id, voice_id)
        return {'selected_voice': voice_id, 'scope': 'this_conversation', 'applied_to_existing_video': False}
    if action != 'apply':
        raise ValueError('지원하지 않는 음성 작업')
    old = store.get(job_id)
    if old['session_id'] != session_id:
        raise ValueError('이 대화의 작업이 아닙니다')
    if old['status'] not in ('completed', 'preview_ready'):
        raise ValueError('완성되거나 미리보기가 준비된 영상에만 음성을 적용할 수 있습니다')
    root = store.ROOT / job_id
    plan = json.loads((root / 'storyboard.json').read_text())
    plan['voice']['preset'] = voice_id
    plan['voice']['reference_revision'] = item['revision']
    # Keep words and visuals fixed. If speech cannot fit, report it rather than
    # quietly rewriting the user's video during a voice-only change.
    plan['verbatim_narration'] = True
    for i, scene in enumerate(plan['scenes']):
        scene['reuse_scene'] = i
    plan = Plan.model_validate(plan).model_dump()
    data = deepcopy(old['project'])
    data.update(source_job_id=job_id, scenes=[], plan=plan, voice_preset=voice_id, voice_revision=item['revision'],
                design_mode='preserve', preview_only=False, voice_only=True,
                brief=f"Change narration voice to {voice_id}. Preserve words, images, music and visual design.")
    return store.create(store.Project.model_validate(data))
