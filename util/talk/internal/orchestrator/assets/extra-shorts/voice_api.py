"""Private CPU-worker voice API. Talk owns browser auth, uploads and publication."""
import json
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

import store
import voice_analysis
import voice_catalog
import voice_jobs
import voice_library

router=APIRouter()
MAX_SOURCE_BYTES=512*1024*1024


def capabilities(session_id):
    return {'version':1,'reference_conditioning':True,'weight_training':False,'reference_max_seconds':15,'preferred_speech_seconds':5,
            'minimum_speech_seconds':1,'short_reference_warning_below_seconds':5,'temporary_retention_seconds':voice_library.TEMP_LIFETIME_SECONDS,
            'max_auto_analysis_seconds':voice_analysis.MAX_SOURCE_SECONDS,'tagged_spoken_characters':160,'max_generation_characters':500,
            'input_formats':['mp3','wav','flac','ogg','oga','opus','aac','m4a','alac','aiff','aif','caf','wma','amr','ape','mka','mp4','mov','webm'],
            'input_requires_successful_decode':True,'raw_pcm_requires_format_metadata':True,'preview_format':'WAV PCM16 mono 44100Hz',
            'voices':voice_catalog.public_catalog(session_id),'selected_voice':store.selected_voice(session_id),
            'temporary_voices':[x for x in voice_catalog.public_catalog(session_id,True) if x.get('scope')=='temporary'],
            'diarization':'Use available ASR speaker information; unavailability is explicitly reported. Speaker labels belong to a candidate, not an identity.',
            'lifecycle':'Persistent voices are available across this local Talk library; temporary references are conversation-scoped. Source attachments and generated results follow conversation retention.',
            'limitations':['Emotion and scoped direction adherence are not guaranteed.','No voice identity similarity score is claimed.','Help/list/cached preview do not invoke Fish.']}


class Manage(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1,max_length=128,pattern=r'^[a-zA-Z0-9_-]+$')
    action: str
    voice_id: str = ''
    name: str = Field(default='',max_length=80)


@router.post('/v1/voice-tools')
def manage(request:Manage):
    try:
        if request.action in ('help','capabilities','list'):return capabilities(request.session_id)
        if request.action=='inspect':
            item=voice_catalog.get_voice(request.voice_id,request.session_id)
            return {k:v for k,v in item.items() if k not in ('reference_file','preview_file','owner_session')}
        if request.action in ('rename','delete','promote'):return voice_library.manage(request.session_id,request.voice_id,request.action,request.name)
        raise ValueError('지원하지 않는 음성 관리 작업')
    except (ValueError,OSError) as error:raise HTTPException(422,str(error)) from error


@router.post('/v1/voice-jobs')
def create(request:voice_jobs.Request):
    try:return voice_jobs.create(request)
    except (ValueError,OSError) as error:raise HTTPException(422,str(error)) from error


@router.post('/v1/voice-jobs/upload')
async def upload(metadata:str=Form(...),file:UploadFile=File(...)):
    try:
        if len(metadata)>64*1024:raise ValueError('음성 작업 정보가 너무 큽니다')
        if file.size is None or file.size<1 or file.size>MAX_SOURCE_BYTES:raise ValueError('음성 원본의 허용 크기를 초과했거나 빈 파일입니다')
        request=voice_jobs.Request.model_validate_json(metadata)
        return voice_jobs.create(request,file.file)
    except (ValueError,OSError) as error:raise HTTPException(422,str(error)) from error
    finally:await file.close()


@router.get('/v1/voice-jobs')
def jobs(session_id:str):return voice_jobs.list_jobs(session_id)


@router.get('/v1/voice-jobs/{job_id}')
def job(job_id:str,session_id:str):
    try:return voice_jobs.get(job_id,session_id)
    except ValueError as error:raise HTTPException(404,str(error)) from error


class JobAction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:str
    action:str
    changes:dict=Field(default_factory=dict)


@router.post('/v1/voice-jobs/{job_id}/action')
def action(job_id:str,request:JobAction):
    try:return voice_jobs.action(job_id,request.session_id,request.action,request.changes)
    except (ValueError,OSError) as error:raise HTTPException(422,str(error)) from error


class Acknowledge(BaseModel):
    session_id:str
    version:int=Field(ge=1)


@router.post('/v1/voice-jobs/{job_id}/published')
def published(job_id:str,request:Acknowledge):
    try:
        voice_jobs.get(job_id,request.session_id);voice_jobs.acknowledge(job_id,request.version);return {'status':'ok'}
    except ValueError as error:raise HTTPException(404,str(error)) from error


@router.get('/v1/voice-jobs/{job_id}/files/{name}')
def artifact(job_id:str,name:str,session_id:str):
    try:
        value=voice_jobs.get(job_id,session_id)
        if not re.fullmatch(r'(?:source\.bin|candidate-[0-9]\.wav|reference\.wav|preview\.wav|output\.wav|rejected-[12]\.wav)',name):raise ValueError('잘못된 음성 작업 파일')
        p=voice_jobs.folder(job_id)/name
        if not p.is_file():raise ValueError('음성 작업 파일이 없습니다')
        mime='audio/wav' if name.endswith('.wav') else value['request'].get('source_mime') or 'application/octet-stream'
        return FileResponse(p,media_type=mime,filename=name)
    except ValueError as error:raise HTTPException(404,str(error)) from error


class Resolve(BaseModel):
    session_id:str
    voice_id:voice_catalog.VoiceID
    revision:int|None=Field(default=None,ge=1)
    job_id:str=''
    preparing:bool=False


@router.post('/v1/voice-resolve')
def resolve(request:Resolve):
    try:
        pinned=False;pending=False;revision=request.revision
        if request.job_id:
            try:
                current=voice_jobs.get(request.job_id,request.session_id)
            except ValueError:
                current=None
            if current:
                if current['status']!='running' or current['canceled']:raise ValueError('실행 중인 음성 작업이 아닙니다')
                if current['request'].get('voice_id')==request.voice_id and current['request'].get('revision')==revision:
                    pinned=True
                elif current['result'].get('staged_voice_id')==request.voice_id and request.preparing:
                    pending=True
                else:raise ValueError('음성 작업에 고정된 참조와 요청이 다릅니다')
            else:
                current=store.get(request.job_id)
                if current['session_id']!=request.session_id or current['status']!='running' or current['canceled']:raise ValueError('이 대화의 실행 중인 영상 작업이 아닙니다')
                plan=json.loads((store.ROOT/request.job_id/'storyboard.json').read_text());voice=plan['voice']
                if voice['preset']!=request.voice_id or voice.get('reference_revision')!=revision:raise ValueError('영상 작업에 고정된 참조와 요청이 다릅니다')
                pinned=True
        elif request.preparing:raise ValueError('준비 중인 참조는 소속 작업에서만 사용할 수 있습니다')
        item=voice_catalog.get_voice(request.voice_id,request.session_id,revision,allow_pending=pending,allow_history=pinned)
        return {'voice_id':item['id'],'revision':item['revision'],'reference_sha256':item['reference_sha256'],'preparing_reference':pending,'pinned_reference':pinned}
    except (ValueError,KeyError,OSError) as error:raise HTTPException(422,str(error)) from error


@router.post('/v1/voice-sessions/{session_id}/cancel')
def cancel_session(session_id:str):
    return voice_jobs.cancel_session(session_id)
