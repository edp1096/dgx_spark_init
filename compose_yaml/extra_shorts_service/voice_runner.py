"""One durable user-voice job. Managed Talk bridges own ASR/Fish admission."""
import base64
import hashlib
import json
import os
import shutil
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

import speech
import voice_analysis
import voice_catalog
import voice_jobs
import voice_library
from voice_quality import coverage, normalize

TALK=os.environ.get('SHORTS_TALK_ENDPOINT','http://127.0.0.1:8585').rstrip('/')
PREVIEW_TEXT='안녕하세요. 이 목소리로 새로운 이야기를 전해 드립니다.'


class Canceled(Exception):pass


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def file_record(root,name,label):
    p=root/name
    if not p.is_file():return None
    return {'file':name,'name':label,'sha256':digest(p),'bytes':p.stat().st_size}


def bridge(path,payload,timeout=1200):
    with httpx.Client(timeout=timeout) as client:
        r=client.post(TALK+path,json=payload)
        if r.status_code!=200:
            raise ValueError(f'음성 처리 요청이 실패했습니다 (HTTP {r.status_code}): '+r.text[:1000])
        return r.json()


def asr(job,path):
    return bridge('/api/voices/transcribe',{'session_id':job['session_id'],'job_id':job['id'],'audio':base64.b64encode(path.read_bytes()).decode()},180)


def transcribe_candidate(job,root,candidate,check):
    check();key=f"candidate-{candidate['index']}-asr.json";p=root/candidate['file'];data=read(root/key)
    sha=digest(p)
    if not data or data.get('audio_sha256')!=sha:
        data={**asr(job,p),'audio_sha256':sha};voice_jobs.atomic_json(root/key,data)
    text=data.get('text','').strip()
    speakers=set(x for turn in data.get('turns',[]) for x in turn.get('speakers',[]))
    overlapping=any(len(turn.get('speakers',[]))>1 for turn in data.get('turns',[]))
    candidate.update(transcript=text,language=data.get('language',''),diarization_status=data.get('diarization_status','unavailable'),speaker_count=len(speakers),speaker_turns=data.get('turns',[]),overlapping=overlapping)
    return candidate,data


def generate(job,request,root,voice,reference_pending,check,stage):
    text=request.text if request.action=='generate' else PREVIEW_TEXT
    model_input=speech.directed_text(text,request.instructions if request.action=='generate' else '')
    expected=speech.spoken_text(text)
    continuous=request.continuous or request.action=='prepare'
    fingerprint=hashlib.sha256(json.dumps({'voice':voice['id'],'revision':voice['revision'],'reference':voice['reference_sha256'],'transcript':voice.get('reference_text_sha256'),'input':model_input,'policy':'native-tags-v1',**({'continuous':True} if continuous else {})},sort_keys=True).encode()).hexdigest()
    final=read(root/'generation.json')
    if final and final.get('fingerprint')==fingerprint and (root/final['file']).is_file() and digest(root/final['file'])==final['sha256']:
        return final
    for attempt in range(1,3):
        check();stage(f'음성 생성·대사 확인 {attempt}/2',65+(attempt-1)*12)
        meta_path=root/f'attempt-{attempt}.json';data=read(meta_path)
        wav_path=root/f'rejected-{attempt}.wav'
        if not data or data.get('fingerprint')!=fingerprint or not wav_path.is_file() or digest(wav_path)!=data.get('sha256'):
            response=bridge('/api/shorts/speech',{'session_id':job['session_id'],'job_id':job['id'],'texts':[model_input],
                             'voice':{'preset':voice['id'],'reference_revision':voice['revision'],'preparing_reference':reference_pending,'sentence_pause':.18,'continuous':continuous}})
            if response.get('voice_preset')!=voice['id'] or response.get('reference_sha256')!=voice['reference_sha256']:
                raise ValueError('요청한 참조 음성과 실제 생성 음성이 일치하지 않습니다')
            if len(response.get('items',[]))!=1:raise ValueError('음성 생성 결과 개수가 다릅니다')
            item=response['items'][0];audio=base64.b64decode(item.pop('audio'),validate=True);wav_path.write_bytes(audio)
            with wave.open(str(wav_path)) as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,44100):raise ValueError('생성 음성 규격이 올바르지 않습니다')
            data={'fingerprint':fingerprint,'sha256':digest(wav_path),'generation':item,'attempt':attempt}
            voice_jobs.atomic_json(meta_path,data)
        check();stage('생성한 대사를 음성 인식으로 확인',80+(attempt-1)*8)
        validation_path=root/f'attempt-{attempt}-validation.json';validation=read(validation_path)
        if not validation or validation.get('audio_sha256')!=data['sha256']:
            recognized=asr(job,wav_path);validation={**coverage(expected,recognized.get('text','')),'audio_sha256':data['sha256'],'asr':recognized}
            # Quiet speech can be missed while the louder prefix is recognized.
            # Check a cropped tail independently, then compare the unprompted
            # combined transcript. Never change the delivered waveform.
            if not validation['passed'] and recognized.get('words'):
                diagnostics=[]
                with wave.open(str(wav_path)) as wav:rate=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float64)/32768
                cuts=[]
                words=recognized['words']
                for i,word in enumerate(words[:-1]):
                    if words[i+1]['start']-word['end']>.35:cuts.append((i,float(word['end'])))
                if words and float(words[-1]['end'])<len(pcm)/rate-.5:cuts.append((len(words)-1,float(words[-1]['end'])))
                for ordinal,(last,cut) in enumerate(cuts[:2]):
                    tail=pcm[round(cut*rate):]
                    if len(tail)<rate*.3 or np.max(np.abs(tail))<.0002:continue
                    gain=max(1.,min(31.622,.1/max(float(np.sqrt(np.mean(tail*tail))),1e-12),.95/max(float(np.max(np.abs(tail))),1e-12)))
                    path=root/f'diagnostic-{attempt}-{ordinal}.wav';path.write_bytes(speech.wav_bytes(np.clip(tail*gain*32768,-32768,32767).astype('<i2').tobytes(),rate))
                    retry=asr(job,path);joined=' '.join(x['word'] for x in words[:last+1])+' '+retry.get('text','');result=coverage(expected,joined)
                    diagnostics.append({'start':cut,'gain_db':float(20*np.log10(gain)),'result':result,'delivered_audio_changed':False})
                    if result['passed']:
                        validation['passed']=True;validation['diagnostic_recovery']=result;break
                validation['diagnostics']=diagnostics
            voice_jobs.atomic_json(validation_path,validation)
        if validation['passed']:
            name='output.wav' if request.action=='generate' else 'preview.wav';shutil.copy2(wav_path,root/name)
            final={**data,'file':name,'text':text,'model_input':model_input,'validation':validation,'requested_instructions':request.instructions,'reference_revision':voice['revision'],'tag_effect_verified':False}
            voice_jobs.atomic_json(root/'generation.json',final);return final
    raise ValueError('두 번 생성했지만 대사 확인을 통과하지 못했습니다. 대사를 바꾸지 않았습니다. 참조 구간이나 대사를 확인해 주세요.')


def run(job_id):
    job=voice_jobs.get(job_id);request=voice_jobs.Request.model_validate(job['request']);root=voice_jobs.folder(job_id);partial=job.get('result',{})
    def check():
        if voice_jobs.get(job_id)['canceled']:raise Canceled()
    def stage(message,progress):
        check();voice_jobs.update(job_id,'running',message,progress,result=partial)
    def complete(voice,result):
        visible={k:voice.get(k) for k in ['id','name','kind','scope','revision','status','reference_sha256','reference_source','expires_at','warnings']}
        files=[]
        if (root/'reference.wav').exists():files.append(file_record(root,'reference.wav','사용한 참조 구간.wav'))
        files.append(file_record(root,result['file'],('생성 음성.wav' if request.action=='generate' else '등록 미리 듣기.wav')))
        partial.update(voice=visible,voice_id=voice['id'],generation=result,files=files,
                       registered=voice.get('scope')=='persistent' and voice.get('status')=='ready' and not request.voice_id,
                       temporary=voice.get('scope')=='temporary',limitations=['샘플 참조 방식이며 모델을 재학습하지 않습니다.','대사 검사와 감정·음색의 주관적 품질은 다른 항목입니다.','일회성 참조의 정리와 대화 원본·생성 파일의 보관은 별개입니다.'])
        check();voice_jobs.update(job_id,'completed','음성 준비 완료 · 대화 게시 대기',100,result=partial)
    try:
        if partial.get('committed_voice_id'):
            voice_library.repair_manifests()
            voice=voice_catalog.get_voice(partial['committed_voice_id'],request.session_id,partial['committed_revision'],allow_history=True)
            result=read(root/'generation.json')
            if not result:raise ValueError('등록은 끝났지만 생성 결과 기록을 복구해야 합니다')
            if not (root/result['file']).is_file() or digest(root/result['file'])!=result['sha256']:
                shutil.copy2(voice_catalog.preview(voice['id'],request.session_id,voice['revision']),root/result['file'])
            if digest(root/result['file'])!=result['sha256']:raise ValueError('복구한 미리 듣기 파일이 등록 기록과 다릅니다')
            complete(voice,result);return
        if request.voice_id:
            voice=voice_catalog.get_voice(request.voice_id,request.session_id,request.revision,allow_history=True);reference_pending=False
        else:
            source=root/'source.bin'
            if not source.is_file():raise ValueError('참조할 원본 파일을 찾을 수 없습니다')
            source_sha=digest(source);analysis_key=hashlib.sha256(json.dumps({'sha':source_sha,'start':request.start,'end':request.end,'track':request.track,'channel':request.channel},sort_keys=True).encode()).hexdigest()
            analysis=read(root/'analysis.json')
            if not analysis or analysis.get('key')!=analysis_key or any(not (root/c['file']).exists() for c in analysis.get('candidates',[])):
                stage('원본 파일 확인·발화 구간 분석',8)
                analysis=voice_analysis.analyze(source,root,request,check,lambda fraction:stage('발화 구간 분석 중',10+fraction*25));analysis['key']=analysis_key;voice_jobs.atomic_json(root/'analysis.json',analysis)
            partial.update(source=analysis['source'],source_attachment=request.source_attachment,source_url=request.source_url,candidates=analysis['candidates'],files=[file_record(root,c['file'],f"후보 {c['index']+1} · {c['start']:.2f}–{c['end']:.2f}초.wav") for c in analysis['candidates']])
            valid=[];data_by_index={}
            for c in analysis['candidates']:
                stage(f"참조 후보 {c['index']+1}의 대사·화자 확인",40+c['index']*4)
                candidate,recognized=transcribe_candidate(job,root,c,check);data_by_index[c['index']]=recognized
                if candidate['speaker_count']<=1 and not candidate['overlapping'] and normalize(candidate['transcript']):valid.append(candidate)
            partial['candidates']=analysis['candidates'];voice_jobs.atomic_json(root/'analysis.json',analysis)
            if request.candidate is not None:
                candidate=next((c for c in analysis['candidates'] if c['index']==request.candidate),None)
                if candidate is None:raise voice_analysis.NeedsInput('선택한 후보 번호가 없습니다.',partial)
                if candidate['speaker_count']>1 or candidate['overlapping']:raise voice_analysis.NeedsInput('선택한 구간에 여러 화자가 있습니다. 한 화자가 말하는 더 짧은 구간을 지정해 주세요.',partial)
            elif valid:candidate=valid[0]
            elif request.transcript:
                candidate=analysis['candidates'][0]
                if candidate['speaker_count']>1 or candidate['overlapping']:raise voice_analysis.NeedsInput('한 화자가 말하는 구간을 선택해 주세요.',partial)
            else:raise voice_analysis.NeedsInput('참조 대사 또는 한 화자의 구간을 확인하지 못했습니다. 구간이나 대사를 지정해 주세요.',partial)
            transcript=request.transcript.strip() or candidate['transcript']
            if not normalize(transcript):raise voice_analysis.NeedsInput('선택 구간에서 실제로 말한 대사를 입력해 주세요.',partial)
            reference=root/'reference.wav';shutil.copy2(root/candidate['file'],reference)
            recognized=data_by_index[candidate['index']]
            # Refine a cut inside continuous speech using the last complete ASR word.
            if candidate['boundary_uncertain'] and not request.transcript and recognized.get('words'):
                last=float(recognized['words'][-1]['end']);finish=min(candidate['end'],candidate['start']+last+.12)
                if finish-candidate['start']>=1 and candidate['end']-finish>.2:
                    voice_analysis.extract(source,reference,candidate['start'],finish,candidate['track'],candidate['channel'],check)
                    refined=asr(job,reference)
                    if normalize(refined.get('text','')):
                        transcript=refined['text'].strip();candidate={**candidate,'end':finish,'duration_seconds':finish-candidate['start']};recognized=refined
                    else:raise voice_analysis.NeedsInput('잘린 문장 경계의 대사를 확인하지 못했습니다. 구간을 다시 선택해 주세요.',partial)
            warnings=list(candidate['warnings'])
            if candidate['diarization_status']!='completed':warnings.append('자동 화자 분리가 확인되지 않았습니다. 선택된 구간의 목소리를 미리 듣기로 확인해 주세요.')
            warnings.append('원본 전체의 화자 동일성을 확인한 결과는 아닙니다. 후보 안의 화자 정보만 사용했습니다.')
            metadata={**candidate,'source_sha256':source_sha,'source_name':request.source_name,'source_attachment':request.source_attachment,'source_url':request.source_url,'transcript_source':'user' if request.transcript else 'asr','language':recognized.get('language',''),'warnings':warnings,'job_id':job_id}
            key=hashlib.sha256((digest(reference)+'\n'+transcript).encode()).hexdigest();prepared=read(root/'prepared.json')
            staged_id=None
            if prepared and prepared.get('key')==key:
                try:
                    previous=voice_catalog.get_voice(prepared['voice_id'],request.session_id,allow_pending=True)
                    staged_id=previous['id']
                except ValueError:pass
            if staged_id:
                voice=previous
            else:
                voice=voice_library.stage(reference,transcript,request.session_id,request.name,metadata)
                voice_jobs.atomic_json(root/'prepared.json',{'key':key,'voice_id':voice['id']})
            partial.update(staged_voice_id=voice['id'],reference={**metadata,'transcript':transcript,'file':'reference.wav','sha256':voice['reference_sha256']},warnings=warnings)
            stage('참조 음성 준비 완료',60);reference_pending=voice['status']=='preparing'
        result=generate(job,request,root,voice,reference_pending,check,stage)
        if not request.voice_id:
            stage('음성 등록 정보 저장',95)
            voice=voice_library.finish(voice['id'],request.session_id,root/result['file'],result['text'],result['validation'],request.scope,request.name,request.target_voice_id)
        complete(voice,result)
    except Canceled:
        voice_jobs.update(job_id,'canceled','음성 작업 중지됨',0,result=partial)
    except voice_analysis.NeedsInput as error:
        partial.update(error.details);partial['question']=str(error);partial['files']=[x for x in partial.get('files',[]) if x]
        voice_jobs.update(job_id,'needs_input',str(error),50,result=partial)
    except Exception as error:
        files=[file_record(root,f'rejected-{i}.wav',f'대사 확인 실패 샘플 {i}.wav') for i in [1,2]]
        partial['files']=[x for x in files if x]
        voice_jobs.update(job_id,'failed','음성 처리 실패',0,result=partial,error=str(error))
        (root/'error.txt').write_text(str(error))
    finally:
        (root/'analysis.f32').unlink(missing_ok=True)


if __name__=='__main__':run(sys.argv[1])
