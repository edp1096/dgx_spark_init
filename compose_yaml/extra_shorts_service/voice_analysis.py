"""Bounded local decoding, CPU VAD and contiguous reference candidates."""
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REFERENCE_LIMIT = 15.0
MIN_SPEECH_SECONDS = 1.0
SHORT_SPEECH_SECONDS = 5.0
MAX_SOURCE_SECONDS = int(os.environ.get('VOICE_MAX_SOURCE_SECONDS', '7200'))
VAD_SHA = '7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49'


class NeedsInput(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def command(args, check, timeout=600):
    # Every child stays in the job's process group so cancellation kills FFmpeg too.
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        start=time.monotonic()
        while True:
            try:
                out,err=process.communicate(timeout=.5)
                break
            except subprocess.TimeoutExpired:
                try:check()
                except BaseException:
                    process.kill();process.communicate();raise
                if time.monotonic()-start>timeout:
                    process.kill();process.communicate();raise ValueError('음성 파일 처리 제한 시간을 초과했습니다')
        if process.returncode:
            detail=err.decode(errors='replace')[-1500:]
            raise ValueError('음성 파일을 디코딩하지 못했습니다: '+detail)
        return out


def probe(source, check):
    raw=command(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-show_streams','-show_format','-of','json',str(source)],check,30)
    value=json.loads(raw);streams=[]
    for item in value.get('streams',[]):
        if item.get('codec_type')=='audio':
            streams.append({'index':int(item['index']),'codec':item.get('codec_name','unknown'),'sample_rate':int(item.get('sample_rate') or 0),'channels':int(item.get('channels') or 0),'language':item.get('tags',{}).get('language','')})
    if not streams:raise ValueError('파일에 읽을 수 있는 오디오 트랙이 없습니다')
    try:duration=float(value.get('format',{}).get('duration',0))
    except (TypeError,ValueError):duration=0
    if not math.isfinite(duration) or duration<0:raise ValueError('오디오 길이 정보가 올바르지 않습니다')
    return {'duration_seconds':duration,'format':value.get('format',{}).get('format_name','unknown'),'streams':streams,'bytes':source.stat().st_size}


def extract(source, destination, start, end, track, channel, check):
    if not all(math.isfinite(x) for x in [start,end]) or start<0 or end<=start or end-start>REFERENCE_LIMIT+.00001:
        raise ValueError('참조 구간은 0초 이후의 15초 이하 연속 구간이어야 합니다')
    command(['ffmpeg','-y','-v','error','-nostdin','-protocol_whitelist','file,pipe','-ss',str(start),'-i',str(source),'-t',str(end-start),'-map',f'0:{track}','-vn','-af',f'pan=mono|c0=c{channel}','-ar','44100','-ac','1','-c:a','pcm_s16le',str(destination)],check)


def _vad(data, check, progress):
    import onnxruntime as ort
    path=Path(__file__).parent/'vad/silero_vad_16k_op15.onnx'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=VAD_SHA:raise ValueError('발화 검출 모델 파일 검증 실패')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'],sess_options=options)
    channels=data.shape[1];state=np.zeros((2,channels,128),dtype=np.float32);context=np.zeros((channels,64),dtype=np.float32)
    peak=np.zeros(channels)
    for offset in range(0,len(data),16000*60):peak=np.maximum(peak,np.max(np.abs(data[offset:offset+16000*60]),axis=0))
    gains=np.minimum(31.622,np.maximum(1.,.8/np.maximum(peak,1e-12))).astype(np.float32)
    probabilities=[];last_check=0
    for offset in range(0,len(data),512):
        block=np.asarray(data[offset:offset+512],dtype=np.float32).T
        if block.shape[1]<512:block=np.pad(block,((0,0),(0,512-block.shape[1])))
        block=block*gains[:,None];x=np.concatenate((context,block),axis=1)
        output,state=session.run(None,{'input':x,'state':state,'sr':np.array(16000,dtype=np.int64)})
        probabilities.append(output[:,0]);context=x[:,-64:]
        if time.monotonic()-last_check>1:
            check();progress(offset/max(1,len(data)));last_check=time.monotonic()
    return np.asarray(probabilities),[float(20*np.log10(x)) for x in gains]


def _intervals(probabilities, duration):
    ranges=[];begin=None;last_voice=None
    for i,p in enumerate(probabilities):
        if p>=.5:
            if begin is None:begin=i*.032
            last_voice=(i+1)*.032
        elif begin is not None and i*.032-last_voice>=.35:
            if last_voice-begin>=.2:ranges.append((begin,min(duration,last_voice)))
            begin=None
    if begin is not None and last_voice-begin>=.2:ranges.append((begin,min(duration,last_voice)))
    return ranges


def _windows(intervals,duration):
    windows=[]
    for index,(start,end) in enumerate(intervals):
        if end-start>14.6:
            # Long continuous speech has no trustworthy sentence boundary yet;
            # ASR word timing will refine the end of the chosen candidate.
            for begin in np.arange(start,end-1,12):
                finish=min(end,begin+14.6)
                windows.append((max(0,float(begin)-.12),min(duration,float(finish)+.12),True))
        else:
            finish=end
            for next_start,next_end in intervals[index+1:]:
                if next_start-finish>1.0 or next_end-start>14.6:break
                finish=next_end
            windows.append((max(0,start-.12),min(duration,finish+.12),False))
    return windows


def analyze(source,root,request,check,progress):
    info=probe(source,check);streams=info['streams']
    if request.track is None and len(streams)>1:
        raise NeedsInput('오디오 트랙이 여러 개입니다. 사용할 트랙 번호를 골라 주세요.',{'source':info,'selection':'track'})
    stream=next((x for x in streams if request.track is None or x['index']==request.track),None)
    if stream is None:raise ValueError('선택한 오디오 트랙이 없습니다')
    channels=stream['channels']
    if channels<1:raise ValueError('오디오 채널 정보를 읽을 수 없습니다')
    if request.channel is not None and request.channel>=channels:raise ValueError('선택한 오디오 채널이 없습니다')
    if channels>2 and request.channel is None:
        raise NeedsInput('다채널 오디오입니다. 사용할 채널 번호를 골라 주세요. 채널은 0부터 셉니다.',{'source':info,'selection':'channel','track':stream['index']})
    if info['duration_seconds']>MAX_SOURCE_SECONDS and request.start is None:
        raise NeedsInput(f'자동 분석은 {MAX_SOURCE_SECONDS//60}분까지 지원합니다. 사용할 15초 이하 구간을 지정해 주세요.',{'source':info,'selection':'range'})
    offset=request.start or 0.;length=request.end-request.start if request.start is not None else MAX_SOURCE_SECONDS+1
    if info['duration_seconds'] and offset>=info['duration_seconds']:raise ValueError('선택한 시작 시간이 파일 길이를 벗어났습니다')
    decoded=root/'analysis.f32';args=['ffmpeg','-y','-v','error','-nostdin','-protocol_whitelist','file,pipe','-ss',str(offset),'-i',str(source),'-t',str(length),'-map',f"0:{stream['index']}",'-vn']
    if request.channel is not None:args+=['-af',f'pan=mono|c0=c{request.channel}'];nchannels=1
    else:nchannels=channels
    args+=['-ar','16000','-ac',str(nchannels),'-c:a','pcm_f32le','-f','f32le',str(decoded)]
    command(args,check)
    size=decoded.stat().st_size
    if not size or size%(4*nchannels):raise ValueError('디코딩된 오디오가 비어 있거나 잘렸습니다')
    frame_count=size//(4*nchannels);duration=frame_count/16000
    if duration>MAX_SOURCE_SECONDS and request.start is None:raise ValueError('자동 분석 최대 길이를 넘었습니다. 사용할 구간을 지정해 주세요')
    if duration<1:raise NeedsInput('참조할 발화가 너무 짧습니다. 말이 이어지는 샘플을 더 올려 주세요.',{'source':info,'selection':'more_audio'})
    data=np.memmap(decoded,dtype='<f4',mode='r',shape=(frame_count,nchannels))
    if not np.all(np.isfinite(data)):raise ValueError('오디오에 유효하지 않은 샘플 값이 있습니다')
    probabilities,gains=_vad(data,check,progress)
    candidates=[]
    for ch in range(nchannels):
        intervals=_intervals(probabilities[:,ch],duration)
        windows=[(0.,duration,False)] if request.start is not None else _windows(intervals,duration)
        for start,end,uncertain in windows:
            span=end-start
            if span>REFERENCE_LIMIT+.001 or span<1:continue
            first=int(start/.032);last=min(len(probabilities),int(math.ceil(end/.032)))
            speech_seconds=float(np.count_nonzero(probabilities[first:last,ch]>=.5)*.032)
            if speech_seconds<MIN_SPEECH_SECONDS:continue
            samples=np.asarray(data[round(start*16000):round(end*16000),ch]);rms=float(np.sqrt(np.mean(samples*samples)));clip=float(np.mean(np.abs(samples)>=.999))
            if clip>.01:continue
            warnings=[]
            if speech_seconds<SHORT_SPEECH_SECONDS:warnings.append('실제 발화가 5초 미만인 짧은 참조입니다. 미리 듣기로 확인해 주세요.')
            if rms<.003:warnings.append('참조 음량이 작습니다. 원본 음량을 유지하며 결과를 확인합니다.')
            if clip>.001:warnings.append('참조에 일부 클리핑이 있습니다.')
            if stream['sample_rate']<16000:warnings.append('원본 샘플레이트가 낮아 음색 재현에 한계가 있을 수 있습니다.')
            if uncertain:warnings.append('연속 발화에서 추출한 구간이며 끝 경계를 대사 시간 정보로 다시 확인합니다.')
            score=min(speech_seconds,12)+3*speech_seconds/span-100*clip-1.5*uncertain
            candidates.append({'start':offset+start,'end':offset+end,'track':stream['index'],'channel':request.channel if request.channel is not None else ch,'duration_seconds':span,'speech_seconds':speech_seconds,'rms_dbfs':20*math.log10(max(rms,1e-12)),'clipped_fraction':clip,'boundary_uncertain':uncertain,'score':score,'warnings':warnings,'vad_analysis_gain_db':gains[ch]})
    selected=[]
    for candidate in sorted(candidates,key=lambda x:x['score'],reverse=True):
        if any(candidate['channel']==old['channel'] and max(0,min(candidate['end'],old['end'])-max(candidate['start'],old['start']))>.7*min(candidate['duration_seconds'],old['duration_seconds']) for old in selected):continue
        selected.append(candidate)
        if len(selected)>=3:break
    if not selected:
        raise NeedsInput('충분한 발화가 있는 참조 구간을 찾지 못했습니다. 무음·잘림·잡음이 적은 샘플이나 다른 구간을 지정해 주세요.',{'source':info,'selection':'more_audio'})
    for index,candidate in enumerate(selected):
        candidate['index']=index;candidate['file']=f'candidate-{index}.wav'
        extract(source,root/candidate['file'],candidate['start'],candidate['end'],candidate['track'],candidate['channel'],check)
    return {'source':info,'candidates':selected,'analysis':'Silero VAD CPU; contiguous segments; original gain retained','scanned_start':offset,'scanned_duration':duration}
