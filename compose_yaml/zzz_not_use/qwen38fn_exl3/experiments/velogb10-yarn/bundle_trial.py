"""Compare the existing Talk EXL3 bundle with an isolated Velo replacement.

Uses cached images/weights, a private Talk database/configuration, and restores
every initially stopped service. No production configuration is edited.
"""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

import yaml
from compare import stream, color_image

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[3]
SOURCE=Path('/home/edp1096/.cache/model-download-jobs/velogb10-yarn-20261009/source')
MODEL=Path('/home/edp1096/.cache/huggingface/hub/models--alesha-pro--Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6/snapshots/3b585c458f9fcf3322e76cff2c635c2cb81c5869')
LLM='sparktalk-qwen38fn_exl3'
AUX=['sparktalk-qwim-mmh3','sparktalk-nemotron-asr','sparktalk-qwen3-tts','sparktalk-embedding']
OWN=[LLM,*AUX,'sparktalk-extra-media']
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def http(base,path,data=None,timeout=1800,raw=False,headers=None):
    req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json',**(headers or {})})
    with OPENER.open(req,timeout=timeout) as r:
        value=r.read()
        return (value,dict(r.headers)) if raw else json.loads(value)

def wait(base,path,seconds=900):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:return http(base,path,timeout=3)
        except (OSError,ValueError):time.sleep(1)
    raise TimeoutError(base+path)

def identities(names=AUX):
    ds=json.loads(subprocess.check_output(['docker','inspect',*names]))
    return {d['Name'].lstrip('/'):{'id':d['Id'],'pid':d['State']['Pid'],'running':d['State']['Running'],'oom':d['State']['OOMKilled']} for d in ds}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--candidate-1m',action='store_true')
    ap.add_argument('--only',choices=['both','exllama','velo'],default='both')
    ap.add_argument('--noswap',action='store_true');ap.add_argument('--managed-start',action='store_true')
    ap.add_argument('--talk-binary',type=Path,default=WORKSPACE/'util/talk/dist/sparktalk-linux-arm64')
    ap.add_argument('--context',type=int,choices=[524288,1048576],default=1048576)
    ap.add_argument('--reference-container',default=LLM);ap.add_argument('--reference-port',type=int,default=18002)
    ap.add_argument('--full-context',action='store_true');ap.add_argument('--extended-aux',action='store_true');a=ap.parse_args()
    owned=[a.reference_container,*AUX,'sparktalk-extra-media']
    R=a.root;results={};velo=None;talk=None;logs=[]
    initial=identities(owned)
    if any(x['running'] for x in initial.values()):
        raise RuntimeError('Trial requires the listed model services and Extra Media initially stopped')
    (R/'initial-identities.json').write_text(json.dumps(initial,indent=2))
    def phase(label):
        (R/'phase').write_text(label);print(label,flush=True)
        if (R/'memory-floor.json').exists():raise RuntimeError('System available memory floor reached')
    def record(out,name,fn):
        t=time.monotonic()
        try:v=fn();v={'ok':True,'wall_seconds':time.monotonic()-t,'result':v}
        except Exception as e:
            v={'ok':False,'wall_seconds':time.monotonic()-t,'error':repr(e)}
            if isinstance(e,urllib.error.HTTPError):v['response']=e.read().decode(errors='replace')[:5000]
        (out/(name+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2));print(name,json.dumps(v,ensure_ascii=False)[:350],flush=True)
        return v
    def start_talk(out,base):
        cfg=yaml.safe_load((WORKSPACE/'util/talk/dist/sparktalk.yaml').read_text())
        cfg['server']={'listen_addr':'127.0.0.1:18586','database':str(out/'talk.db')}
        cfg['runtime']['auto_start']=False;cfg['runtime']['bundle']='qwen38fn_exl3';cfg['runtime']['active_bundle']='qwen38fn_exl3'
        cfg['runtime']['catalog']['network']['enabled']=False
        for b in cfg['runtime']['catalog']['bundles']:
            if b['id']=='qwen38fn_exl3':
                b['context_tokens']=a.context
                b['description']=b.get('description','').replace('1M','512K' if a.context==524288 else '1M')
        for x in cfg['runtime']['catalog']['components']:
            if x['id']=='qwen38fn_exl3':x.update(endpoint=base,health_url=base+'/health',controller='external',auto_address=False)
        cfg['model'].update(endpoint=base,default_model='qwen38fn_exl3',reasoning_effort='none')
        cfg['context'].update(output_auto=False,output_reserve=2048)
        cfg['image']['enabled']=True
        path=out/'sparktalk.yaml';path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));path.chmod(0o600)
        log=(out/'talk.log').open('w');logs.append(log)
        proc=subprocess.Popen([str(a.talk_binary)],cwd=out,stdout=log,stderr=subprocess.STDOUT)
        for _ in range(60):
            if proc.poll() is not None:raise RuntimeError('Private Talk failed to start; see talk.log')
            try:http('http://127.0.0.1:18586','/api/health',timeout=2);break
            except OSError:time.sleep(1)
        else:raise TimeoutError('Private Talk startup')
        actual=http('http://127.0.0.1:18586','/api/config')
        assert actual['model']['endpoint']==base,(actual['model']['endpoint'],base)
        assert actual['context']['window_tokens']==a.context,actual['context']
        return proc
    def talk_chat(out,name,prompt,tools=False):
        base='http://127.0.0.1:18586';session=http(base,'/api/sessions',{'title':'[isolated bundle audit] '+name,'model':'qwen38fn_exl3'})
        body={'session_id':session['id'],'content':prompt,'model':'qwen38fn_exl3','tools_enabled':tools}
        raw,_=http(base,'/api/chat',body,raw=True)
        (out/(name+'.sse')).write_bytes(raw)
        msgs=http(base,'/api/sessions/'+session['id']+'/messages');(out/(name+'-messages.json')).write_text(json.dumps(msgs,ensure_ascii=False,indent=2))
        attachments=[x for m in msgs for x in m.get('attachments',[])]
        saved=[]
        for x in attachments:
            if x.get('mime') not in ['image/png','video/mp4']:continue
            data,_=http(base,x['url'],raw=True)
            p=out/(name+('.mp4' if x['mime']=='video/mp4' else '.png'));p.write_bytes(data);saved.append(str(p))
        return {'session_id':session['id'],'saved':saved,'text':[m.get('content') for m in msgs if m.get('role')=='assistant'],'events':raw.decode(errors='replace')[-2500:]}
    def speech(out):
        pcm,headers=http('http://127.0.0.1:18586','/api/tts/speech',{'text':'안녕하세요. 오늘 영상은 가로 팔백육십사, 세로 사백팔십입니다. 초당 이십사 프레임으로 재생합니다.'},raw=True)
        rate=int(next(v for k,v in headers.items() if k.lower()=='x-audio-sample-rate'))
        assert len(pcm)>1000
        p=out/'speech.wav'
        with wave.open(str(p),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
        return {'bytes':len(pcm),'rate':rate,'duration_seconds':len(pcm)/(2*rate),'file':str(p)}
    def transcribe(out):
        boundary='audit'+uuid.uuid4().hex
        body=('--'+boundary+'\r\nContent-Disposition: form-data; name="audio"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+(out/'speech.wav').read_bytes()+('\r\n--'+boundary+'--\r\n').encode()
        req=urllib.request.Request('http://127.0.0.1:18586/api/asr/transcribe',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
        with OPENER.open(req,timeout=1800) as r:d=json.load(r)
        assert d.get('text'),d
        return d
    try:
        for label,base in [('exllama','http://127.0.0.1:'+str(a.reference_port)),('velo','http://127.0.0.1:19330')]:
            if a.only!='both' and label!=a.only:continue
            out=R/label;out.mkdir(exist_ok=False);results[label]={}
            phase(label+'-llm-startup')
            if label=='exllama':
                subprocess.run(['docker','start',a.reference_container],check=True,stdout=subprocess.PIPE)
                wait(base,'/health')
                card=http(base,'/v1/model');params=card['parameters']
                assert params['max_seq_len']==a.context and params['cache_size']==a.context and params['cache_mode']=='Q8' and params['use_vision'],card
                (out/'capacity.json').write_text(json.dumps(card,indent=2))
                code="import importlib.util,json;from pathlib import Path;s=importlib.util.spec_from_file_location('closed','/opt/sparktalk-qwen38fn_exl3/release_weight_cache.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps(m.release_cache(Path('/runtime/models/qwen38fn_exl3'))))"
                advice=subprocess.run(['docker','exec',a.reference_container,'python','-c',code],check=True,stdout=subprocess.PIPE,text=True)
                (out/'weight-cache-release.json').write_text(json.dumps({'ok':True,'result':json.loads(advice.stdout)},indent=2))
            else:
                cmd=['taskset','-c','5-9,15-19',str(SOURCE/'target/release/gb10_inference'),'--server','--model-dir',str(MODEL),'--host','127.0.0.1','--port','19330','--model-name','qwen38fn_exl3','--max-seq-len','1048576','--rope-yarn-factor','4','--max-batch','1','--ple-ram','ssd','--kv-cache','q8','--exl3-mtp-k','3','--draft-confidence','0','--prefix-cache','on','--prefix-ckpt-mem-gb','4','--prefill-chunk','2048','--tune-table','off','--reasoning-effort','none','--thinking','off','--exl3-pdl','0','--exl3-mtp-head-n','0']
                cmd[cmd.index('--max-seq-len')+1]=str(a.context)
                cmd[cmd.index('--rope-yarn-factor')+1]=str(a.context/262144)
                if a.noswap:
                    cmd=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit=velo-talk-bundle-noswap','-p','MemoryMax=112G','-p','MemorySwapMax=0','-p','WorkingDirectory='+str(SOURCE),'-p','Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu',*cmd]
                (out/'command.json').write_text(json.dumps(cmd,indent=2));log=(out/'server.log').open('w');logs.append(log)
                env=os.environ.copy();env['LD_LIBRARY_PATH']='/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
                velo=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pid=velo.pid
                if a.noswap:
                    for _ in range(60):
                        p=subprocess.run(['systemctl','--user','show','velo-talk-bundle-noswap','-p','MainPID','--value'],capture_output=True,text=True)
                        if p.stdout.strip().isdigit() and int(p.stdout)>0:pid=int(p.stdout);break
                        time.sleep(.2)
                    else:raise RuntimeError('No Velo systemd MainPID')
                (R/'velo.pid').write_text(str(pid));wait(base,'/health')
                cg=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split('0::')[1].strip().lstrip('/')
                (out/'cgroup-limits.json').write_text(json.dumps({k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.current','memory.swap.current']},indent=2))
                spec=importlib.util.spec_from_file_location('closed_weights',HERE.parents[1]/'release_weight_cache.py')
                cache_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache_module)
                record(out,'weight-cache-release',lambda:cache_module.release_cache(MODEL,Path('/proc')/str(pid)))
            record(out,'model-info',lambda:http(base,'/v1/models'))
            (out/'llm-ready-memory.json').write_text((R/'memory.jsonl').read_text().splitlines()[-1])
            phase(label+'-aux-startup')
            if a.managed_start:
                talk=start_talk(out,base)
                record(out,'managed-start-request',lambda:http('http://127.0.0.1:18586','/api/runtime/bundles/qwen38fn_exl3/start',{}))
                for _ in range(1200):
                    state=http('http://127.0.0.1:18586','/api/runtime')
                    if state['operation']['state']!='running':break
                    time.sleep(1)
                (out/'managed-start.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
                if state['operation']['state']!='complete':raise RuntimeError('Managed bundle startup failed: '+state['operation'].get('error',state['operation'].get('detail','')))
            else:
                for n,port,path in [('sparktalk-qwim-mmh3',8730,'/health'),('sparktalk-nemotron-asr',8693,'/ready'),('sparktalk-qwen3-tts',8692,'/ready'),('sparktalk-embedding',8701,'/health')]:
                    subprocess.run(['docker','start',n],check=True,stdout=subprocess.PIPE);health=wait('http://127.0.0.1:'+str(port),path)
                    (out/(n+'-health.json')).write_text(json.dumps(health,indent=2))
            before=identities();(out/'identities-before.json').write_text(json.dumps(before,indent=2))
            if talk is None:talk=start_talk(out,base)
            phase(label+'-resident-idle');time.sleep(5)
            record(out,'talk-chat',lambda:talk_chat(out,'chat','17×23의 결과를 숫자만 답해라.'))
            phase(label+'-speech');record(out,'tts',lambda:speech(out))
            phase(label+'-transcription');record(out,'asr',lambda:transcribe(out))
            phase(label+'-embedding');record(out,'embedding',lambda:{'dimensions':len(http('http://127.0.0.1:8701','/v1/encode',{'task':'query','inputs':[{'text':'이미지와 음성 모델을 메모리에 계속 유지하는 설정'}]})['data'][0]['embedding'])})
            if a.extended_aux:
                for name,package,test,env_values in [
                    ('diarization','./internal/asr','TestLiveDiarization',{'TALK_DIAR_LIVE_AUDIO':'/home/edp1096/.cache/model-download-jobs/nemo-asr-v020-deploy-20261008/human-two-speakers.wav','TALK_DIAR_LIVE_ENDPOINT':'http://127.0.0.1:8693','TALK_DIAR_REQUIRE_MULTI':'1'}),
                    ('hybrid-retrieval','./internal/server','TestRetrievalLiveKoreanGPU',{'SPARKTALK_EMBEDDING_LIVE_ENDPOINT':'http://127.0.0.1:8701','SPARKTALK_RETRIEVAL_REPORT':str(out/'retrieval-quality.json')})]:
                    phase(label+'-'+name)
                    def run_aux(package=package,test=test,env_values=env_values,name=name):
                        env=os.environ.copy();env.update(env_values)
                        p=subprocess.run(['go','test',package,'-run','^'+test+'$','-count=1','-v'],cwd=WORKSPACE/'util/talk',env=env,capture_output=True,text=True,timeout=180)
                        (out/(name+'.log')).write_text(p.stdout+p.stderr)
                        assert p.returncode==0,p.stdout+p.stderr
                        return {'returncode':p.returncode,'log':p.stdout}
                    record(out,name,run_aux)
            phase(label+'-image')
            results[label]['image']=record(out,'talk-image',lambda:talk_chat(out,'image','image_generate를 딱 한 번 호출하여 다음 프롬프트와 seed 42, 1024x1024로 이미지를 생성하고 첨부해라. 다른 도구는 필요 없다.\nA cute penguin standing on snowy ice, detailed wildlife photograph, soft morning sunlight.',True))
            phase(label+'-video')
            results[label]['video']=record(out,'talk-video',lambda:talk_chat(out,'video','video_generate를 딱 한 번 호출하여 다음 프롬프트와 seed 42로 음성을 포함한 영상을 생성하고 첨부해라. 다른 도구는 필요 없다.\nA cute penguin waddles across snowy ice toward the camera, soft morning sunlight, gentle wind and quiet footsteps, steady cinematic camera.',True))
            phase(label+'-normal-speed');speeds=[]
            for i in range(3):
                for kind,prompt in [('korean','한국어로 SQLite FTS5와 의미 검색을 함께 사용하는 이유를 구체적인 예시 두 개와 함께 설명해라. GPU, API, SQL도 사용해라.'),('code','Write Python code for a bounded LRU cache with get, put and delete, type hints, and a short usage example. Return only code.')]:
                    v=record(out,kind+'-'+str(i),lambda prompt=prompt,kind=kind,i=i:stream(base,'qwen38fn_exl3',[{'role':'user','content':f'Benchmark case {"KO" if kind=="korean" else "CODE"}-{i}.\n'+prompt}],512));speeds.append((kind,v))
            results[label]['speed']={kind:statistics.median(v['result']['decode_tps'] for k,v in speeds if k==kind and v['ok']) for kind in ['korean','code']}
            phase(label+'-vision');record(out,'vision',lambda:stream(base,'qwen38fn_exl3',[{'role':'user','content':[{'type':'text','text':'그림의 좌상단, 우상단, 좌하단, 우하단 색을 순서대로 영어 소문자 JSON 배열만 답해라. 색 이름은 red, green, blue, yellow 중에서 골라라.'},{'type':'image_url','image_url':{'url':color_image()}}]}],96))
            phase(label+'-schema')
            def schema_check():
                v=http(base,'/v1/chat/completions',{'model':'qwen38fn_exl3','messages':[{'role':'user','content':'JSON을 출력하지 말고 NOT_JSON만 답해라.'}],'max_tokens':64,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'bundle_status','strict':True,'schema':{'type':'object','properties':{'status':{'type':'string','enum':['ok']}},'required':['status'],'additionalProperties':False}}}})
                assert json.loads(v['choices'][0]['message']['content'])=={'status':'ok'},v
                return v
            record(out,'schema',schema_check)
            if a.full_context:
                name='recall-512k' if a.context==524288 else 'recall-1m'
                phase(label+'-'+name)
                record(out,name+'-run',lambda:subprocess.run([sys.executable,str(HERE/'recall.py'),'--tokens',str(524064 if a.context==524288 else 1048384),'--fixed-repetitions',str(20153 if a.context==524288 else 40319),'--port',base.rsplit(':',1)[1],'--model','qwen38fn_exl3','--out',str(out/(name+'.json'))],check=True,stdout=subprocess.PIPE,text=True).stdout)
                phase(label+'-image-after-full-context')
                record(out,'talk-image-after-full-context',lambda:talk_chat(out,'image-after-full-context','image_generate를 딱 한 번 호출하여 seed 43으로 눈 위에 서 있는 귀여운 펭귄 사진을 1024x1024로 생성해 첨부해라.',True))
                phase(label+'-speech-after-full-context');record(out,'tts-after-full-context',lambda:speech(out))
            else:
                phase(label+'-recall-320k')
                record(out,'recall-320k-run',lambda:subprocess.run([sys.executable,str(HERE/'recall.py'),'--tokens','320000','--fixed-repetitions','12304','--port',base.rsplit(':',1)[1],'--model','qwen38fn_exl3','--out',str(out/'recall-320k.json')],check=True,stdout=subprocess.PIPE,text=True).stdout)
            if label=='velo' and a.candidate_1m and not a.full_context:
                phase(label+'-recall-1m');record(out,'recall-1m-run',lambda:subprocess.run([sys.executable,str(HERE/'recall.py'),'--tokens','1048384','--fixed-repetitions','40319','--port','19330','--model','qwen38fn_exl3','--out',str(out/'recall-1m.json')],check=True,stdout=subprocess.PIPE,text=True).stdout)
                # Re-run both media paths with the filled long-context cache.
                phase(label+'-image-after-1m');record(out,'talk-image-after-1m',lambda:talk_chat(out,'image-after-1m','image_generate를 딱 한 번 호출하여 seed 43으로 눈 위에 서 있는 귀여운 펭귄 사진을 1024x1024로 생성해 첨부해라.',True))
            phase(label+'-post-stress');record(out,'post-stress-chat',lambda:talk_chat(out,'post-stress','17×23의 결과를 숫자만 답해라.'))
            after=identities();results[label]['core_identities_unchanged']=before==after;(out/'identities-after.json').write_text(json.dumps(after,indent=2))
            events=subprocess.check_output(['docker','exec','sparktalk-qwim-mmh3','cat','/job/events.jsonl']);(out/'media-events.jsonl').write_bytes(events)
            record(out,'media-runtime',lambda:http('http://127.0.0.1:8730','/v1/runtime/memory'))
            talk.terminate();talk.wait(timeout=25);talk=None
            phase(label+'-cleanup');subprocess.run(['docker','stop','--time','10',*AUX,'sparktalk-extra-media'],check=True,stdout=subprocess.PIPE)
            if label=='exllama':subprocess.run(['docker','stop','--time','20',a.reference_container],check=True,stdout=subprocess.PIPE)
            else:
                if a.noswap:subprocess.run(['systemctl','--user','stop','velo-talk-bundle-noswap'],check=True)
                else:os.killpg(velo.pid,signal.SIGTERM)
                velo.wait(timeout=25);velo=None
            (R/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        phase('completed')
    finally:
        if talk is not None:talk.terminate();talk.wait(timeout=25)
        if velo is not None and velo.poll() is None:
            if a.noswap:subprocess.run(['systemctl','--user','stop','velo-talk-bundle-noswap'])
            else:os.killpg(velo.pid,signal.SIGTERM)
            velo.wait(timeout=25)
        subprocess.run(['docker','stop','--time','10',*owned],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        for f in logs:f.close()
        (R/'monitor.stop').touch()
        (R/'cleanup.json').write_text(json.dumps(identities(owned),indent=2))

if __name__=='__main__':main()
