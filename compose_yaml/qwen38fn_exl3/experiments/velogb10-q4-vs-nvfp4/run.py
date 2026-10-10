"""Matched Q4/Velo vs production-weight NVFP4/SGLang 1M trials.

Only isolated LLMs and initially stopped auxiliary containers are used. All
production settings, model files and existing Talk processes are preserved.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'velogb10-yarn'
sys.path.insert(0,str(OLD))
from compare import stream, color_image
import coding
import vision_quality

ROOT=Path('/home/edp1096/.cache/model-download-jobs/velogb10-q4-vs-nvfp4-20261010')
WORKSPACE=Path('/home/edp1096/workspace/dgx_spark_init')
SOURCE=Path('/home/edp1096/.cache/model-download-jobs/velogb10-yarn-20261009/source')
Q4=Path('/home/edp1096/.cache/huggingface/hub/models--alesha-pro--Huihui-Qwen3.8-Flash-Next-abliterated-exl3-4bit-hq_h6_ng6/snapshots/e884d3e5e38d53e3b50a59e02d1b7a4cb1e5d75e')
NV_MODEL='edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
AUX=['sparktalk-nemotron-asr','sparktalk-embedding','sparktalk-extra-media','sparktalk-extra-documents']
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
IMAGE=json.loads((ROOT/'nvfp4-container.json').read_text())[0]['Image']


def command(*cmd,**kw):
    return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,**kw).strip()


def http(base,path,data=None,timeout=1800):
    q=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with OPENER.open(q,timeout=timeout) as r:
        raw=r.read()
        if not raw:return {'status':r.status}
        try:return json.loads(raw)
        except ValueError:return {'status':r.status,'body':raw.decode()}


def phase(name):
    (ROOT/'phase').write_text(name);print(name,flush=True)
    if (ROOT/'memory-floor.json').exists() and 'cleanup' not in name and 'completed' not in name:raise RuntimeError('Memory floor reached')


def wait(base, path='/health', process=None):
    for _ in range(1500):
        if process is not None and process.poll() is not None:raise RuntimeError('Candidate process exited; see server.log')
        try:return http(base,path,timeout=2)
        except (OSError,ValueError):time.sleep(1)
    raise TimeoutError(base+path)


def record(out,name,fn):
    t=time.monotonic()
    try:r=dict(ok=True,result=fn())
    except Exception as e:
        r=dict(ok=False,error=repr(e))
        if isinstance(e,urllib.error.HTTPError):r['body']=e.read().decode(errors='replace')[:6000]
    r['seconds']=time.monotonic()-t
    (out/(name+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(name,str(r)[:350],flush=True)
    return r


def closed_cache(model,pid):
    files=sorted(model.glob('model-*-of-00009.safetensors'))
    assert len(files)==9
    resolved={str(p.resolve()) for p in files}
    active=set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            maps=(process/'maps').read_text()
            active.update(p for p in resolved if p in maps)
            for fd in (process/'fd').iterdir():
                try:
                    v=str(fd.resolve(strict=True))
                    if v in resolved:active.add(v)
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    assert not active,active
    for p in files:
        with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return dict(files=9,ple_untouched=True,policy='closed, unmapped checkpoint files only')


def ids(names=AUX):
    ds=json.loads(command('docker','inspect',*names))
    return {d['Name'].lstrip('/'):{'id':d['Id'],'pid':d['State']['Pid'],'running':d['State']['Running']} for d in ds}


def auxiliary(out):
    command('docker','start','sparktalk-extra-media');wait('http://127.0.0.1:8690')
    for name,port,path in [('sparktalk-nemotron-asr',8693,'/ready'),('sparktalk-embedding',8701,'/health')]:
        command('docker','start',name);record(out,name+'-health',lambda port=port,path=path:wait(f'http://127.0.0.1:{port}',path))
    record(out,'embedding',lambda:http('http://127.0.0.1:8701','/v1/encode',{'task':'query','inputs':[{'text':'메모리에 상주하는 모델의 사용량은?'}]}))
    record(out,'embedding-memory',lambda:http('http://127.0.0.1:8701','/v1/runtime/memory'))
    for name,package,test,env in [
        ('diarization','./internal/asr','TestLiveDiarization',{'TALK_DIAR_LIVE_AUDIO':'/home/edp1096/.cache/model-download-jobs/nemo-asr-v020-deploy-20261008/human-two-speakers.wav','TALK_DIAR_LIVE_ENDPOINT':'http://127.0.0.1:8693','TALK_DIAR_REQUIRE_MULTI':'1'}),
        ('hybrid-retrieval','./internal/server','TestRetrievalLiveKoreanGPU',{'SPARKTALK_EMBEDDING_LIVE_ENDPOINT':'http://127.0.0.1:8701','SPARKTALK_RETRIEVAL_REPORT':str(out/'retrieval-quality.json')})]:
        def run_test(package=package,test=test,env=env,name=name):
            p=subprocess.run(['go','test',package,'-run','^'+test+'$','-count=1','-v'],cwd=WORKSPACE/'util/talk',env=os.environ|env,capture_output=True,text=True,timeout=180)
            (out/(name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
            return dict(returncode=p.returncode,log=p.stdout)
        record(out,name,run_test)


def benchmark(out,base,model):
    values=[]
    prompts={'korean':'한국어로 SQLite FTS5와 의미 검색을 함께 사용하는 이유를 구체적인 예시 두 개와 함께 설명해라. GPU, API, SQL도 사용해라.',
             'code':'Write Python code for a bounded LRU cache with get, put and delete, type hints, and a short usage example. Return only code.'}
    record(out,'warmup',lambda:stream(base,model,[{'role':'user','content':'List all integers from 1 to 1000 separated by commas.'}],128))
    for i in range(3):
        for kind,prompt in prompts.items():
            r=record(out,kind+'-'+str(i),lambda kind=kind,prompt=prompt,i=i:stream(base,model,[{'role':'user','content':f'Benchmark case {"KO" if kind=="korean" else "CODE"}-{i}.\n'+prompt}],512))
            values.append((kind,r))
    import statistics
    return {k:statistics.median(r['result']['decode_tps'] for kind,r in values if kind==k and r['ok']) for k in prompts}


def vision_long(base,model):
    text='This entry records a routine maintenance check. All tests passed.\n'*10000
    parts=[{'type':'text','text':text+'\n그림의 좌상단, 우상단, 좌하단, 우하단 색을 순서대로 영어 소문자 JSON 배열만 답해라. 색 이름은 red, green, blue, yellow 중에서 골라라.'},{'type':'image_url','image_url':{'url':color_image()}}]
    r=stream(base,model,[{'role':'user','content':parts}],96)
    r['pass']=json.loads(r['text'])==['red','green','blue','yellow'];return r


def tests(out,label,base,model):
    phase(label+'-aux-startup');auxiliary(out);before=ids();(out/'resident-before.json').write_text(json.dumps(before,indent=2))
    phase(label+'-normal-speed');speed=benchmark(out,base,model)
    phase(label+'-suite')
    p=subprocess.run([sys.executable,str(OLD/'compare.py'),'--base',base,'--model',model,'--out',str(out/'common-suite')],capture_output=True,text=True,timeout=1800)
    (out/'common-suite.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
    phase(label+'-coding');record(out,'coding',lambda:coding.run(base,model,out/'coding',IMAGE))
    phase(label+'-vision-quality');record(out,'vision-quality',lambda:vision_quality.run(base,model,out/'vision-quality'))
    phase(label+'-schema')
    def schema():
        r=stream(base,model,[{'role':'user','content':'JSON을 출력하지 말고 NOT_JSON만 답해라.'}],64,response_format={'type':'json_schema','json_schema':{'name':'status','strict':True,'schema':{'type':'object','properties':{'status':{'type':'string','enum':['ok']}},'required':['status'],'additionalProperties':False}}})
        r['pass']=json.loads(r['text'])=={'status':'ok'};return r
    record(out,'schema',schema)
    phase(label+'-recall-1m')
    if ':19333' in base:
        for attempt in range(100):
            try:http(base,'/flush_cache',{});break
            except urllib.error.HTTPError as e:
                if e.code!=400:raise
                time.sleep(.2)
        else:raise RuntimeError('Could not flush SGLang before cold full-context request')
    def recall():
        p=subprocess.run([sys.executable,str(OLD/'recall.py'),'--tokens','1048384','--fixed-repetitions','40319','--port',base.rsplit(':',1)[1],'--model',model,'--out',str(out/'recall-1m.json')],capture_output=True,text=True,timeout=7200)
        (out/'recall.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
        return json.loads((out/'recall-1m.json').read_text())
    record(out,'recall-1m-run',recall)
    phase(label+'-vision-260k');record(out,'vision-long',lambda:vision_long(base,model))
    phase(label+'-post-stress');record(out,'post-stress-chat',lambda:stream(base,model,[{'role':'user','content':'17×23의 결과를 숫자만 답해라.'}],32))
    after=ids();(out/'resident-after.json').write_text(json.dumps(after,indent=2))
    assert before==after,(before,after)
    return dict(speed=speed,resident_ids_unchanged=True)


def main():
    velo=None;log=None;results={}
    initial=ids()
    if any(d['running'] for d in initial.values()):
        assert (ROOT/'owned-resume.json').exists() and initial==json.loads((ROOT/'owned-resume.json').read_text()),initial
    try:
        base='http://127.0.0.1:19333';out=ROOT/'nvfp4-1m'
        wait(base);info=http(base,'/get_server_info');(out/'server-info.json').write_text(json.dumps(info,indent=2))
        assert info['context_length']==1048576 and info['max_total_num_tokens']==1048576,info
        reclaim=(WORKSPACE/'compose_yaml/qwen38_fn_sglang/experiments/radixark-fp8-hybrid/reclaim_checkpoint_cache.py').read_text()
        record(out,'closed-cache',lambda:command('docker','exec','nvfp4-q4-comparison','python3','-c',reclaim,'/hf/'+NV_MODEL))
        phase('waiting-for-verified-q4-download')
        while True:
            status=json.loads((ROOT/'download-status.json').read_text())
            if status.get('complete'):break
            if status['errors']:raise RuntimeError(status['errors'])
            time.sleep(10)
        # Downloads and checksum reads finish before the measured A/B phases.
        results['nvfp4-1m']=tests(out,'nvfp4-1m',base,NV_MODEL)
        (out/'server-info-after.json').write_text(json.dumps(http(base,'/get_server_info'),indent=2))
        (out/'server.log').write_text(command('docker','logs','nvfp4-q4-comparison'))
        command('docker','stop','-t','20',*AUX,'nvfp4-q4-comparison')
        phase('q4-1m-startup');out=ROOT/'q4-1m';out.mkdir()
        cmd=['taskset','-c','5-9,15-19',str(SOURCE/'target/release/gb10_inference'),'--server','--model-dir',str(Q4),'--host','127.0.0.1','--port','19334','--model-name','q4-exl3-audit','--max-seq-len','1048576','--rope-yarn-factor','4','--max-batch','1','--ple-ram','ssd','--kv-cache','q8','--exl3-mtp-k','3','--draft-confidence','0','--prefix-cache','on','--prefix-ckpt-mem-gb','4','--prefill-chunk','2048','--tune-table','off','--reasoning-effort','none','--thinking','off','--exl3-pdl','0','--exl3-mtp-head-n','0']
        (out/'command.json').write_text(json.dumps(cmd,indent=2))
        cmd=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit=velo-q4-comparison','-p','MemoryMax=112G','-p','MemorySwapMax=0','-p','WorkingDirectory='+str(SOURCE),'-p','Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu',*cmd]
        log=(out/'server.log').open('w');velo=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(60):
            pid=command('systemctl','--user','show','velo-q4-comparison','-p','MainPID','--value')
            if pid.isdigit() and int(pid):break
            time.sleep(.2)
        (ROOT/'velo.pid').write_text(pid);wait('http://127.0.0.1:19334',process=velo)
        record(out,'closed-cache',lambda:closed_cache(Q4,int(pid)))
        results['q4-1m']=tests(out,'q4-1m','http://127.0.0.1:19334','q4-exl3-audit')
        (ROOT/'primary-results.json').write_text(json.dumps(results,indent=2))
    finally:
        phase('primary-cleanup')
        subprocess.run(['docker','stop','-t','20',*AUX,'nvfp4-q4-comparison'],capture_output=True)
        if velo is not None:
            subprocess.run(['systemctl','--user','stop','velo-q4-comparison'],capture_output=True)
            try:velo.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(velo.pid,signal.SIGTERM)
        if log is not None:log.close()
        (ROOT/'primary-ended.json').write_text(json.dumps(dict(results=results,restored=ids(),finished=time.time()),indent=2))
        phase('primary-completed')


if __name__=='__main__':main()
