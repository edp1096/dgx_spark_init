"""Temporarily test this NVFP4 set's ASR/embedding residency, restoring states."""
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import trial


def snapshot():
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3}
    return dict(time=time.time(),available_bytes=mem['MemAvailable'],free_bytes=mem['MemFree'],
                swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],gpu_processes=trial.command('nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'))


def http(port,path,body=None):
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        with trial.OPENER.open(req,timeout=120) as r:
            raw=r.read()
            try:return json.loads(raw)
            except json.JSONDecodeError:return raw.decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'{port}{path}: HTTP {error.code}: {error.read().decode()}') from error


def wait(name,port,path):
    for attempt in range(180):
        try:return http(port,path)
        except (OSError,ValueError):
            state=json.loads(trial.command('docker','inspect',name))[0]['State']
            if not state['Running']:raise RuntimeError(f'{name} exited during residency test')
            time.sleep(1)
    raise TimeoutError(name)


def run(label):
    result={'before':snapshot()};started=[]
    identity=json.loads(trial.command('docker','inspect',trial.NAME))[0]
    result['llm_container_id']=identity['Id'];result['llm_init_pid']=identity['State']['Pid']
    try:
        model=trial.request('/get_server_info')['model_path']
        result['checkpoint_cache_reclaim']=trial.command('docker','exec',trial.NAME,'python3',
                                                       '/experiment/reclaim_checkpoint_cache.py',model)
        result['after_checkpoint_reclaim']=snapshot()
        for name,port,path in [('sparktalk-nemotron-asr',8693,'/ready'),('sparktalk-embedding',8701,'/health')]:
            state=json.loads(trial.command('docker','inspect',name))[0]['State']
            if not state['Running']:
                trial.command('docker','start',name);started.append(name)
            result[name]=wait(name,port,path)
        encoded=http(8701,'/v1/encode',{'task':'query','inputs':[{'text':'현재 GPU 모델의 메모리 사용량은?'}]})
        result['embedding_response_keys']=list(encoded)
        result['embedding_memory']=http(8701,'/v1/runtime/memory')
        assert result['embedding_memory']['ready'] and result['embedding_memory']['device']=='cuda'
        result['resident']=snapshot()
        asr_pids={line.split()[0] for line in trial.command('docker','top','sparktalk-nemotron-asr','-eo','pid').splitlines()[1:]}
        gpu_pids={line.split(',')[0].strip() for line in result['resident']['gpu_processes'].splitlines()}
        assert asr_pids & gpu_pids, 'ASR did not retain a GPU context'
        result['asr_gpu_pids']=sorted(asr_pids & gpu_pids)
        response=trial.stream(dict(messages=[{'role':'user','content':'17×23을 숫자만 답해라.'}],max_tokens=32))
        assert response['text'].strip()=='391'
        result['llm_response']=response
        current=json.loads(trial.command('docker','inspect',trial.NAME))[0]
        assert current['Id']==result['llm_container_id'] and current['State']['Pid']==result['llm_init_pid']
        result['after_request']=snapshot()
    finally:
        for name in reversed(started):trial.command('docker','stop','-t','20',name)
        result['restored']=snapshot();result['temporarily_started']=started
        (trial.OUT/label/'resident.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('RESIDENT',label,result['resident']['available_bytes']/2**30,flush=True)
