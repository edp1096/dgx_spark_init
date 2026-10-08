#!/usr/bin/env python3
"""Compare fixed and native adaptive DFlash depth on the pinned Entrpi image."""
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ds_trial',ROOT.parent/'ds4fve_vllm/bench/run_update_trial.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
save=shared.save


def main():
    if 'adaptive_speculative_tokens_window' not in (ROOT/'entrypoint.sh').read_text():
        raise SystemExit('Archived experiment: apply bench/adaptive-attempt.patch to reproduce. The pinned DFlash2 runtime rejects the adaptive arm; no server started.')
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--mode',choices=['baseline','adaptive'],required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    private=out/'.runtime';private.mkdir();os.chmod(private,0o700)
    overrides=dict(MAX_MODEL_LEN='131072',MAX_NUM_SEQS='2',MAX_NUM_BATCHED_TOKENS='4096',KV_CACHE_MEMORY='8589934592',
                   ADAPTIVE_SPECULATIVE_TOKENS_WINDOW='8' if args.mode=='adaptive' else '0',
                   CUDA_GRAPH_CAPTURE_SIZES='"1 2 3 4 5 6 7 8 10 12 14 16"')
    profile=private/'profile.env';profile.write_text((ROOT/'.env').read_text()+'\n'+'\n'.join(k+'='+v for k,v in overrides.items())+'\n');os.chmod(profile,0o600)
    env=os.environ.copy();env['RUNTIME_ENV_FILE']=str(profile)
    def variable(name):
        return subprocess.check_output(['bash','-c','source "$1"; printf "%s" "${!2}"','bash',str(profile),name],text=True)
    worker=variable('WORKER_USER')+'@'+variable('WORKER_LAN_IP');remote_dir=variable('REMOTE_COMPOSE_DIR');url='http://127.0.0.1:'+variable('API_PORT')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',worker]
    # runtime.sh synchronizes the selected profile to the worker; restore its
    # exact previous private configuration after the isolated trial.
    remote_env=remote_dir+'/.env'
    previous=subprocess.run(ssh+['cat '+shlex.quote(remote_env)],capture_output=True)
    if previous.returncode:raise RuntimeError('Cannot preserve worker configuration')
    (private/'worker.env').write_bytes(previous.stdout);os.chmod(private/'worker.env',0o600)
    def probe(host):
        command=['python3','-c',shared.PROBE] if host=='head' else ssh+['python3 -c '+shlex.quote(shared.PROBE)]
        return json.loads(subprocess.check_output(command,text=True,timeout=20))
    before={h:probe(h) for h in ['head','worker']}
    if any(r['available']<100*2**30 for r in before.values()):raise RuntimeError('Another workload is using memory')
    cases=shared.tasks()
    cases.extend([
        dict(id='prose_cache',prompt='한국어로 600자 안팎으로 설명하세요. 작은 웹서비스에서 캐시를 도입할 때 TTL, 무효화, 캐시 스탬피드의 의미와 대응을 각각 설명하고 오래된 데이터 허용 여부에 따른 선택 기준을 제시하세요.',review=True),
        dict(id='prose_search',prompt='한국어로 600자 안팎으로 설명하세요. 정렬된 배열의 이진 탐색과 정렬되지 않은 배열의 선형 탐색을 비교하세요. 탐색 시간복잡도와 정렬 전처리 비용, 질의 횟수에 따른 선택 기준을 구체적으로 설명하세요.',review=True),
    ])
    for case in cases:case['chat_template_kwargs']={'enable_thinking':False}
    save(out/'tasks.json',cases);save(out/'plan.json',dict(mode=args.mode,overrides=overrides,baseline=before,image=variable('GLM53_IMAGE'),
        started=datetime.datetime.now().astimezone().isoformat(),limitations=['Native Entrpi acceptance controller, not MiaAI EMA patch.','128K cap, two sequences, 8 GiB KV pool; production 524K has not been qualified.']))
    abort=[];done=threading.Event();phase=['loading'];ready_swap={}
    def stop_containers():
        commands=[['docker','stop','-t','5','glm53-head'],ssh+['docker stop -t 5 glm53-worker']]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(lambda cmd:subprocess.run(cmd,capture_output=True,timeout=25),commands))
    def monitor():
        failures=0
        with (out/'memory.jsonl').open('w') as f:
            while not done.is_set():
                try:
                    with concurrent.futures.ThreadPoolExecutor(2) as pool:values=dict(zip(['head','worker'],pool.map(probe,['head','worker'])))
                    f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),phase=phase[0],hosts=values))+'\n');f.flush();failures=0
                    for host,r in values.items():
                        active_swap=host in ready_swap and r['swap']-ready_swap[host]>512*2**20
                        if r['boot']!=before[host]['boot'] or r['available']<8*2**30 or r['swap_free']<2*2**30 or active_swap:
                            abort.append(host+': reboot or memory guard');stop_containers();return
                except Exception as exc:
                    failures+=1
                    if failures>=3:abort.append(str(exc));stop_containers();return
                done.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        with (out/'start.log').open('w') as log:
            result=subprocess.run(['bash',str(ROOT/'runtime.sh'),'start'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:raise RuntimeError('launcher failed')
        # Bench runs do not auto-resurrect after a host reset.
        subprocess.run(['docker','update','--restart=no','glm53-head'],check=True,capture_output=True)
        subprocess.run(ssh+['docker update --restart=no glm53-worker'],check=True,capture_output=True)
        deadline=time.monotonic()+3600
        while True:
            if abort:raise RuntimeError(str(abort))
            try:
                with shared.http(url+'/health') as response:
                    if response.status==200:break
            except Exception:pass
            state=subprocess.check_output(['docker','inspect','glm53-head','--format','{{.State.Running}}'],text=True).strip()
            if state!='true':raise RuntimeError('head container exited before readiness; inspect saved server logs')
            if time.monotonic()>deadline:raise RuntimeError('readiness timeout')
            time.sleep(5)
        ready_swap.update({h:probe(h)['swap'] for h in ['head','worker']})
        save(out/'containers.json',dict(head=shared.container_metadata('glm53-head'),worker=shared.container_metadata('glm53-worker',ssh)))
        save(out/'warmup.json',shared.request_task(url,cases[2],-1))
        with shared.http(url+'/metrics') as response:(out/'metrics-before.txt').write_bytes(response.read())
        phase[0]='benchmark';rows=[]
        for repeat in range(3):
            for task in cases:
                if abort:raise RuntimeError(str(abort))
                call=dict(task)
                if call.get('review'):call['expected']={}
                row=shared.request_task(url,call,repeat)
                if task.get('review'):row['passed']=None;row['requires_review']=True
                rows.append(row);save(out/'responses.json',rows)
                print(f"{len(rows)}/24 {row['id']} pass={row['passed']} elapsed={row['elapsed_s']:.2f}s",flush=True)
        with shared.http(url+'/metrics') as response:(out/'metrics.txt').write_bytes(response.read())
        save(out/'summary.json',dict(requests=len(rows),passed=sum(r['passed'] is True for r in rows),pending_review=sum(r['passed'] is None for r in rows),completed=True))
    except BaseException as exc:save(out/'error.json',dict(error=repr(exc),aborted=abort));raise
    finally:
        for host in ['head','worker']:
            cmd=['docker','logs','glm53-head'] if host=='head' else ssh+['docker logs glm53-worker']
            with (out/(host+'-server.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=30)
        phase[0]='stopping'
        with (out/'stop.log').open('w') as log:result=subprocess.run(['bash',str(ROOT/'runtime.sh'),'stop'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        done.set();thread.join(timeout=25)
        restore=subprocess.run(ssh+['umask 077; cat > '+shlex.quote(remote_env)],input=previous.stdout,capture_output=True)
        save(out/'cleanup.json',dict(exit_code=result.returncode,worker_env_restored=restore.returncode==0,final={h:probe(h) for h in ['head','worker']}))


if __name__=='__main__':main()
