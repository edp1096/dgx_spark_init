#!/usr/bin/env python3
"""Bounded two-node A/B trial; uses a private profile, never edits production .env."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CONTAINER = 'deepseek-v4-flash-vllm-dspark-1'
PROBE = '''import json,pathlib,subprocess
m={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in pathlib.Path('/proc/meminfo').read_text().splitlines())}
p=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
print(json.dumps(dict(boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),available=m['MemAvailable'],swap=m['SwapTotal']-m['SwapFree'],swap_free=m['SwapFree'],pressure=pathlib.Path('/proc/pressure/memory').read_text().strip(),gpu=p.stdout.strip())))'''


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def http(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=240)


def container_metadata(container, ssh=None):
    command=['docker','inspect',container] if ssh is None else ssh+['docker inspect '+shlex.quote(container)]
    row=json.loads(subprocess.check_output(command,text=True,timeout=20))[0]
    allowed={'MAX_MODEL_LEN','MAX_NUM_SEQS','MAX_NUM_BATCHED_TOKENS','GPU_MEMORY_UTILIZATION','GPU_MEMORY_UTILIZATION_TEXT',
             'KV_CACHE_MEMORY','DFLASH_TOKENS','ADAPTIVE_SPECULATIVE_TOKENS_WINDOW','CUDA_GRAPH_CAPTURE_SIZES',
             'DSPARK_ENABLE_DSPARK_SWA_PREFIX','DSPARK_ENABLE_DSML_RECOVERY','DSPARK_ENABLE_C128A_PREFILL_CACHE'}
    variables=dict(x.split('=',1) for x in row['Config']['Env'] if '=' in x)
    return dict(image_id=row['Image'],image=row['Config']['Image'],started=row['State']['StartedAt'],
                restart=row['HostConfig']['RestartPolicy']['Name'],environment={k:v for k,v in variables.items() if k in allowed})


def tasks():
    tool = {'type':'function','function':{'name':'get_weather','description':'Look up weather for a city on a date.',
        'parameters':{'type':'object','properties':{'city':{'type':'string'},'date':{'type':'string'}},'required':['city','date']}}}
    records = '\n'.join(f'기록 {i:04d}: 부서=자료관리, 보관함={i%97:02d}, 확인상태=완료.' for i in range(700))
    base = [
        dict(id='repeat_json60', prompt='JSON 객체만 출력하세요. numbers에는 1부터 60까지 모든 정수를 순서대로, sum에는 그 합을 넣으세요.', expected={'numbers':list(range(1,61)),'sum':1830}),
        dict(id='repeat_long_facts', prompt=records+'\n최종 확정: 담당자=박민수, 장소=서울, 수량=37, 단가=1200원. 앞의 기록은 무시하고 확정 내용을 사용하세요. JSON만 출력: 담당자, 장소, 수량, 총액. 총액은 수량×단가입니다.', expected={'담당자':'박민수','장소':'서울','수량':37,'총액':44400}),
        dict(id='korean_arithmetic', prompt='가상 재고: 시작 230개, 입고 47개, 출고 89개, 불량 폐기 6개. JSON만 출력하세요: remaining은 최종 재고, boxes는 한 상자 12개 기준 필요한 최소 상자 수.', expected={'remaining':182,'boxes':16}),
        dict(id='tool_auto', prompt='2026-09-10 서울 날씨를 get_weather로 조회하세요. 실제 조회 결과를 추측하지 마세요.', tools=[tool], tool_choice='auto', expected_tool={'city':'서울','date':'2026-09-10'}),
        dict(id='tool_required', prompt='2026-09-11 부산 날씨를 조회하세요.', tools=[tool], tool_choice={'type':'function','function':{'name':'get_weather'}}, expected_tool={'city':'부산','date':'2026-09-11'}),
        dict(id='tool_none', prompt='도구를 쓰지 말고 JSON {"status":"not_requested"}만 출력하세요.', tools=[tool], tool_choice='none', expected={'status':'not_requested'}),
    ]
    return base


def request_task(url, task, repeat):
    payload = dict(messages=[{'role':'system','content':'지시와 JSON 필드명을 정확히 따르세요.'},{'role':'user','content':task['prompt']}],
                   temperature=0, max_tokens=768, stream=True, stream_options={'include_usage':True},
                   chat_template_kwargs=task.get('chat_template_kwargs',{'thinking':False}))
    for key in ['tools','tool_choice']:
        if key in task: payload[key]=task[key]
    started = time.monotonic(); first=None; content=''; reasoning=''; calls={}; usage={}; finish=None
    with http(url+'/v1/chat/completions',payload) as response:
        for line in response:
            if not line.startswith(b'data: '): continue
            raw=line[6:].strip()
            if raw==b'[DONE]': break
            event=json.loads(raw)
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if any(delta.get(k) for k in ['content','reasoning_content','reasoning','tool_calls']) and first is None:first=time.monotonic()-started
                content+=delta.get('content') or '';reasoning+=delta.get('reasoning_content') or delta.get('reasoning') or ''
                for call in delta.get('tool_calls',[]):
                    target=calls.setdefault(call['index'],{'name':'','arguments':''})
                    for k in target:target[k]+=call.get('function',{}).get(k) or ''
                if choice.get('finish_reason'):finish=choice['finish_reason']
    elapsed=time.monotonic()-started
    try:
        if 'expected_tool' in task:
            passed=len(calls)==1 and calls[0]['name']=='get_weather' and json.loads(calls[0]['arguments'])==task['expected_tool'] and finish=='tool_calls'
        else:
            cleaned=content.strip()
            if cleaned.startswith('```'):cleaned='\n'.join(cleaned.splitlines()[1:-1])
            passed=json.loads(cleaned)==task['expected'] and not calls and finish=='stop'
    except (ValueError,KeyError):passed=False
    return dict(id=task['id'],repeat=repeat,content=content,reasoning=reasoning,tool_calls=calls,usage=usage,
                elapsed_s=elapsed,ttft_s=first,finish_reason=finish,passed=passed)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--mode',choices=['baseline','candidate','dsml'],required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    private=out/'.runtime';private.mkdir();os.chmod(private,0o700)
    source=(ROOT/'.env').read_text()
    overrides=dict(MAX_MODEL_LEN='131072',MAX_NUM_SEQS='2',MAX_NUM_BATCHED_TOKENS='4096',GPU_MEMORY_UTILIZATION_TEXT='0.78',
                   DSPARK_RESTART_POLICY='no',DSPARK_ENABLE_C128A_PREFILL_CACHE='1',
                   DSPARK_ENABLE_DSPARK_SWA_PREFIX=str(int(args.mode=='candidate')),DSPARK_ENABLE_DSML_RECOVERY=str(int(args.mode!='baseline')))
    profile=private/'profile.env';profile.write_text(source+'\n'+'\n'.join(k+'='+v for k,v in overrides.items())+'\n');os.chmod(profile,0o600)
    env=os.environ.copy();env['ENV_FILE']=str(profile)
    worker=subprocess.check_output(['bash','-c','source "$1"; printf "%s" "$WORKER_HOST"','bash',str(profile)],text=True)
    port=subprocess.check_output(['bash','-c','source "$1"; printf "%s" "${VLLM_PORT:-8888}"','bash',str(profile)],text=True)
    url='http://127.0.0.1:'+port
    def probe(host):
        cmd=['python3','-c',PROBE] if host=='head' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',worker,'python3 -c '+shlex.quote(PROBE)]
        return json.loads(subprocess.check_output(cmd,text=True,timeout=20))
    baseline={h:probe(h) for h in ['head','worker']}
    if any(r['available']<100*2**30 for r in baseline.values()):raise RuntimeError('Insufficient idle memory for isolated trial')
    save(out/'plan.json',dict(mode=args.mode,overrides=overrides,baseline=baseline,started=datetime.datetime.now().astimezone().isoformat(),
                             guard=dict(min_available_gib=8,min_swap_free_gib=2,benchmark_swap_growth_mib=512),
                             limitations=['128K cap and concurrency 2; does not qualify 1M/concurrency 6.','CPU parser fixtures cover malformed DSML; live model may not emit malformed DSML.']))
    save(out/'tasks.json',tasks());abort=[];done=threading.Event();phase=['loading'];process=None;ready_swap={}
    def emergency_stop():
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
        commands=[['docker','stop','-t','5',CONTAINER],['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',worker,'docker stop -t 5 '+CONTAINER]]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda cmd:subprocess.run(cmd,capture_output=True,timeout=25),commands))
    def monitor():
        errors=0
        with (out/'memory.jsonl').open('w') as f:
            while not done.is_set():
                try:
                    with concurrent.futures.ThreadPoolExecutor(2) as pool:values=dict(zip(['head','worker'],pool.map(probe,['head','worker'])))
                    f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),phase=phase[0],hosts=values))+'\n');f.flush()
                    errors=0
                    for host,r in values.items():
                        swap_base=ready_swap.get(host,baseline[host]['swap'])
                        reason=('boot changed' if r['boot']!=baseline[host]['boot'] else
                                'available below 8 GiB' if r['available']<8*2**30 else
                                'swap free below 2 GiB' if r['swap_free']<2*2**30 else
                                'benchmark swap growth above 512 MiB' if host in ready_swap and r['swap']-swap_base>512*2**20 else '')
                        if reason:
                            abort.append(host+': '+reason);emergency_stop();return
                except Exception as exc:
                    errors+=1
                    if errors>=3:abort.append('monitor failed: '+str(exc));emergency_stop();return
                done.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    stop=ROOT/'upstream/stop-deepseek-v4-flash-dspark.sh';start=ROOT/'upstream/start-deepseek-v4-flash-dspark.sh'
    try:
        with (out/'start.log').open('w') as log:
            process=subprocess.Popen(['bash',str(start)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+3600
            while process.poll() is None:
                if abort or time.monotonic()>deadline:raise RuntimeError(str(abort or 'startup timeout'))
                time.sleep(2)
            if process.returncode:raise RuntimeError('launcher failed: '+str(process.returncode))
        if abort:raise RuntimeError(str(abort))
        save(out/'containers.json',dict(head=container_metadata(CONTAINER),worker=container_metadata(CONTAINER,['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',worker])))
        ready_swap.update({host:probe(host)['swap'] for host in ['head','worker']})
        with http(url+'/metrics') as response:(out/'metrics-before.txt').write_bytes(response.read())
        phase[0]='benchmark';rows=[]
        for repeat in range(3):
            for task in tasks():
                if abort:raise RuntimeError(str(abort))
                row=request_task(url,task,repeat);rows.append(row);save(out/'responses.json',rows)
                print(f"{len(rows)}/18 {task['id']} pass={row['passed']} elapsed={row['elapsed_s']:.2f}s",flush=True)
        with http(url+'/metrics') as response:(out/'metrics.txt').write_bytes(response.read())
        save(out/'summary.json',dict(requests=len(rows),passed=sum(r['passed'] for r in rows),elapsed_s=sum(r['elapsed_s'] for r in rows),completed=True))
    except BaseException as exc:
        save(out/'error.json',dict(error=repr(exc),aborted=abort));raise
    finally:
        phase[0]='stopping'
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
        with (out/'stop.log').open('w') as log:
            result=subprocess.run(['bash',str(stop)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        done.set();thread.join(timeout=25)
        save(out/'cleanup.json',dict(exit_code=result.returncode,final={h:probe(h) for h in ['head','worker']}))


if __name__=='__main__':main()
