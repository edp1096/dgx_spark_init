"""Isolated, sequential A/B trials; never changes the managed Talk containers."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import urllib.error

OUT = Path('/home/edp1096/.cache/model-download-jobs/radixark-fp8-hybrid-20261009')
HERE = Path(__file__).resolve().parent
NAME = 'radixark-hybrid-audit'
PORT = 19303
BASE = f'http://127.0.0.1:{PORT}'
SOURCE = '/hf/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
MODEL = 'radixark-hybrid-audit'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def request(path, data=None, timeout=1800):
    req = urllib.request.Request(BASE+path, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    for attempt in range(100):
        try:
            with OPENER.open(req, timeout=timeout) as response:
                raw=response.read()
                try:return json.loads(raw)
                except json.JSONDecodeError:return raw.decode()
        except urllib.error.HTTPError as error:
            if path!='/flush_cache' or error.code!=400 or attempt==99:raise
            # Final SSE delivery can precede scheduler cleanup; never measure a failed flush.
            time.sleep(.2)


def start(label):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1', PORT))
    source = json.loads(command('docker', 'inspect', 'sglang-qwen38-fn-radixark'))[0]
    args = source['Args'][1:]
    def option(flag, value):
        args[args.index(flag)+1] = value
    option('--host', '127.0.0.1')
    option('--port', str(PORT))
    option('--served-model-name', MODEL)
    hybrid = label.startswith('hybrid')
    if hybrid:
        option('--model-path', SOURCE+'-fp8hybrid')
    cmd = ['docker', 'run', '-d', '--init', '--name', NAME, '--network', 'host', '--gpus', 'all',
           '--memory', '112g', '--memory-swap', '112g', '--cpuset-cpus', '5-9,15-19',
           '--shm-size', '16g', '--ulimit', 'memlock=-1:-1', '--security-opt', 'seccomp=unconfined']
    env = [s for s in source['Config']['Env'] if s.startswith(
        ('SGLANG_', 'SPARKTALK_', 'HF_', 'B12X_', 'TRITON_', 'TORCHINDUCTOR_', 'PYTORCH_', 'MAX_JOBS='))]
    env += ['PYTHONUNBUFFERED=1', 'PYTHONPATH=/experiment:/opt/qad-tp1',
            'RADIXARK_FP8_HYBRID_PATH='+ (SOURCE+'-fp8hybrid' if hybrid else '')]
    if 'opt' in label:
        env += ['RADIXARK_LITERAL_GUARD=1','RADIXARK_FAST_LOADER=1']
    if hybrid:
        args += ['--fp8-gemm-backend','flashinfer_cutlass']
    for value in env:
        cmd += ['-e', value]
    for value in [
        '/home/edp1096/.cache/huggingface:/hf:ro',
        '/home/edp1096/.local/share/sparktalk/cache/sglang-flash-next-qad:/root/.cache/sglang',
        '/home/edp1096/.local/share/sparktalk/cache/sglang-flash-next-qad/ple:/ple',
        str(HERE)+':/experiment:ro',
        str(HERE.parents[1]/'radixark_launch.py')+':/opt/radixark/launch.py:ro',
        str(HERE.parents[1]/'radixark_launch.py')+':/opt/qad-tp1/qad_loader.py:ro',
    ]:
        cmd += ['-v', value]
    cmd += ['--entrypoint', 'python3', source['Image'], '/experiment/launch.py', *args]
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'command.json').write_text(json.dumps(cmd, indent=2))
    (folder/'started.json').write_text(json.dumps(dict(wall_time=time.time(), label=label)))
    print(command(*cmd), flush=True)


def monitor(label):
    folder = OUT / label
    seen_healthy = False
    gpu_checked = 0
    with (folder/'memory.jsonl').open('a') as handle:
        while True:
            info = json.loads(command('docker', 'inspect', NAME))[0]
            row = dict(time=time.time(), state=info['State']['Status'], oom=info['State']['OOMKilled'])
            mem = {line.split(':')[0]: int(line.split()[1])*1024
                   for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3}
            row['host'] = {k: mem[k] for k in ['MemAvailable', 'MemFree', 'SwapTotal', 'SwapFree']}
            vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
            row['swap_io_pages'] = {k: int(vm[k]) for k in ['pswpin','pswpout']}
            if time.monotonic()-gpu_checked >= 10:
                gpu_checked=time.monotonic()
                try:
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],
                                       text=True,capture_output=True,timeout=3)
                    row['gpu_processes']=gpu.stdout.strip()
                except subprocess.TimeoutExpired:row['gpu_query_timeout']=True
            pid = info['State']['Pid']
            if pid:
                cg = Path('/sys/fs/cgroup') / Path(f'/proc/{pid}/cgroup').read_text().split('::',1)[1].strip().lstrip('/')
                row['cgroup'] = {}
                for key in ['memory.current','memory.peak','memory.swap.current','memory.events']:
                    try: row['cgroup'][key] = (cg/key).read_text().strip()
                    except FileNotFoundError: pass
                stat = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
                row['cgroup_stat'] = {k:int(stat[k]) for k in ['anon','file','shmem','kernel']}
            if not seen_healthy and pid:
                try:
                    with OPENER.open(BASE+'/health',timeout=5) as response:
                        seen_healthy = response.status==200
                    if seen_healthy:
                        ready = dict(seconds=time.time()-json.loads((folder/'started.json').read_text())['wall_time'])
                        (folder/'ready.json').write_text(json.dumps(ready))
                        (folder/'server-info-ready.json').write_text(json.dumps(request('/get_server_info'),indent=2))
                        print('READY',label,ready,flush=True)
                except Exception: pass
            row['healthy'] = seen_healthy
            handle.write(json.dumps(row)+'\n');handle.flush()
            if row['state'] != 'running': break
            time.sleep(1)
    (folder/'docker.log').write_text(command('docker','logs',NAME))


def stream(body):
    body.update(model=MODEL,stream=True,stream_options={'include_usage':True},temperature=0,
                chat_template_kwargs={'enable_thinking':False})
    req = urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    start = time.monotonic(); first=None; chunks=[];usage=None;done=False;finish=None;calls={}
    with OPENER.open(req,timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            payload=line[6:].strip()
            if payload==b'[DONE]':done=True;break
            event=json.loads(payload)
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                text=delta.get('content') or ''
                if text or delta.get('tool_calls'):
                    if first is None:first=time.monotonic()-start
                chunks.append(text)
                for call in delta.get('tool_calls') or []:
                    key=call['index']; acc=calls.setdefault(key,{'name':'','arguments':''})
                    f=call.get('function',{});acc['name']+=f.get('name') or '';acc['arguments']+=f.get('arguments') or ''
                finish=choice.get('finish_reason') or finish
    elapsed=time.monotonic()-start
    assert done and usage,(done,usage)
    return dict(text=''.join(chunks),calls=list(calls.values()),usage=usage,finish=finish,
                seconds=elapsed,ttft=first,decode_tps=usage['completion_tokens']/max(elapsed-(first or 0),.001))


def bench(label):
    folder=OUT/label; results=[]
    def sample(name,prompt,max_tokens=256,**extra):
        result=stream(dict(messages=[{'role':'user','content':prompt}],max_tokens=max_tokens,**extra))
        result['name']=name;results.append(result)
        (folder/'bench.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print(name,round(result['ttft'],3),round(result['decode_tps'],2),result['finish'],flush=True)
        return result
    sample('warmup','1부터 100까지 정수를 쉼표로 구분하여 나열해라. 설명 없이 숫자만 출력해라.',128)
    for rep in range(3):
        request('/flush_cache')
        sample(f'korean_{rep}','한국어로 SQLite FTS5와 의미 검색을 함께 사용하는 이유를 구체적인 예시 두 개와 함께 설명해라. 약어 GPU, API, SQL도 사용해라.',256)
        request('/flush_cache')
        sample(f'code_{rep}','Write Python code for a bounded LRU cache with get, put and delete, type hints, and a short usage example. Return only code.',256)
    for n in [1024,4096]:
        request('/flush_cache')
        prompt=('이 기록은 배경 자료다. 항목에는 특별한 의미가 없다.\n'*n
                +'\n1부터 1000까지 정수를 쉼표로 나열해라. 설명과 생략 없이 계속 출력해라.')
        sample(f'fresh_{n}',prompt,256)
        sample(f'cached_{n}',prompt,256)
    tasks=[
        ('arithmetic','17×23의 결과를 숫자만 답해라.',lambda x:x.strip()=='391'),
        ('units','가로 864, 세로 480, 초당 24프레임인 영상의 해상도와 프레임률을 864×480, 24 FPS 형식으로만 답해라.',lambda x:'864' in x and '480' in x and '24' in x),
        ('japanese','「今日は良い天気です」を韓国語に翻訳してください。翻訳だけを書いてください。',lambda x:'날씨' in x and ('좋' in x or '맑' in x)),
        ('chinese','把「明天上午九点开会」翻译成韩语。只输出译文。',lambda x:'내일' in x and ('아홉' in x or '9' in x) and '회의' in x),
        ('json','정확히 {"name":"펭귄","count":3} 형태의 JSON 객체만 출력해라. Markdown 없이.',lambda x:json.loads(x)=={'name':'펭귄','count':3}),
        ('logic','상자 A에는 빨간 공만, B에는 파란 공만 있다. 빨간 공 하나를 B로 옮겼다. 이제 B에 있는 공의 색을 모두 말해라.',lambda x:'빨간' in x and '파란' in x),
        ('recall','회의 기록: 프로젝트 암호는 청록펭귄이며 마감은 10월 23일이다.\n'+'보관 기록: 오늘은 점검일이다.\n'*2048+'\n프로젝트 암호와 마감일만 답해라.',lambda x:'청록펭귄' in x and '23' in x),
    ]
    for name,prompt,check in tasks:
        r=sample(name,prompt,128)
        try:r['pass']=bool(check(r['text']))
        except Exception:r['pass']=False
    tool={'type':'function','function':{'name':'move_room','description':'Move the current conversation into a folder.',
            'parameters':{'type':'object','properties':{'folder':{'type':'string'},'confirmed':{'type':'boolean'}},'required':['folder','confirmed']}}}
    for rep in range(3):
        r=sample(f'tool_{rep}','현재 대화방을 뉴스 폴더로 옮겨라. move_room을 호출하고 folder는 뉴스, confirmed는 true로 지정해라.',128,tools=[tool])
        try:r['pass']=len(r['calls'])==1 and r['calls'][0]['name']=='move_room' and json.loads(r['calls'][0]['arguments'])=={'folder':'뉴스','confirmed':True}
        except Exception:r['pass']=False
    # Force the model to describe XML syntax while tools are offered; do not execute calls.
    sample('literal_tool_xml','도구는 호출하지 마라. move_room 호출 XML 예제를 ```xml 코드 블록 안에 작성하고, 블록 뒤에 설명 완료라고 적어라.',192,tools=[tool],tool_choice='auto')
    literal='<tool_call>\n<function=move_room>\n<parameter=folder>뉴스</parameter>\n<parameter=confirmed>true</parameter>\n</function>\n</tool_call>'
    sample('literal_exact_xml','문서 작성 작업이다. 실제 도구 호출은 금지한다. 다음 문자열을 수정 없이 ```xml 코드 블록 안에 복사하고 블록 뒤에 설명 완료라고 적어라:\n'+literal,192,tools=[tool],tool_choice='auto')
    (folder/'bench.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (folder/'server-info-final.json').write_text(json.dumps(request('/get_server_info'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','monitor','bench','stop']);parser.add_argument('label')
    args=parser.parse_args()
    if args.action=='start':start(args.label)
    elif args.action=='monitor':monitor(args.label)
    elif args.action=='bench':bench(args.label)
    else:
        command('docker','stop','-t','30',NAME)
        (OUT/args.label/'docker.log').write_text(command('docker','logs',NAME))
        (OUT/args.label/'container-final.json').write_text(command('docker','inspect',NAME))
        time.sleep(2)
        command('docker','rm',NAME)
