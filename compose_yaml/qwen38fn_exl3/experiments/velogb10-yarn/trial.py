"""Isolated Velo EXL3 YaRN trials; uses only the existing read-only checkpoint."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

DEFAULT_ROOT = Path('/home/edp1096/.cache/model-download-jobs/velogb10-yarn-20261009')
DEFAULT_MODEL = Path('/home/edp1096/.cache/huggingface/hub/models--alesha-pro--Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6/snapshots/3b585c458f9fcf3322e76cff2c635c2cb81c5869')
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    ap.add_argument('--label',default='yarn-1m')
    ap.add_argument('--context',type=int,default=1048576)
    ap.add_argument('--factor',type=float,default=4)
    ap.add_argument('--port',type=int,default=19304)
    ap.add_argument('--kv-cache',choices=['q8','fp8','f16','f32'],default='q8')
    ap.add_argument('--no-mtp',action='store_true')
    ap.add_argument('--prefix-cache',choices=['on','off'],default='off')
    ap.add_argument('--prefix-ckpt-mem-gb',type=float,default=0)
    ap.add_argument('--cpuset')
    ap.add_argument('--mtp-head-n',type=int)
    ap.add_argument('--max-batch',type=int,default=1)
    ap.add_argument('--pdl',choices=['0','1'])
    args=ap.parse_args()
    out=args.root/args.label
    base=f'http://127.0.0.1:{args.port}'
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('127.0.0.1',args.port))
    out.mkdir(parents=True,exist_ok=False)
    cmd=[str(args.root/'source/target/release/gb10_inference'),'--server','--model-dir',str(args.model),
         '--host','127.0.0.1','--port',str(args.port),'--model-name','velo-yarn-audit',
         '--max-seq-len',str(args.context),'--rope-yarn-factor',str(args.factor),'--max-batch',str(args.max_batch),
         '--ple-ram','ssd','--kv-cache',args.kv_cache,'--exl3-mtp-k','3','--draft-confidence','0',
         '--prefix-cache',args.prefix_cache,'--prefix-ckpt-mem-gb',str(args.prefix_ckpt_mem_gb),'--prefill-chunk','2048',
         '--tune-table','off','--reasoning-effort','none','--thinking','off']
    if args.no_mtp:cmd.append('--exl3-no-mtp')
    if args.pdl is not None:cmd.extend(['--exl3-pdl',args.pdl])
    if args.mtp_head_n is not None:
        if args.mtp_head_n < 0:raise ValueError('--mtp-head-n must be nonnegative; 0 means full vocabulary')
        cmd.extend(['--exl3-mtp-head-n',str(args.mtp_head_n)])
    if args.cpuset:cmd=['taskset','-c',args.cpuset,*cmd]
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
    start=time.monotonic(); stop=threading.Event()
    log=(out/'server.log').open('w')
    proc=subprocess.Popen(cmd,cwd=args.root/'source',stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    (out/'pid').write_text(str(proc.pid))
    def request(path,data=None,timeout=30):
        req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(req,timeout=timeout) as res:return json.loads(res.read())
    def monitor():
        with (out/'memory.jsonl').open('w') as f:
            while not stop.is_set() and proc.poll() is None:
                mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if len(l.split())>=3}
                vm=dict(l.split() for l in Path('/proc/vmstat').read_text().splitlines())
                row={'seconds':time.monotonic()-start,'host':{k:mem[k] for k in ['MemAvailable','MemFree','SwapTotal','SwapFree']},'swap_io_pages':{k:int(vm[k]) for k in ['pswpin','pswpout']}}
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text()
                    row['process']={l.split(':')[0]:int(l.split()[1])*1024 for l in status.splitlines() if l.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    row['gpu_processes']=gpu.stdout.strip()
                except (OSError,subprocess.TimeoutExpired):pass
                f.write(json.dumps(row)+'\n');f.flush()
                if mem['MemAvailable']<4*1024**3:
                    (out/'memory-floor-stop.json').write_text(json.dumps(row))
                    os.killpg(proc.pid,signal.SIGTERM);return
                stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        for _ in range(1200):
            if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}; see {out}/server.log')
            try:
                request('/health',timeout=2);break
            except Exception:time.sleep(1)
        else:raise TimeoutError('server boot timed out')
        (out/'ready.json').write_text(json.dumps({'seconds':time.monotonic()-start,'models':request('/v1/models')}))
        print('READY',args.label,round(time.monotonic()-start,2),flush=True)
        results=[]
        def sample(name,prompt,n=128,**extra):
            body={'model':'velo-yarn-audit','messages':[{'role':'user','content':prompt}],
                  'max_tokens':n,'temperature':0,'stream':True,'stream_options':{'include_usage':True},
                  'chat_template_kwargs':{'enable_thinking':False}}|extra
            req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            t0=time.monotonic();first=None;text=[];reason=[];usage=None;calls=[];done=False;finish=None
            with OPENER.open(req,timeout=7200) as res:
                for line in res:
                    if not line.startswith(b'data: '):continue
                    raw=line[6:].strip()
                    if raw==b'[DONE]':done=True;break
                    event=json.loads(raw)
                    usage=event.get('usage') or usage
                    for c in event.get('choices',[]):
                        d=c.get('delta',{});v=d.get('content') or '';r=d.get('reasoning_content') or ''
                        if (v or r or d.get('tool_calls')) and first is None:first=time.monotonic()-t0
                        text.append(v);reason.append(r);calls.extend(d.get('tool_calls') or []);finish=c.get('finish_reason') or finish
            r={'name':name,'text':''.join(text),'reasoning':''.join(reason),'calls':calls,'usage':usage,'finish':finish,
               'seconds':time.monotonic()-t0,'ttft':first,'done':done}
            if usage and first is not None:r['decode_tps']=max(0,usage['completion_tokens']-1)/max(r['seconds']-first,.001)
            results.append(r);(out/'responses.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(name,round(r['seconds'],2),usage,r['text'][:100],flush=True)
            return r
        sample('arithmetic','17×23의 결과를 숫자만 답해라.',32)
        sample('json','정확히 {"name":"펭귄","count":3} 형태의 JSON 객체만 출력해라. Markdown 없이.',64)
        sample('korean','한국어로 SQLite FTS5와 의미 검색을 함께 사용하는 이유를 예시 두 개와 함께 설명해라. GPU, API, SQL도 사용해라.',256)
        sample('code','Write Python code for a bounded LRU cache with get, put and delete, type hints, and a short usage example. Return only code.',256)
        sample('japanese','「今日は良い天気です」を韓国語に翻訳してください。翻訳だけを書いてください。',64)
        sample('chinese','把「明天上午九点开会」翻译成韩语。只输出译文。',64)
        (out/'short-complete').touch()
        # The parent can append authorized long-context trials while this isolated server stays available.
        while not (out/'stop').exists() and proc.poll() is None:time.sleep(1)
    finally:
        stop.set()
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        thread.join(timeout=10);log.close()
        (out/'exit.json').write_text(json.dumps({'code':proc.returncode,'seconds':time.monotonic()-start}))

if __name__=='__main__':main()
