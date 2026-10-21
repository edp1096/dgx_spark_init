from pathlib import Path
import sys
import json,subprocess,time,urllib.request,traceback,hashlib
root=Path(__file__).parent
port=18433
name='gemma31-spec-opt-trial'
cases=json.loads((root/'cases-with-long.json').read_text())

def generate(case):
 body={'input_ids':case['input_ids'],'sampling_params':{'temperature':0,'sampling_seed':42,'max_new_tokens':case['max_new_tokens'],'ignore_eos':case.get('ignore_eos',False),'skip_special_tokens':False},'stream':True}
 req=urllib.request.Request(f'http://127.0.0.1:{port}/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 start=time.monotonic();ttft=None;first_tokens=0;last=None;events=[]
 with urllib.request.urlopen(req,timeout=900) as response:
  for line in response:
   if not line.startswith(b'data:'):continue
   data=line[5:].strip()
   if data==b'[DONE]':break
   obj=json.loads(data)
   if 'error' in obj:raise RuntimeError(obj['error'])
   m=obj.get('meta_info',{});n=m.get('completion_tokens',0)
   elapsed=time.monotonic()-start
   if ttft is None and n>0:ttft=elapsed;first_tokens=n
   events.append({'s':elapsed,'tokens':n});last=obj
 total=time.monotonic()-start
 assert last and ttft is not None,(last,ttft)
 m=last['meta_info'];n=m['completion_tokens'];p=m['prompt_tokens']
 row={'name':case['name'],'kind':case['kind'],'input_sha256':hashlib.sha256(json.dumps(case['input_ids']).encode()).hexdigest(),'text':last['text'],'prompt_tokens':p,'completion_tokens':n,'ttft_s':ttft,'first_chunk_tokens':first_tokens,'total_s':total,'request_tok_s':n/total,'tg_tok_s':(n-first_tokens)/(total-ttft) if n>first_tokens else None,'pp_ttft_tok_s':p/ttft,'meta_info':m,'events':events,'replacement_characters':last['text'].count('\ufffd')}
 if m.get('prefill_finished_time',0)>m.get('forward_entry_time',0)>0:
  row['prefill_s']=m['prefill_finished_time']-m['forward_entry_time'];row['pp_engine_tok_s']=p/row['prefill_s']
 return row


reference=Path('/home/edp1096/.cache/model-download-jobs/gemma31-assistant-bpw-20261021-174321')
arms=sys.argv[1:]
assert arms
for arm in arms:
 out=root/arm;out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():continue
 if (out/'error.txt').exists():raise RuntimeError('Retained failed arm; use a new tag or explicitly resolve error: '+arm)
 bits=next((part[1:] for part in arm.split('-') if part.startswith('b') and part[1:].isdigit()),'4')
 depth=int(arm.rsplit('-',1)[-1]) if arm!='off' else 0
 common=['--model-path','ignored','--served-model-name','gemma31-bpw-test','--trust-remote-code','--host','0.0.0.0','--port','30000','--tp-size','1','--context-length','262144','--max-total-tokens','262144','--swa-full-tokens-ratio','0.006','--mem-fraction-static','0.55','--attention-backend','triton','--chunked-prefill-size','512','--disable-prefill-cuda-graph','--cuda-graph-max-bs-decode','1','--kv-cache-dtype','auto','--disable-radix-cache','--max-running-requests','1','--skip-server-warmup','--enable-metrics']
 if arm!='off':common+=['--speculative-algorithm','DFLASH','--speculative-draft-model-path','ignored','--speculative-draft-model-quantization','unquant','--speculative-num-draft-tokens',str(depth+1),'--speculative-eagle-topk','1']
 cmd=['docker','run','--rm','--name',name,'--gpus','all','--ipc','host','--cpuset-cpus','5-9,15-19','--memory','100g','--memory-swap','100g','-p',f'127.0.0.1:{port}:30000','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONUNBUFFERED=1','-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-e','TORCHINDUCTOR_CACHE_DIR=/root/.cache/sglang/inductor','-e','DRAFT_KIND=dflash','-e',f'ASSISTANT_BODY_BITS={bits}','-e',f'SPARKTALK_GEMMA_ASSISTANT_EXL3={0}','-v',str(root)+':/job:ro','-v',str(reference)+':'+str(reference)+':ro','-e','GEMMA31_VERIFY_SPLITKV=1','-v',str(root/'verify-work/triton_backend_sm121.py')+':/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py:ro','-v',str(root/'verify-work/verify_splitkv_sm121.py')+':/sgl-workspace/sglang/python/sglang/kernels/ops/attention/verify_splitkv.py:ro','-v',str(root/'recipe/launch_exl3.py')+':/opt/gemma4/launch_exl3.py:ro','-v',str(root/'runtime')+':/root/.cache/sglang','gemma31-assistant-bpw:trial',*common]
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('START',arm,flush=True)
 server=None
 try:
  with (out/'server.log').open('w') as log:
   server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   deadline=time.monotonic()+1200
   while True:
    if server.poll() is not None:raise RuntimeError(f'{arm} server exited {server.returncode}')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:assert response.status==200
     break
    except Exception:
     if time.monotonic()>deadline:raise TimeoutError('startup')
     time.sleep(2)
   print('READY',arm,flush=True)
   (out/'warmup.json').write_text(json.dumps(generate(dict(cases[0],max_new_tokens=64,ignore_eos=True)),ensure_ascii=False,indent=2)+'\n')
   rows=[]
   for case in cases:
    row=generate(case);row['arm']=arm;rows.append(row)
    (out/(case['name']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('arm','name','prompt_tokens','completion_tokens','ttft_s','tg_tok_s')}),flush=True)
   (out/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
   print('COMPLETE',arm,flush=True)
 except Exception:
  (out/'error.txt').write_text(traceback.format_exc());raise
 finally:
  subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  if server:server.wait(timeout=30)
print('ALL_ARMS_COMPLETE',flush=True)
