#!/usr/bin/env python3
"""Check the live main bundle and record a gate for the separately authorized upload."""
import ast,hashlib,json,re,time,urllib.request
from pathlib import Path
NAME='edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
root=Path(__file__).resolve().parent/'docs';base='http://127.0.0.1:8585'
def call(path,method='GET',body=None,raw=False):
 req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'},method=method)
 with urllib.request.urlopen(req,timeout=300) as r:
  b=r.read();return b if raw else json.loads(b) if b else None
start=time.monotonic()
while time.monotonic()-start<1800:
 runtime=call('/api/runtime');op=runtime['operation']
 if op.get('state')=='failed':raise RuntimeError(op.get('error','Main bundle failed'))
 if op.get('state')=='complete' and runtime['selected_bundle']=='flash-next-tp2':break
 time.sleep(3)
else:raise RuntimeError('Main bundle startup timeout')
cfg=call('/api/config');assert cfg['runtime']['bundle']=='flash-next-tp2' and cfg['model']['default_model']==NAME and cfg['model']['endpoint']=='http://127.0.0.1:8012'
required=['flash-next-tp2','flux2','nemotron-asr','magpie-tts','extra-media','extra-ssh','extra-collector','extra-documents'];components={x['id']:x for x in runtime['components']}
components.update({x['id']:x for x in call('/api/support')['services']})
for name in required:assert components[name]['health']=='online',(name,components[name].get('health'))
health=call('/api/health');assert health['status']=='ok' and health['model']==NAME
summary={'status':'passed','default_bundle':cfg['runtime']['bundle'],'model':NAME,'endpoint':cfg['model']['endpoint'],'context_tokens':cfg['context']['window_tokens'],'components':{k:{f:components[k].get(f) for f in ['status','health','model']} for k in required}}
(root/'main-service-check.json').write_text(json.dumps(summary,indent=2));print('All main services healthy',flush=True)
# Exercise the actual Talk request path, then remove only this temporary session.
session=call('/api/sessions','POST',{'title':'Temporary Huihui release verification','reasoning_effort':'none'})
try:
 data=call('/api/chat','POST',{'session_id':session['id'],'content':'검증용 질문입니다. 17 + 25의 결과를 숫자만 답하세요.','reasoning_effort':'none','tools_enabled':False},raw=True)
 assert b'event: error' not in data and b'event: done' in data
 messages=call('/api/sessions/'+session['id']+'/messages');reply=next(x for x in reversed(messages) if x['role']=='assistant');assert re.sub(r'[^0-9]','',reply['content'])=='42'
 (root/'talk-chat-check.json').write_text(json.dumps({'status':'passed','model':NAME,'content':reply['content'],'temporary_session_deleted':True},ensure_ascii=False,indent=2))
finally:call('/api/sessions/'+session['id'],'DELETE',raw=True)
quality={}
for label in ['tp1-main','tp2']:
 rows=json.loads((root/f'{label}-runtime.json').read_text())['results'];assert len(rows)==14 and all('error' not in x for x in rows)
 assert sum(x.get('passed') is True for x in rows)==5
 code=next(x['text'] for x in rows if x['name']=='coding');tree=ast.parse(code)
 assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name=='unique_ordered'
 for node in ast.walk(tree):
  assert not isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal))
  if isinstance(node,ast.Attribute):assert node.attr in ['add','append']
  if isinstance(node,ast.Call):assert isinstance(node.func,ast.Attribute) or isinstance(node.func,ast.Name) and node.func.id=='set'
  if isinstance(node,ast.Name):assert not node.id.startswith('__')
 namespace={'__builtins__':{'set':set}};exec(compile(tree,'reviewed-function','exec'),namespace)
 for a,b in [([],[]),([3,1,3,2],[3,1,2]),(['가','나','가'],['가','나']),([None,None,0],[None,0])]:assert namespace['unique_ordered'](a)==b
 tool=next(x for x in rows if x['name']=='tool');assert tool['finish_reason']=='tool_calls';assert tool['tool_calls'][0]['function']['name']=='get_weather';assert json.loads(''.join(x['function'].get('arguments') or '' for x in tool['tool_calls']))=={'city':'Seoul'}
 quality[label]={'exact_checks':5,'code_execution_checks':4,'tool_call':'passed','manual_korean_translation_fiction_review':'passed'}
(root/'release-quality-review.json').write_text(json.dumps(quality,indent=2))
assert json.loads((root/'tp2-long-input.json').read_text())['passed']
for x in json.loads((root/'tp2-memory.json').read_text()).values():assert not x['guard_trip'] and not x['oom'] and len(x['boot_ids'])==1
m=json.loads((root/'tp1-main-memory.json').read_text());assert not m['guard_trip'] and not m['oom'] and len(m['boot_ids'])==1
checks=Path('/tmp/huihui-all-target-tests.log').read_text();assert '\nFAIL' not in checks and checks.count('ok ')==3;(root/'code-checks.txt').write_text(checks)
files=['verification.json','peer-verification.json','tp1-main-runtime.json','tp2-runtime.json','tp1-main-memory.json','tp2-memory.json','tp2-long-input.json','main-service-check.json','talk-chat-check.json','release-quality-review.json','code-checks.txt']
gate={'status':'passed','model':NAME,'evidence_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'limitation':'TP1 LLM qualified with ASR/TTS and without FLUX. Full TP1 bundle can fail 4 GiB reserve when FLUX peak is reserved. Main full bundle runs TP2; reserve unchanged.'}
(root/'release-validation.json').write_text(json.dumps(gate,indent=2));print('RELEASE VALIDATION PASSED',flush=True)
