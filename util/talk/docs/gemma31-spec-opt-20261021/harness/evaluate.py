"""Evaluate generated code inside a disposable container with no network/GPU/mounts."""
import argparse,json,pathlib,re,subprocess
p=argparse.ArgumentParser();p.add_argument('dirs',nargs='+');p.add_argument('--image',default='sparktalk-qwen38fn_exl3:1.5.4-managed1');p.add_argument('--out',required=True);a=p.parse_args()
tests={
'code_unique':'''assert stable_unique([3,1,3,2,1]) == [3,1,2]
assert stable_unique([])==[]
assert stable_unique([0,False,1,True])==[0,1]''',
'code_chunks':'''assert chunks(list(range(7)),3)==[[0,1,2],[3,4,5],[6]]
assert chunks([],2)==[]
for n in [0,-1]:
 try: chunks([1],n)
 except ValueError: pass
 else: raise AssertionError('no ValueError')''',
'code_merge':'''x=[[5,7],[1,3],[3,5],[10,11]];old=[i[:] for i in x]
assert merge_intervals(x)==[[1,7],[10,11]]
assert x==old
assert merge_intervals([])==[]
assert merge_intervals([[-5,-3],[-4,0],[2,2]])==[[-5,0],[2,2]]''',
'code_topo':'''edges=[(0,2),(1,2),(2,3)];order=topological_sort(5,edges)
assert sorted(order)==list(range(5))
pos={v:i for i,v in enumerate(order)}
assert all(pos[u]<pos[v] for u,v in edges)
assert topological_sort(0,[])==[]
try:topological_sort(2,[(0,1),(1,0)])
except ValueError:pass
else:raise AssertionError('cycle accepted')''',
'code_lru':'''from collections import OrderedDict
import random
r=random.Random(512)
for cap in [0,1,3]:
 c=LRUCache(cap);ref=OrderedDict()
 for _ in range(150):
  k=r.randrange(6)
  if r.random()<0.6:
   v=r.randrange(50);c.put(k,v)
   if k in ref:del ref[k]
   if cap:
    ref[k]=v
    if len(ref)>cap:ref.popitem(last=False)
  else:
   want=ref.pop(k,-1)
   if want!=-1:ref[k]=want
   assert c.get(k)==want'''}
def clean(text):return re.sub(r'<[^>]+>','',text).strip()
def code_of(text):
 m=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
 return m.group(1) if m else clean(text)
report={}
for dirname in a.dirs:
 d=pathlib.Path(dirname);rs=json.loads((d/'results.json').read_text());rows=[]
 for x in rs:
  name=x['name'];text=x.get('text','');s=clean(text);passed=None;detail=None
  if name=='arithmetic':passed=s=='449'
  elif name=='json':
   try:
    obj=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',s));passed=obj=={'name':'펭귄','count':3,'ok':True} and type(obj['count']) is int and type(obj['ok']) is bool
   except:passed=False
  elif name=='ko_instruction':passed=bool(re.search(r'^해달은',s)) and bool(re.search(r'(?:\n|[.!?]\s*)펭귄은',s))
  elif name=='japanese':passed=all(k in s for k in ['ペンギン','雪','走'])
  elif name=='reasoning':passed=bool(re.search(r'3\s*/\s*10|\\frac\s*\{3\}\s*\{10\}|0\.3\b|30\s*%',s))
  elif name=='tool_title':passed='<|tool_call>' in text and 'set_session_title' in text and '261010 주요뉴스' in text
  elif name in ['recall_1m_five_depths','recall_256k_five_depths']:
   try:
    obj=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',s));expected=json.loads((d.parent/'probe.json').read_text())['locations'];passed=all(obj.get(k)==v['expected'] for k,v in expected.items());detail={k:obj.get(k)==v['expected'] for k,v in expected.items()}
   except Exception as e:passed=False;detail=str(e)
  elif name.startswith('recall'):passed='청록펭귄' in s
  elif name.startswith('long_ko_'):passed='청록펭귄' in s and x['replacement_characters']==0
  elif name in tests:
   code=code_of(text)+'\n'+tests[name]
   try:
    r=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only','--memory','1g','--memory-swap','1g','--cpus','2','--pids-limit','64','--tmpfs','/tmp:rw,size=64m','--entrypoint','python3',a.image,'-I','-'],input=code,text=True,capture_output=True,timeout=20)
    passed=r.returncode==0;detail=r.stderr[-500:] if not passed else None
   except subprocess.TimeoutExpired:passed=False;detail='timeout'
  if passed is not None:rows.append(dict(name=name,passed=passed,detail=detail))
 report[d.name]=dict(passed=sum(x['passed'] for x in rows),total=len(rows),checks=rows)
pathlib.Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
