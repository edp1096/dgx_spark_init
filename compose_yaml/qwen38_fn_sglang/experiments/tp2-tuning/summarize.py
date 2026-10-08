import argparse,json,re,statistics,ast
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);a=p.parse_args()
for path in a.files:
 d=json.loads(path.read_text());rows=d['rows'];groups={}
 for r in rows:
  if r['name'].startswith('needle_'):
   expected='KEY_'+r['name'].split('_')[1]+'_'+str(r['repeat'])
   r['passed']=r['text'].strip()==expected
  elif r['name']=='json':
   try:r['passed']=json.loads(r['text'])=={'answer':42,'items':[1,2,3]}
   except ValueError:r['passed']=False
  elif r['name']=='code':
   code=re.sub(r'^```(?:python)?\s*|\s*```$','',r['text'].strip())
   try:
    ast.parse(code);namespace={};exec(compile(code,'generated','exec'),namespace)
    fn=namespace['unique_stable'];r['passed']=all(fn(x)==y for x,y in [([],[]),([3,1,3,2],[3,1,2]),(['a','b','a'],['a','b'])])
   except Exception:r['passed']=False
  groups.setdefault(r['name'],[]).append(r)
 report={}
 for name,rs in groups.items():
  report[name]={'requests':len(rs),'tg_median':statistics.median([r['tg'] for r in rs if r['tg']]) if any(r['tg'] for r in rs) else None,'ttft_median':statistics.median([r['ttft'] for r in rs]),'hanzi_requests':sum(bool(r['hanzi']) for r in rs),'accept_length_mean':statistics.mean([r['accept_length'] for r in rs if r.get('accept_length')]) if any(r.get('accept_length') for r in rs) else None,'passed':[r.get('passed') for r in rs]}
 d['summary']=report;path.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(path.name,json.dumps(report,ensure_ascii=False))
