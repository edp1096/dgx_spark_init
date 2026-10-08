#!/usr/bin/env python3
"""Execute held-out functional checks for generated code in CPU-only containers."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import uuid

SHORTLIST_TESTS = r'''
import copy, math
f=ns['select_ids']
def eq(args, want):
 before=copy.deepcopy(args)
 got=f(*args)
 assert got==want,(got,want)
 assert args==before,'mutated input'
def raises(args):
 try:f(*args)
 except ValueError:return
 raise AssertionError('ValueError not raised')
checks=[
 ('sparse',1,lambda:eq(([4,40,400,4000],[400],{40:2},3),[4,40,400])),
 ('protected-invalid',1,lambda:raises(([1,2],[3],{},1))),
 ('size-small',1,lambda:raises(([1,2],[1,2],{},1))),
 ('size-large',1,lambda:raises(([1,2],[],{},3))),
 ('size-bool',1,lambda:raises(([1,2],[],{},True))),
 ('size-float',1,lambda:raises(([1,2],[],{},1.0))),
 ('ranking-ties',2,lambda:eq(([1,2,3,4,5],[5],{1:3,2:9,3:9,4:2},3),[2,3,5])),
 ('ignore-invalid',2,lambda:eq(([1,2,3,4,5,6],[],{999:99,1:float('nan'),2:float('inf'),3:True,4:-1,5:0,6:2},2),[1,6])),
 ('fraction-frequency',2,lambda:eq(([1,2,3],[],{2:.5,3:.25},2),[2,3])),
 ('non-real-frequency',2,lambda:eq(([1,2,3],[],{1:'9',2:None,3:1},1),[3])),
 ('empty',3,lambda:eq(([],[],{},0),[])),
 ('duplicates',3,lambda:eq(([1,1,2,3],[3,3],{},2),[1,3])),
 ('all-protected',3,lambda:eq(([1,2],[2,1],{1:99},2),[1,2])),
 ('fill',3,lambda:eq(([8,2,4,6],[],{8:1},3),[2,4,8])),
]
'''
TELEMETRY_TESTS = r'''
import copy
f=ns['summarize_samples']
def eq(rows, baseline, want):
 before=copy.deepcopy(rows)
 got=f(rows,baseline)
 assert got==want,(got,want)
 assert rows==before,'mutated input'
def raises(b):
 try:f([],b)
 except ValueError:return
 raise AssertionError('invalid baseline accepted')
def metric(n,minimum,median,extra,gap):
 return dict(count=n,min_available=minimum,median_available=median,max_extra_memory=extra,max_gap=gap)
checks=[
 ('duplicate-last-valid',1,lambda:eq([dict(phase='loading',monotonic=2,MemAvailable=8),dict(phase='loading',monotonic=2,MemAvailable=6),dict(phase='loading',monotonic=2,MemAvailable=-1)],10,{'loading':metric(1,6,6,4,0)})),
 ('filter',1,lambda:eq([{},dict(phase='other',monotonic=1,MemAvailable=2),dict(phase='loading',monotonic=True,MemAvailable=2),dict(phase='loading',monotonic=2,MemAvailable=True),dict(phase='loading',monotonic=1,MemAvailable=-2)],10,{})),
 ('baseline-negative',1,lambda:raises(-1)),
 ('baseline-bool',1,lambda:raises(True)),
 ('baseline-nan',1,lambda:raises(float('nan'))),
 ('baseline-inf',1,lambda:raises(float('inf'))),
 ('baseline-text',1,lambda:raises('10')),
 ('median-gap',2,lambda:eq([dict(phase='benchmark',monotonic=9,MemAvailable=6),dict(phase='benchmark',monotonic=1,MemAvailable=10),dict(phase='benchmark',monotonic=4,MemAvailable=4),dict(phase='benchmark',monotonic=5,MemAvailable=8)],12,{'benchmark':metric(4,4,7,8,4)})),
 ('clamp',2,lambda:eq([dict(phase='post_idle',monotonic=1,MemAvailable=15)],10,{'post_idle':metric(1,15,15,0,0)})),
 ('phase-isolation',2,lambda:eq([dict(phase='loading',monotonic=1,MemAvailable=3),dict(phase='benchmark',monotonic=1,MemAvailable=5)],10,{'loading':metric(1,3,3,7,0),'benchmark':metric(1,5,5,5,0)})),
 ('empty',3,lambda:eq([],0,{})),
 ('single',3,lambda:eq([dict(phase='loading',monotonic=-1,MemAvailable=0)],0,{'loading':metric(1,0,0,0,0)})),
 ('generator',3,lambda: (f(iter([dict(phase='loading',monotonic=1,MemAvailable=3)]),5)=={'loading':metric(1,3,3,2,0)}) or (_ for _ in ()).throw(AssertionError('generator'))),
]
'''

def answer_text(text):
    if '</think>' in text:
        text=text.split('</think>',1)[1]
    return text.replace('<|im_end|>','').replace('<|endoftext|>','').strip()

def extract(text):
    text=answer_text(text)
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    return '\n\n'.join(blocks) if blocks else text

def evaluate(code, kind):
    harness='import json\nns={}\nexec(compile('+repr(code)+',"candidate.py","exec"),ns)\n'
    harness+=SHORTLIST_TESTS if kind=='code_shortlist' else TELEMETRY_TESTS
    harness+='''
results=[]
for name,group,check in checks:
 try:check();results.append(dict(name=name,group=group,passed=True))
 except BaseException as exc:results.append(dict(name=name,group=group,passed=False,error=repr(exc)))
print('BROAD_CODE_RESULT '+json.dumps(results))
'''
    container='broad-code-check-'+uuid.uuid4().hex[:12]
    cmd=['docker','run','--rm','--name',container,'--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--pids-limit','64','--memory','256m',
         '--memory-swap','256m','--cpus','1','--tmpfs','/tmp:rw,noexec,nosuid,size=16m',
         '-i','python:3.12-slim','python','-I','-']
    try:
        p=subprocess.run(cmd,input=harness,text=True,capture_output=True,timeout=20)
        lines=[x for x in p.stdout.splitlines() if x.startswith('BROAD_CODE_RESULT ')]
        if p.returncode or not lines:
            return dict(passed=False,groups=[False]*3,error=(p.stderr+p.stdout)[-4000:],exit_code=p.returncode)
        tests=json.loads(lines[-1].split(' ',1)[1])
        groups=[all(t['passed'] for t in tests if t['group']==g) for g in [1,2,3]]
        return dict(passed=all(groups),groups=groups,tests=tests)
    except subprocess.TimeoutExpired:
        return dict(passed=False,groups=[False]*3,error='execution timeout')
    finally:
        subprocess.run(['docker','rm','-f',container],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=10)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--modes', nargs='+', default=['ko64k','ko128k'])
    args=ap.parse_args()
    rows=[]
    for mode in args.modes:
        data=json.loads((args.directory/f'{mode}-benchmark.json').read_text())
        for row in data['rows']:
            if row['id'] in ['code_shortlist','code_telemetry']:
                if row['thinking'] and '</think>' not in row['content']:
                    grade=dict(passed=False,groups=[False]*3,execution_attempted=False,
                               error='No final answer; reasoning text is not submitted code')
                else:
                    grade=evaluate(extract(row['content']),row['id'])
                    grade['execution_attempted']=True
                rows.append(dict(mode=mode,id=row['id'],repeat=row['repeat'],**grade))
                print(mode,row['id'],row['repeat'],grade['groups'],flush=True)
    (args.directory/'code-grades.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
