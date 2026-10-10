"""Matched coding tasks, executed in a networkless, read-only CPU container."""
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'velogb10-yarn'))
from compare import stream

TASKS = {
    'lru': '''Implement Python class LRUCache(capacity: int) with get(key, default=None), put(key,value), delete(key)->bool, and __len__. Generic hashable keys and any values, including None. Capacity must be positive, else ValueError. get and overwriting put promote the entry to most recently used; evict the least recent only when needed. All operations should be O(1). Return a complete module, only Python code.''',
    'jsonl': '''Implement parse_jsonl(text: str)->list[dict] in Python. Ignore empty or whitespace-only lines. Accept a Unicode BOM only at the start of the entire input. Every nonempty line must be a JSON object. On invalid JSON or a non-object, raise ValueError containing the original 1-based line number (including blank lines). Preserve Unicode and JSON values. Return a complete module, only Python code.''',
    'intervals': '''Implement merge_intervals(intervals: list[tuple[int,int]])->list[tuple[int,int]] in Python. Closed intervals, merge overlaps including equal endpoints, but do not merge merely consecutive integer endpoints. Sort results by start. Do not modify the input. Empty input returns []. Any reversed interval raises ValueError. Preserve negative endpoints. Return a complete module, only Python code.''',
    'sqlite': '''Implement search_users(conn, prefix: str, limit: int)->list[tuple[int,str]] using Python sqlite3. Existing table users(id INTEGER PRIMARY KEY, name TEXT). Return id,name for names starting with the literal prefix (percent and underscore in the prefix are literal characters, backslash too). Use SQL parameters, never interpolate user input into SQL. ORDER BY name then id. Limit must be an integer from 1 through 1000, otherwise ValueError. Empty prefix matches all non-NULL names. Do not close or commit the supplied connection. Return a complete module, only Python code.''',
    'toposort': '''Implement topo_sort(nodes: list[str], edges: list[tuple[str,str]])->list[str] in Python. Each edge (a,b) means a comes before b. Repeated edges are ignored. At each step choose the lexicographically smallest currently available node. Raise ValueError on a cycle, duplicate nodes, or any unknown endpoint. Empty nodes with no edges returns []. Do not modify inputs. Return a complete module, only Python code.''',
}

CHECK = r'''
import importlib.util, json, sys, sqlite3
spec=importlib.util.spec_from_file_location('candidate',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
name=sys.argv[2];n=0
def eq(a,b):
 global n
 assert a==b,(a,b);n+=1
def bad(fn):
 global n
 try:fn()
 except ValueError:n+=1;return
 raise AssertionError('ValueError expected')
if name=='lru':
 bad(lambda:m.LRUCache(0));bad(lambda:m.LRUCache(-1))
 c=m.LRUCache(2);eq(c.get('x'),None);eq(c.get('x',9),9);c.put('a',1);c.put('b',2);eq(len(c),2)
 eq(c.get('a'),1);c.put('c',3);eq(c.get('b'),None);c.put('a',4);c.put('d',5);eq(c.get('c'),None);eq(c.get('a'),4)
 eq(c.delete('a'),True);eq(c.delete('a'),False);eq(len(c),1);c.put('none',None);eq(c.get('none',7),None)
 eq(c.delete('none'),True);c.put(('key',2),8);eq(c.get(('key',2)),8)
if name=='jsonl':
 eq(m.parse_jsonl(''),[]);eq(m.parse_jsonl(' \n\t\n'),[])
 eq(m.parse_jsonl('\ufeff{"가":"펭귄"}\n\n {"x":null,"b":false}\n'),[{'가':'펭귄'},{'x':None,'b':False}])
 for text,line in [('{}\n\n[1]',3),('\n\ninvalid',3),('{}\n\ufeff{}',2),('null',1),('42',1),('"x"',1)]:
  try:m.parse_jsonl(text)
  except ValueError as e:assert str(line) in str(e);n+=1
  else:raise AssertionError((text,'ValueError expected'))
if name=='intervals':
 f=m.merge_intervals;eq(f([]),[]);eq(f([(1,3),(3,5)]),[(1,5)]);eq(f([(1,2),(3,4)]),[(1,2),(3,4)])
 x=[(7,9),(-3,-1),(1,4),(2,3),(4,8),(-3,-1)];old=x.copy();eq(f(x),[(-3,-1),(1,9)]);eq(x,old)
 eq(f([(0,0),(0,0)]),[(0,0)]);bad(lambda:f([(3,2)]));bad(lambda:f([(0,1),(9,8)]))
if name=='sqlite':
 c=sqlite3.connect(':memory:');c.execute('CREATE TABLE users(id INTEGER PRIMARY KEY,name TEXT)')
 rows=[(1,'a_b'),(2,'axb'),(3,'a%b'),(4,'a\\b'),(5,'alpha'),(6,'alpha'),(7,None),(8,"x' OR 1=1 --"),(9,'한글')]
 c.executemany('INSERT INTO users VALUES (?,?)',rows)
 eq(m.search_users(c,'a_',10),[(1,'a_b')]);eq(m.search_users(c,'a%',10),[(3,'a%b')]);eq(m.search_users(c,'a\\',10),[(4,'a\\b')])
 eq(m.search_users(c,'alpha',1),[(5,'alpha')]);eq(m.search_users(c,"x' OR 1=1 --",100),[(8,"x' OR 1=1 --")]);eq(m.search_users(c,'한',1),[(9,'한글')])
 eq(len(m.search_users(c,'',100)),8);eq(c.in_transaction,True)
 for lim in [0,-1,1001,1.5,'2']:bad(lambda lim=lim:m.search_users(c,'',lim))
if name=='toposort':
 f=m.topo_sort;eq(f([],[]),[]);eq(f(['z','a','c'],[]),['a','c','z'])
 nodes=['a','b','c','d'];edges=[('a','c'),('b','c'),('c','d'),('a','c')];old=(nodes.copy(),edges.copy())
 eq(f(nodes,edges),['a','b','c','d']);eq((nodes,edges),old)
 eq(f(['a','b','c'],[('c','a')]),['b','c','a'])
 bad(lambda:f(['a','b'],[('a','b'),('b','a')]));bad(lambda:f(['a'],[('a','a')]))
 bad(lambda:f(['a'],[('a','z')]));bad(lambda:f(['a','a'],[]))
print(json.dumps({'pass':True,'assertions':n}))
'''


def run(base, model, folder, image):
    folder.mkdir(exist_ok=True);folder.chmod(0o755)
    (folder/'check.py').write_text(CHECK)
    results=[]
    for name,prompt in TASKS.items():
        response=stream(base,model,[{'role':'user','content':prompt}],4096)
        text=response['text'].strip()
        match=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',text,re.S)
        code=match.group(1) if match else text
        path=folder/(name+'.py');path.write_text(code);path.chmod(0o644)
        cmd=['docker','run','--rm','--network','none','--read-only','--memory','512m','--memory-swap','512m',
             '--cpus','1','--pids-limit','32','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--user','65534:65534','--tmpfs','/tmp:rw,nosuid,nodev,size=32m','-v',str(folder)+':/work:ro',
             '--entrypoint','python3',image,'-I','/work/check.py','/work/'+name+'.py',name]
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            result=dict(name=name,pass_=p.returncode==0,stdout=p.stdout,stderr=p.stderr,response=response)
        except subprocess.TimeoutExpired:
            result=dict(name=name,pass_=False,error='Execution exceeded 30 seconds',response=response)
        results.append(result);(folder/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print('CODING',name,result['pass_'],flush=True)
    return dict(passed=sum(r['pass_'] for r in results),total=len(results))
