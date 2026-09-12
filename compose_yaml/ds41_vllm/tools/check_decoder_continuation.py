"""Verify baseline/selected-cache interoperability, fallback and DSpark TG."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,ast,hashlib,json,random,subprocess,time,typing,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'results/decoder-rows-continuation.json'
ssh=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reuse-prefix-results',type=Path)
args=parser.parse_args()
results=([x for x in json.loads(args.reuse_prefix_results.read_text()) if x['kind']=='prefix'] if args.reuse_prefix_results else [])
if args.reuse_prefix_results:assert len(results)==3

def control(enabled):
    epoch=time.time_ns()
    c={'epoch':epoch,'graphs':True,'expert_io':'batch_overlap','kernel_tokens':2048,
       'shared_buffers':True,'final_decoder_rows':enabled,'validate_final_decoder':False}
    data=json.dumps(c).encode();path=root/'graph-control.json'
    subprocess.run(ssh+[f"cat > '{path}.next' && mv '{path}.next' '{path}'"],input=data,check=True)
    tmp=path.with_suffix('.json.next');tmp.write_bytes(data);tmp.replace(path)
    return epoch

def call(messages,salt,**extras):
    req={'model':'deepseek-v4.1-flash','messages':messages,'temperature':0,
         'chat_template_kwargs':{'thinking':False},'max_completion_tokens':180,'seed':42,
         'cache_salt':salt}|extras
    start=time.monotonic()
    r=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=600) as response:reply=json.load(response)
    return {'reply':reply,'seconds':time.monotonic()-start}

base='\n'.join(f'Record {i}: the sample color is blue and the value is {i%23}.' for i in range(210))
requests=[base+'\nReply with exactly cobalt731.',base+'\nAdditional context: '+('An extra sample. '*60)+'\nReply with exactly amber962.',base.replace('Record 170:','Modified entry 170:')+'\nReply with exactly violet543.']
# Cross-mode reuse both ways, including a junction inside the original prefix.
salt='decoder-continuation-'+str(time.time_ns())
for index,enabled in enumerate(() if args.reuse_prefix_results else (False,True,False)):
    control(enabled)
    row=call([{'role':'user','content':requests[index]}],salt,max_completion_tokens=16)
    row.update(kind='prefix',enabled=enabled,index=index)
    choice=row['reply']['choices'][0]
    assert choice['finish_reason']=='stop' and choice['message']['content'].strip()==('cobalt731','amber962','violet543')[index],row
    if index:assert row['reply']['usage']['prompt_tokens_details']['cached_tokens']>1000,row
    results.append(row);out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('PREFIX_PASS '+str(index),flush=True)

prompts=[
'Write a Python function implementing binary search on a sorted list of integers. Include type hints and a concise docstring. Return only code.',
'Explain the difference between concurrency and parallelism in three concise sentences.']
for pi,prompt in enumerate(prompts):
    reference=None
    for enabled in (False,True):
        control(enabled)
        row=call([{'role':'user','content':base+'\nIgnore the sample records for this task. '+prompt}],f'decoder-tg-{pi}-{enabled}-{time.time_ns()}')
        choice=row['reply']['choices'][0];text=choice['message']['content']
        assert choice['finish_reason']=='stop',row
        if reference is None:reference=text
        # This pinned probabilistic DSpark runtime also varies across repeated
        # full-path requests with the same seed. Preserve literal equality as
        # evidence, but require behavior instead of pretending it is deterministic.
        row['matches_reference_text'] = text == reference
        if pi == 0:
            code=text.strip().removeprefix('```python').removeprefix('```').removesuffix('```').strip()
            tree=ast.parse(code)
            functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='binary_search']
            assert len(functions)==1
            body=ast.Module(body=functions,type_ignores=[])
            assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) for n in ast.walk(body))
            ns={'__builtins__':{'len':len,'range':range,'int':int,'list':list,'min':min,'max':max},'List':typing.List,'Optional':typing.Optional}
            exec(compile(body,'<generated binary search>','exec'),ns)
            fn=ns['binary_search'];rng=random.Random(72)
            checks=0
            for size in (0,1,2,7,31,128):
                arr=sorted(rng.randrange(-30,31) for _ in range(size))
                for target in range(-33,34):
                    value=fn(arr.copy(),target)
                    if target in arr:assert isinstance(value,int) and 0<=value<len(arr) and arr[value]==target
                    else:assert value in (None,-1)
                    checks+=1
            row['binary_search_checks']=checks
        else:
            assert len(text.split())>20 and 'concurr' in text.lower() and 'parallel' in text.lower()

        row.update(kind='decode',prompt_id=pi,enabled=enabled)
        results.append(row);out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print('DECODE_PASS '+str(pi)+' '+str(enabled),flush=True)

# Prompt logprobs consume every hidden row: selection must fall back entirely.
reference=None
for enabled in (False,True):
    epoch=control(enabled)
    row=call([{'role':'user','content':'A short verification passage with enough words to exceed sixteen input tokens. Reply with exactly OK.'}],f'decoder-logprobs-{epoch}',prompt_logprobs=2,max_completion_tokens=8)
    assert row['reply']['choices'][0]['message']['content'].strip()=='OK',row
    log=subprocess.check_output(['docker','logs','--since','10m','ds41-stream-0'],stderr=subprocess.STDOUT).decode().split('EXPERT_CACHE_RESET '+str(epoch),1)[1]
    assert 'FINAL_DECODER_ROWS {' not in log,log[-300:]
    # Retain actual logprobs when this API build exposes them.
    row.update(kind='prompt_logprobs_fallback',enabled=enabled)
    results.append(row);out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('PROMPT_LOGPROBS_FALLBACK_PASS '+str(enabled),flush=True)
print('DECODER_CONTINUATION_PASS',flush=True)
