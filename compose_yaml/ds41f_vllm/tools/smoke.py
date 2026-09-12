"""Serial streamed requests, with server token counts and observed latency."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',default='/tmp/ds41-smoke.json')
p.add_argument('--base',default='http://127.0.0.1:8010')
p.add_argument('--suite',choices=['short','long','varied','heldout'],default='short')
p.add_argument('--repeat',type=int,default=1)
args=p.parse_args()
for _ in range(360):
    try:
        urllib.request.urlopen(args.base+'/health',timeout=2)
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('API not ready')
results=[]
prompts = ['대한민국의 수도는 어디인가? 한 단어로 답해.',
           'Write a Python function add(a, b) that returns their sum. Output only code.',
           'Count from 1 to 10, separated by commas.'] if args.suite == 'short' else [
           'Write a Python function implementing binary search on a sorted list of integers. Include type hints and a concise docstring. Return only code.',
           'Explain why the sky appears blue in 80 to 100 words.']
if args.suite == 'varied':
    prompts=[
        'Write a Python function that groups a list of words by anagram. Include type hints. Return only code.',
        'Explain how a refrigerator moves heat in 80 to 100 words.',
        '한국어로 TCP와 UDP의 차이를 세 문장으로 설명해.',
        'Write a PostgreSQL query returning the top three products by total revenue in each category from sales(category, product, revenue). Return only SQL.']
if args.suite == 'heldout':
    prompts=[
        'Write a Python function that merges overlapping integer intervals. Include type hints. Return only code.',
        'Explain how a DNS lookup works in 80 to 100 words.',
        '한국어로 캐시와 데이터베이스의 차이를 두 문장으로 설명해.',
        'Write PostgreSQL SQL to sum orders(amount, created_at) by calendar month for the year 2025, ordered by month. Return only SQL.']
for prompt in [p for p in prompts for _ in range(args.repeat)]:
    data={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],
          'max_tokens':48 if args.suite == 'short' else 160,'temperature':0,'stream':True,'stream_options':{'include_usage':True},
          'chat_template_kwargs':{'thinking':False},'return_token_ids':True}
    req=urllib.request.Request(args.base+'/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic();first=None;usage=None;parts=[];first_chunk_tokens=None;content_events=0;finish_reason=None
    with urllib.request.urlopen(req,timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '): continue
            if line.strip()==b'data: [DONE]': break
            obj=json.loads(line[6:])
            if obj.get('usage'): usage=obj['usage']
            for choice in obj.get('choices',[]):
                if choice.get('finish_reason') is not None: finish_reason=choice['finish_reason']
                content=choice.get('delta',{}).get('content')
                if content:
                    content_events+=1
                    if first is None:
                        first=time.monotonic()
                        token_ids=choice.get("token_ids")
                        if token_ids is not None: first_chunk_tokens=len(token_ids)
                    parts.append(content)
    end=time.monotonic()
    item={'prompt':prompt,'text':''.join(parts),'usage':usage,'finish_reason':finish_reason,
          'requested_max_tokens':data['max_tokens'],'total_seconds':end-start,
          'ttft_seconds':first-start if first else None,
          'overall_tokens_per_second':usage['completion_tokens']/(end-start) if usage else None,
          'first_content_event_tokens':first_chunk_tokens,'content_events':content_events,
          'decode_tokens_per_second':(usage['completion_tokens']-first_chunk_tokens)/(end-first) if usage and first and end>first and content_events>=2 and first_chunk_tokens is not None else None}
    results.append(item)
    Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(json.dumps(item,ensure_ascii=False),flush=True)
