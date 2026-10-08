"""Verify actual uncached long-input completion, not just configured capacity."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'lil-qad-tp1'))
from probe_context import make_prompt

BASE='http://127.0.0.1:8017'
MODEL='qwen38-qad-vllm-trial'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(path,data=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(data).encode() if data is not None else None,
                               headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=7200) as r:
        body=r.read()
        return json.loads(body) if body else None

def tokens(repetitions,diverse):
    return request('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':make_prompt(repetitions,diverse)}],
                               'chat_template_kwargs':{'enable_thinking':False}})['tokens']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    results=[]
    def save():args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2))
    models=request('/v1/models');results.append({'name':'models','response':models});save()
    short=request('/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':'17 더하기 8의 값을 숫자만 답하세요.'}],
                                        'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}})
    results.append({'name':'short','response':short});save()
    assert short['choices'][0]['message']['content'].strip()=='25',short
    print('Short text passed',flush=True)
    for target,diverse in [(131072-128,True),(1048576-128,False)]:
        request('/reset_prefix_cache',{})
        repeats=max(1,(target-64)//24)
        ids=tokens(repeats,diverse)
        for _ in range(8):
            if len(ids)<=target and target-len(ids)<48:break
            per_repeat=(len(ids)-64)/repeats
            repeats=max(1,repeats+int((target-len(ids))/per_repeat)-int(len(ids)>target))
            ids=tokens(repeats,diverse)
        assert len(ids)<=target and target-len(ids)<48,(target,len(ids))
        result={'name':'retrieval','input_tokens':len(ids),'diverse':diverse,'status':'running'}
        results.append(result);save()
        print(f'Beginning actual prompt_tokens={len(ids)}',flush=True)
        start=time.monotonic();first=None;parts=[];events=[];done=False;usage=None;finish=None
        req=urllib.request.Request(BASE+'/v1/completions',data=json.dumps({'model':MODEL,'prompt':ids,
            'temperature':0,'max_tokens':64,'stream':True,'stream_options':{'include_usage':True}}).encode(),
            headers={'Content-Type':'application/json'})
        with OPENER.open(req,timeout=7200) as response:
            for line in response:
                if not line.startswith(b'data: '):continue
                raw=line[6:].strip()
                if raw==b'[DONE]':done=True;break
                event=json.loads(raw);events.append(event)
                if event.get('usage'):usage=event['usage']
                for choice in event.get('choices',[]):
                    text=choice.get('text','')
                    if text and first is None:first=time.monotonic()-start
                    parts.append(text)
                    if choice.get('finish_reason'):finish=choice['finish_reason']
        answer=''.join(parts).strip()
        result.update(status='completed',seconds=time.monotonic()-start,ttft_seconds=first,
                      answer=answer,usage=usage,finish_reason=finish,stream_done=done,
                      correct=answer.replace(' ','')=='MAPLE,COMET',events=events)
        save()
        print(f'Completed {len(ids)} tokens: {answer!r}, seconds={result["seconds"]:.2f}, correct={result["correct"]}',flush=True)
        assert done and result['correct'] and finish=='stop',result
        assert usage and usage['prompt_tokens']==len(ids),usage
        details=usage.get('prompt_tokens_details') or {}
        assert not details.get('cached_tokens',0),details

if __name__=='__main__':main()
