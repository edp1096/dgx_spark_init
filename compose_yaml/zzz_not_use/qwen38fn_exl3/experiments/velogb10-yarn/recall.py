"""Probe the real tokenizer and a fresh long-context three-position retrieval request."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokens',type=int,default=270000)
    ap.add_argument('--port',type=int,default=19304);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default='velo-yarn-audit');ap.add_argument('--fixed-repetitions',type=int)
    a=ap.parse_args();base=f'http://127.0.0.1:{a.port}'
    def request(path,body):
        q=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with opener.open(q,timeout=120) as r:return json.loads(r.read())
    def make(n):
        filler='This entry records a routine maintenance check. All tests passed.\n'
        return ('자료에서 시작, 중간, 끝의 암호를 찾아 마지막 질문에 답하라.\n시작 암호: 청록펭귄.\n'
                +filler*n+'\n중간 암호: 은빛해달.\n'+filler*n
                +'\n끝 암호: 자주빛고래.\n질문: 시작, 중간, 끝 암호를 순서대로 모두 출력해라. 암호 세 개만 답해라.')
    n=a.fixed_repetitions or max(1,(a.tokens-150)//26)
    count=None
    for _ in range(5):
        prompt=make(n);body={'model':a.model,'messages':[{'role':'user','content':prompt}],
            'chat_template_kwargs':{'enable_thinking':False}}
        if a.fixed_repetitions:break
        count=request('/v1/tokenize',body)['count']
        if a.tokens-64<=count<=a.tokens:break
        n=max(1,n+(a.tokens-count)//26)
    body|={'max_tokens':96,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    pending={'target_tokens':a.tokens,'actual_tokenize_count':count,'input_bytes':len(prompt.encode()),'filler_repetitions_per_half':n,'started_wall':time.time()}
    a.out.with_suffix('.pending.json').write_text(json.dumps(pending,indent=2))
    print('START',count,flush=True)
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic();first=None;chunks=[];usage=None;finish=None;done=False
    with opener.open(req,timeout=7200) as res:
        for line in res:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':done=True;break
            event=json.loads(raw);usage=event.get('usage') or usage
            for c in event.get('choices',[]):
                v=c.get('delta',{}).get('content') or ''
                if v and first is None:first=time.monotonic()-start
                chunks.append(v);finish=c.get('finish_reason') or finish
    text=''.join(chunks)
    result=pending|{'text':text,'usage':usage,'finish':finish,'done':done,'seconds':time.monotonic()-start,'ttft':first,
        'pass':all(k in text for k in ['청록펭귄','은빛해달','자주빛고래'])}
    if first:result['prefill_tps']=(usage['prompt_tokens'] if usage else count)/first
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
