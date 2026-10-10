"""Task-level checks beyond numeric tensor roundtrips; no real tools execute."""
import json
import sys
import trial


def run(label):
    results=[]
    tasks=[
        ('math_sub_mul','(83-19)×7의 값을 숫자만 답해라.','448'),
        ('math_div_add','144÷12+17의 값을 숫자만 답해라.','29'),
        ('power','2의 10제곱을 숫자만 답해라.','1024'),
        ('sort','[5,2,9,2]를 오름차순으로 정렬하고 JSON 배열만 출력해라.',[2,2,5,9]),
        ('filter','[3,8,11,14,20]에서 짝수만 골라 JSON 배열로 답해라.',[8,14,20]),
        ('extract','김민수는 오전 9시에 GPU API 회의에 참석한다. name, hour(정수), topic 키의 JSON만 출력해라.',{'name':'김민수','hour':9,'topic':'GPU API'}),
        ('missing','자료: 프로젝트명은 펭귄이고 담당자는 영희다. 자료에 없는 비밀번호를 물으면 정확히 없음이라고 답해라. 비밀번호는?','없음'),
        ('negation','중국어 지시 不要删除文件 表示하지 말아야 할 행동은 파일 삭제다. 이 지시를 따라 파일을 삭제해야 하는지 {"delete":true 또는 false} JSON으로만 답해라.',{'delete':False}),
        ('japanese_fields','音声速度を1.25にして、解像度は864×480のまま維持してください。rate,width,heightキーだけのJSONを返してください。',{'rate':1.25,'width':864,'height':480}),
        ('units_json','영상 864×480, 24 FPS를 width,height,fps 정수 키의 JSON으로만 답해라.',{'width':864,'height':480,'fps':24}),
        ('acronyms','GPU, API, SQL, FPS 네 약어를 순서 그대로 JSON 문자열 배열로만 출력해라.',['GPU','API','SQL','FPS']),
        ('logical_order','A는 B보다 크고 B는 C보다 크다. 가장 작은 항목을 한 글자로만 답해라.','C'),
    ]
    for name,prompt,expected in tasks:
        r=trial.stream(dict(messages=[{'role':'user','content':prompt}],max_tokens=128))
        r['name']=name
        try:r['pass']=(json.loads(r['text'])==expected) if not isinstance(expected,str) else r['text'].strip()==expected
        except Exception:r['pass']=False
        if name=='extract':
            try:
                value=json.loads(r['text'])
                r['pass']=set(value)=={'name','hour','topic'} and value['name']=='김민수' and value['hour']==9 and value['topic'] in ('GPU API','GPU API 회의')
                r['rubric_note']='Both GPU API and GPU API 회의 describe the supplied meeting topic.'
            except Exception:r['pass']=False
        results.append(r)
        (trial.OUT/label/'quality.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    tool={'type':'function','function':{'name':'move_room','description':'Move the current conversation into an existing folder.',
           'parameters':{'type':'object','properties':{'folder':{'type':'string'},'confirmed':{'type':'boolean'}},'required':['folder','confirmed']}}}
    tools=[{'type':'function','function':{'name':f'archive_lookup_{i}','description':f'Search archived document collection {i}.',
           'parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}} for i in range(20)] + [tool]
    r=trial.stream(dict(messages=[{'role':'user','content':'현재 대화방을 뉴스 폴더로 옮겨라. 내가 요청한 이동이므로 confirmed는 true다.'}],max_tokens=256,tools=tools))
    r['name']='tool_selection_21'
    try:r['pass']=len(r['calls'])==1 and r['calls'][0]['name']=='move_room' and json.loads(r['calls'][0]['arguments'])=={'folder':'뉴스','confirmed':True}
    except Exception:r['pass']=False
    results.append(r)
    if r['pass']:
        messages=[{'role':'user','content':'현재 대화방을 뉴스 폴더로 옮겨라. 내가 요청한 이동이므로 confirmed는 true다.'},
                  {'role':'assistant','content':None,'tool_calls':[{'id':'audit_call_0','type':'function','function':r['calls'][0]}]},
                  {'role':'tool','tool_call_id':'audit_call_0','content':'{"success":true,"folder":"뉴스"}'},
                  {'role':'user','content':'이동 결과를 {"success":true,"folder":"뉴스"} 형태의 JSON으로만 답해라.'}]
        r=trial.stream(dict(messages=messages,max_tokens=128,tools=tools));r['name']='tool_result_followup'
        try:r['pass']=not r['calls'] and json.loads(r['text'])=={'success':True,'folder':'뉴스'}
        except Exception:r['pass']=False
        results.append(r)
    (trial.OUT/label/'quality.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print('EXTENDED_QUALITY',label,sum(x['pass'] for x in results),len(results),flush=True)
    thinking=[]
    for name,prompt,with_tool in [
        ('thinking_math','17×23을 계산하고 최종 답변에는 숫자만 써라.',False),
        ('thinking_tool','현재 대화방을 뉴스 폴더로 옮겨라. confirmed는 true로 해라.',True),
    ]:
        body=dict(model=trial.MODEL,messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=768,
                  chat_template_kwargs={'enable_thinking':True})
        if with_tool:body['tools']=[tool]
        r=trial.request('/v1/chat/completions',body)
        message=r['choices'][0]['message'];calls=message.get('tool_calls') or []
        if with_tool:
            passed=len(calls)==1 and calls[0]['function']['name']=='move_room' and json.loads(calls[0]['function']['arguments'])=={'folder':'뉴스','confirmed':True}
        else:passed=(message.get('content') or '').strip()=='391' and not calls
        thinking.append(dict(name=name,pass_=passed,response=r))
        (trial.OUT/label/'thinking-quality.json').write_text(json.dumps(thinking,ensure_ascii=False,indent=2))
    print('THINKING_QUALITY',label,sum(x['pass_'] for x in thinking),len(thinking),flush=True)
    import subprocess
    probe=subprocess.run(['docker','exec',trial.NAME,'python3','/experiment/loader_timing.py'],text=True,capture_output=True,check=True)
    (trial.OUT/label/'loader-copy-ab.log').write_text(probe.stdout+'\n'+probe.stderr)
    import resident_probe
    resident_probe.run(label)


if __name__=='__main__':run(sys.argv[1])
