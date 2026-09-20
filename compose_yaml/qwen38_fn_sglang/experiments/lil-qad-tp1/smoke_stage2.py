"""Image, legacy-client and speculative serving probes; local endpoint only."""
import argparse,base64,json,pathlib,re,struct,subprocess,sys,time,urllib.request,zlib
HERE=pathlib.Path(__file__).resolve().parent

def image(left,right):
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    row=b'\x00'+bytes(left)*128+bytes(right)*128
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',256,128,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(row*128))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(png).decode()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
    subprocess.run([sys.executable,str(HERE/'smoke_test.py'),'--output',str(args.output)],check=True)
    results=json.loads(args.output.read_text())
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,body=None):
        req=urllib.request.Request('http://127.0.0.1:8016'+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=600) as response:return json.load(response)
    def run(name,messages,**kw):
        body=dict(model='qwen38-qad-sglang-trial',messages=messages,temperature=0,max_tokens=512,
                  stream=False,reasoning_effort='none');body.update(kw)
        start=time.monotonic();response=request('/v1/chat/completions',body)
        results.append(dict(name=name,seconds=time.monotonic()-start,response=response))
        args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2))
        choice=response['choices'][0]
        assert choice['finish_reason']!='length',(name,choice)
        return choice['message']
    msg=run('legacy_thinking_off',[{'role':'user','content':'7*8의 결과만 숫자로 답하세요.'}])
    assert (msg.get('content') or '').strip()=='56' and not msg.get('reasoning_content'),msg
    for name,left,right,expected in (
        ('vision_red_blue',(255,0,0),(0,0,255),('red','blue')),
        ('vision_yellow_green',(255,255,0),(0,128,0),('yellow','green'))):
        msg=run(name,[{'role':'user','content':[
            {'type':'image_url','image_url':{'url':image(left,right)}},
            {'type':'text','text':'Identify the left and right region colors. Reply with exactly two lowercase English color names, left first, separated by a comma.'}]}])
        text=(msg.get('content') or '').lower().strip().strip('.').replace(' ','')
        assert text==','.join(expected),(name,msg)
    tool=next(r for r in results if r['name']=='tool_call')['response']['choices'][0]['message']['tool_calls'][0]
    tool={k:tool[k] for k in ('id','type','function')}
    msg=run('tool_round_trip',[
        {'role':'user','content':'Use get_weather to find the Seoul temperature. Reply with only the Celsius number.'},
        {'role':'assistant','content':'','tool_calls':[tool]},
        {'role':'tool','tool_call_id':tool['id'],'content':'{"city":"Seoul","temperature_c":23}'}],
        tools=[{'type':'function','function':{'name':'get_weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}])
    assert (msg.get('content') or '').strip()=='23',msg
    msg=run('counting',[{'role':'user','content':'Write the integers from 1 through 40 inclusive, in ascending order, separated by commas. Nothing else.'}])
    assert re.findall(r'\d+',msg.get('content') or '')==[str(i) for i in range(1,41)],msg
    info=request('/server_info')
    states=info.get('internal_states',[])
    accepts=[s['avg_spec_accept_length'] for s in states if 'avg_spec_accept_length' in s]
    results.append(dict(name='speculative_metrics',accept_lengths=accepts))
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2))
    assert accepts and min(accepts)>1.0,('MTP has not demonstrated accepted proposals',accepts)
    print(f'Vision, legacy off, tool round trip, counting and MTP probes passed; acceptance lengths={accepts}',flush=True)
    # On the larger graph/production candidate, also exercise batching, SSE
    # and sparse attention beyond the short-context probe's capacity.
    if info.get('context_length',0)>=32768:
        subprocess.run([sys.executable,str(HERE/'smoke_stage3.py'),
                        '--output',str(args.output)],check=True)

if __name__=='__main__':main()
