#!/usr/bin/env python3
"""Identical deterministic smoke cases plus streamed latency on both checkpoints."""
import argparse,base64,json,struct,time,urllib.request,zlib
from pathlib import Path

def image():
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 w=h=224;raw=b''.join(b'\0'+bytes([255,0,0])*w for _ in range(h))
 png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
 return 'data:image/png;base64,'+base64.b64encode(png).decode()

def request(base,payload):
 start=time.monotonic();first=None;text='';reasoning='';usage=None;tools=[];finish=None
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   if line.strip()==b'data: [DONE]':break
   data=json.loads(line[6:]);usage=data.get('usage') or usage
   for choice in data.get('choices',[]):
    finish=choice.get('finish_reason') or finish
    d=choice.get('delta',{});c=d.get('content') or '';r=d.get('reasoning_content') or '';tc=d.get('tool_calls') or []
    if first is None and (c or r or tc):first=time.monotonic()
    text+=c;reasoning+=r;tools.extend(tc)
 elapsed=time.monotonic()-start;ttft=None if first is None else first-start
 return {'text':text,'reasoning':reasoning,'tool_calls':tools,'finish_reason':finish,'usage':usage,'seconds':elapsed,'ttft_s':ttft,'stream_tg_estimate':None if not usage or first is None or elapsed<=ttft else max(0,usage['completion_tokens']-1)/(elapsed-ttft)}

def cases():
 yield 'warmup','Reply with READY.',None
 yield 'arithmetic','Compute 137 * 29. Reply only with the integer.', '3973'
 yield 'json','Return only JSON, without markdown: an object with keys name and count; name must be "테스트", count must be the integer 7.',None
 yield 'sorting','Sort these numbers ascending. Reply only with a JSON array: 9, -3, 2, 2, 0.',None
 yield 'korean','다음 문장을 존댓말로 자연스럽게 고쳐라. 설명 없이 문장만 써라: 내일 자료 보내줄게.',None
 yield 'translation','Translate into Korean, preserving the negation: "The server is not down; only the worker is restarting."',None
 yield 'coding','Write only a Python function named unique_ordered(items) that removes duplicates preserving order. Items are hashable. No markdown.',None
 yield 'fiction','Write a short fictional dialogue in Korean between a detective and a thief caught stealing a painting. Keep it under 120 words.',None
 yield 'tool','Use the provided tool to check the weather in Seoul.',None
 yield 'vision',[{'type':'text','text':'Name the dominant color of this image in one English word.'},{'type':'image_url','image_url':{'url':image()}}], 'red'
 filler='This record contains routine inventory entries and no access key.\n'*500
 yield 'long_needle',filler[:len(filler)//2]+'\nThe retrieval key is SPARK-7429-MINT.\n'+filler[len(filler)//2:]+'\nReturn only the retrieval key stated in these records.', 'SPARK-7429-MINT'
 for i in range(3):yield 'speed_'+str(i),'Explain how a hash table works in clear English. Include collision handling and resizing. Write at least 300 words. Trial '+str(i),None

def main(base,out,model='huihui-validation'):
 results=[]
 for name,content,expect in cases():
  payload={'model':model,'messages':[{'role':'user','content':content}],'temperature':0,'seed':42,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
  if name=='tool':payload.update(tools=[{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}],tool_choice='auto')
  try:
   row=request(base,payload);row.update(name=name)
   if expect:row['passed']=row['text'].strip().lower().rstrip('.')==expect.lower()
   if name=='json':row['passed']=json.loads(row['text'])=={'name':'테스트','count':7}
   if name=='sorting':row['passed']=json.loads(row['text'])==[-3,0,2,2,9]
  except Exception as e:row={'name':name,'error':str(e),'passed':False}
  results.append(row);out.write_text(json.dumps({'base':base,'results':results},ensure_ascii=False,indent=2));print(name,json.dumps(row,ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',default='http://192.168.100.60:30123');p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='huihui-validation');a=p.parse_args();main(a.base,a.output,a.model)
