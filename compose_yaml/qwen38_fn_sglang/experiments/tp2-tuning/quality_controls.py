import json,base64,zlib,struct
from pathlib import Path
from measure import chat
out=[]
tool={'type':'function','function':{'name':'get_weather','description':'Get weather for a location. Keep the city name exactly as requested.','parameters':{'type':'object','properties':{'location':{'type':'string'}},'required':['location'],'additionalProperties':False}}}
r=chat('서울 날씨를 확인하기 위해 get_weather 도구를 호출해 주세요. location은 정확히 서울로 지정하세요.',tools=[tool],max_tokens=128)
name='';args=''
for c in r['calls']:
 f=c.get('function',{});name+=f.get('name') or '';args+=f.get('arguments') or ''
try:r['passed']=name=='get_weather' and json.loads(args)=={'location':'서울'}
except Exception:r['passed']=False
r['name']='tool_call';out.append(r)
def chunk(kind,payload):return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*64)*64))+chunk(b'IEND',b'')
r=chat([{'type':'text','text':'이 이미지의 색상을 한국어로 한 단어로 답하세요.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}],max_tokens=32)
r['passed']=any(s in r['text'].lower() for s in ['빨간','빨강','붉','red']);r['name']='image';out.append(r)
(Path(__file__).parent/'results/quality-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print([(r['name'],r['passed']) for r in out],flush=True)
