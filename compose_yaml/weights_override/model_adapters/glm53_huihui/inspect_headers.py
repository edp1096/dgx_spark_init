"""Inspect pinned GGUF headers using validated HTTP ranges, without weight downloads."""
import io,json,struct,time,urllib.request
from pathlib import Path
import threading
_URL_CACHE={}
_URL_LOCK=threading.Lock()
SOURCES={'original':('unsloth/GLM-5.3-Flash-GGUF','621d456e93e926e4b52f85cff5f634358c1828f9'),'huihui':('huihui-ai/GLM-5.3-Flash-abliterated-GGUF','50d39f500cbd1d0478295da89d417607521d378b')}
FORMATS={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
_SESSIONS=threading.local()
def get_range(repo,rev,file,offset,length):
    import requests
    if not hasattr(_SESSIONS,'session'):_SESSIONS.session=requests.Session()
    key=(repo,rev,file)
    for attempt in range(8):
        with _URL_LOCK:cached=_URL_CACHE.get(key)
        url=cached[0] if cached and time.monotonic()-cached[1]<300 else f'https://huggingface.co/{repo}/resolve/{rev}/{file}?download=true&t={time.time_ns()}'
        try:
            with _SESSIONS.session.get(url,headers={'Range':f'bytes={offset}-{offset+length-1}','Accept-Encoding':'identity'},stream=True,timeout=(20,20)) as r:
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-{offset+length-1}/'):raise ValueError('Range not honored')
                with _URL_LOCK:_URL_CACHE[key]=(r.url,time.monotonic())
                deadline=time.monotonic()+60;data=bytearray()
                for block in r.iter_content(256<<10):
                    if time.monotonic()>deadline:raise TimeoutError('Range transfer deadline')
                    data.extend(block)
                    if len(data)>length:raise ValueError('Oversized range')
                if len(data)!=length:raise EOFError('Short range')
                return bytes(data)
        except Exception:
            with _URL_LOCK:_URL_CACHE.pop(key,None)
            if attempt==7:raise
            time.sleep(min(20,2**attempt))

def parse(data):
    f=io.BytesIO(data)
    def number(fmt):
        n=struct.calcsize('<'+fmt); b=f.read(n)
        if len(b)!=n:raise EOFError('Incomplete header')
        return struct.unpack('<'+fmt,b)[0]
    def string():
        n=number('Q');b=f.read(n)
        if len(b)!=n:raise EOFError('Incomplete string')
        return b.decode('utf8')
    def value(t):
        if t==8:return string()
        if t==9:
            st,n=number('I'),number('Q')
            return [value(st) for _ in range(n)]
        return number(FORMATS[t])
    if f.read(4)!=b'GGUF':raise ValueError('Not GGUF')
    version=number('I')
    if version!=3:raise ValueError('Unexpected GGUF version')
    nt,nm=number('Q'),number('Q');meta={}
    for _ in range(nm):
        key=string();meta[key]=value(number('I'))
    ts=[]
    for _ in range(nt):
        name=string();nd=number('I');shape=[number('Q') for _ in range(nd)]
        ts.append({'name':name,'shape':shape,'type':number('I'),'offset':number('Q')})
    align=meta.get('general.alignment',32);start=(f.tell()+align-1)//align*align
    for t in ts:t['offset']+=start
    return {'metadata':meta,'tensors':ts,'data_start':start}
def run(out):
    out.mkdir(parents=True,exist_ok=True)
    for side,(repo,rev) in SOURCES.items():
        info=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true',timeout=30))
        for file in info['siblings']:
            name=file['rfilename']
            if not name.startswith('UD-Q4_K_XL/') or not name.endswith('.gguf'):continue
            dest=out/(side+'-'+Path(name).name+'.json')
            if dest.exists():continue
            data=get_range(repo,rev,name,0,min(file['size'],16<<20));result=parse(data)
            result.update(repo=repo,revision=rev,file=name,size=file['size'])
            dest.write_text(json.dumps(result))
            print(side,Path(name).name,'tensors',len(result['tensors']),'data_start',result['data_start'],flush=True)
if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]))
