"""Capture real SparkTalk request construction using an isolated database copy.

The capture model returns markers and never forwards generation requests.
Raw fixtures are private and must be outside the repository.
"""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,hashlib,json,os,shutil,sqlite3,subprocess,threading,time,urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import yaml
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();os.umask(0o077)
root=Path(__file__).resolve().parents[1];dist=root.parents[1]/'util/talk/dist'
work=a.output.parent/'capture-instance';work.mkdir(parents=True,exist_ok=False)
cfg=yaml.safe_load((dist/'sparktalk.yaml').read_text())
assert cfg['runtime']['mode']=='external' and not cfg['runtime']['auto_start']
source=Path(cfg['server']['database']);source=source if source.is_absolute() else dist/source
with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src,sqlite3.connect(work/'capture.db') as dst:src.backup(dst)
cfg['server']['listen_addr']='127.0.0.1:18585';cfg['server']['database']=str(work/'capture.db')
cfg['model']['endpoint']='http://127.0.0.1:18586';cfg['model']['api_key']=''
cfg['runtime']['data_dir']=str(work/'runtime')
(work/'sparktalk.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
captures={};model=cfg['model']['default_model']
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  raw=json.dumps({'object':'list','data':[{'id':model,'object':'model','max_model_len':65536}]}).encode()
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  text=json.dumps(body.get('messages',[]),ensure_ascii=False)
  marker=next((m for m in ('cobalt731','marble482') if m in text),'OK')
  if marker!='OK':captures[marker]=body
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
  for event in ({'id':'capture','choices':[{'index':0,'delta':{'role':'assistant','content':marker},'finish_reason':None}]},{'id':'capture','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}):
   self.wfile.write(b'data: '+json.dumps(event).encode()+b'\n\n')
  self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
server=ThreadingHTTPServer(('127.0.0.1',18586),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
def api(path,body=None):
 req=urllib.request.Request('http://127.0.0.1:18585'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 return urllib.request.urlopen(req,timeout=90)
log=(work/'capture.log').open('wb');proc=subprocess.Popen([str(dist/'sparktalk-linux-arm64')],cwd=work,stdout=log,stderr=subprocess.STDOUT)
try:
 for _ in range(30):
  try:api('/api/config').close();break
  except OSError:time.sleep(.5)
 else:raise RuntimeError('Capture instance health timeout')
 doc=(root/'README.md').read_text()[:20000]
 cases=[('talk-tools-r5','cobalt731','성능 점검용 임시 대화다. 검증 표식 cobalt731 만 출력해.'),
        ('mixed-document-r5','marble482','다음 자료를 읽고 검증 표식 marble482 만 출력해.\n<document>\n'+doc+'\n</document>\n검증 표식만 출력해.')]
 fixtures=[]
 for name,marker,prompt in cases:
  with api('/api/sessions',{'title':'Isolated request capture'}) as f:session=json.load(f)
  with api('/api/chat',{'session_id':session['id'],'content':prompt,'tools_enabled':True}) as f:
   for line in f:
    if line.startswith(b'event: error'):raise RuntimeError('Capture chat failed')
  assert marker in captures
  request=captures[marker]
  assert len(request.get('tools',[]))>=10,'Real tool registry was not captured'
  fixtures.append({'name':name,'expected':marker,'request':request})
 a.output.write_text(json.dumps(fixtures,ensure_ascii=False,indent=2)+'\n');a.output.chmod(0o600)
 print(json.dumps([{'name':x['name'],'tools':len(x['request']['tools']),'sha256':hashlib.sha256(json.dumps(x['request'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()} for x in fixtures]),flush=True)
finally:
 proc.terminate()
 try:proc.wait(timeout=10)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 server.shutdown();log.close();shutil.rmtree(work)
