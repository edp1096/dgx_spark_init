"""HTTP checks using cell.width only, including explicit conflict rejection."""
import argparse,base64,json,time,urllib.request,urllib.error,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18696');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
widths=[7,11,19,23,40]
rows=[[{'text':name,'width':width} for name,width in zip(['구분','건립','역할','주요 건물','현재 상태'],widths)]]+[[name,'건립 시기','역할 설명','주요 건물 설명','현재 상태를 설명하는 넓은 열입니다.'] for name in ['경복궁','창덕궁','창경궁','경희궁','덕수궁']]
results={}
for kind in ['table','block','merged']:
 table={'rows':rows,'style':{'size':14}}
 if kind=='merged':table={'rows':[[{'text':'병합 제목','col_span':2,'width':10}],[{'text':'구분','width':3},'설명'],['예시','병합 셀 너비 확인']]}
 slide={'title':'셀 너비 검증',**({'blocks':[{'type':'table',**table}]} if kind=='block' else {'table':table})}
 data={'format':'pptx','title':'셀 너비 검증','slides':[slide]};folder=a.out/kind;folder.mkdir(exist_ok=True);(folder/'input.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
 start=time.monotonic();req=urllib.request.Request(a.url+'/v1/documents',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req) as response:result=json.load(response)
 assert not result.get('warning'),result
 for f in result.pop('files'):(folder/f['name']).write_bytes(base64.b64decode(f['data']))
 report=result['presentation'];expected=[3.6,8.4] if kind=='merged' else [.84,1.32,2.28,2.76,4.8]
 assert report['slide_count']==1 and report['tables'][0]['split'] is False
 actual=report['tables'][0]['column_widths_inches'];assert all(abs(x-y)<1e-6 for x,y in zip(actual,expected))
 with zipfile.ZipFile(folder/'document.pptx') as z:
  xml=ET.fromstring(z.read('ppt/slides/slide1.xml'));ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
  stored=[int(c.attrib['w'])/914400 for c in xml.findall('.//a:tblGrid/a:gridCol',ns)]
  assert stored==actual
 results[kind]={**result,'seconds':time.monotonic()-start};print(kind,actual,flush=True)
bad={'format':'pptx','title':'충돌','slides':[{'title':'충돌','table':{'rows':[[{'text':'가','width':2}],[{'text':'나','width':3}]]}}]}
try:urllib.request.urlopen(urllib.request.Request(a.url+'/v1/documents',data=json.dumps(bad).encode(),headers={'Content-Type':'application/json'}));raise AssertionError('Conflict accepted')
except urllib.error.HTTPError as e:
 body=json.load(e);assert e.code==422 and 'Conflicting cell.width' in body['error'];results['conflict']=body
(a.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
