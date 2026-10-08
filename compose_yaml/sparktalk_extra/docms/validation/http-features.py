"""Measure the isolated or deployed document API with real generated files."""
import argparse,base64,json,time,urllib.request,subprocess,zipfile,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18696');p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
inputs={}
for fmt in ['docx','hwp','hwpx','pdf']:
 inputs[fmt]={'format':fmt,'title':'문서 기능 실측','page':{'header':'SparkTalk 문서 검증','page_numbers':True},'sections':[{'blocks':[{'type':'heading','text':'생성 결과'},{'type':'paragraph','text':[{'text':'강조 본문','style':{'bold':True,'color':'224488'}}]},{'type':'table','rows':[['번호','내용']]+[[str(i),'검증 행 '+str(i)] for i in range(1,21)]},{'type':'paragraph','text':'FINAL_DOC_CHECK'}]}]}
inputs['pptx']={'format':'pptx','title':'발표자료 실측','slides':[{'title':'편집 가능한 표','blocks':[{'type':'table','rows':[['번호','내용']]+[[str(i),'검증 행 '+str(i)] for i in range(1,51)]},{'type':'paragraph','text':'FINAL_DOC_CHECK'}]}]}
inputs['xlsx']={'format':'xlsx','title':'스프레드시트 실측','sheets':[{'name':'실측','columns':[{'title':'항목'},{'title':'금액','format':'number'},{'title':'누적','format':'number'}],'rows':[[str(i) if i<100 else 'FINAL_DOC_CHECK',i,{'formula':f'SUM(B2:B{i+1})'}] for i in range(1,101)],'cells':[{'cell':'B2','style':{'background':'E7EEF5','bold':True}}],'charts':[{'kind':'line','title':'금액 추이','cell':'E2','categories':'A2:A101','series':[{'name':'금액','values':'B2:B101'}]}]}]}
(a.out/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
measurements=[]
for repeat in range(a.repeats):
 for fmt,data in inputs.items():
  started=time.monotonic();request=urllib.request.Request(a.url+'/v1/documents',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'});response=urllib.request.urlopen(request,timeout=100);result=json.load(response);elapsed=time.monotonic()-started
  assert not result.get('warning'),result.get('warning');assert len(result['files'])==(1 if fmt=='pdf' else 2)
  sizes={};folder=a.out/fmt;folder.mkdir(exist_ok=True)
  for f in result['files']:
   content=base64.b64decode(f['data']);sizes[f['name']]=len(content);target=folder/f['name'];target.write_bytes(content)
   if f['name'].endswith(('.docx','.pptx','.xlsx','.hwpx')):
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
  text=subprocess.check_output(['pdftotext',str(folder/'document.pdf'),'-'],text=True)
  assert 'FINAL_DOC_CHECK' in text,(fmt,text[-300:])
  measurements.append({'format':fmt,'repeat':repeat,'seconds':elapsed,'sizes':sizes,'passed':True});print(fmt,repeat,round(elapsed,3),sizes,flush=True)
summary={fmt:{'median_seconds':statistics.median(r['seconds'] for r in measurements if r['format']==fmt),'runs':a.repeats} for fmt in inputs}
(a.out/'http-results.json').write_text(json.dumps({'endpoint':a.url,'rows':measurements,'summary':summary},indent=2)+'\n')
print(summary)
