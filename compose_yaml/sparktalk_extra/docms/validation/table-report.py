"""Actual HTTP PPTX/PDF table widths and generated-structure report checks."""
import argparse,base64,json,time,urllib.request,zipfile,xml.etree.ElementTree as ET,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18696');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=[['구분','건립','역할','주요 건물','현재 상태']]+[[n,'건립 시기','궁궐 역할 설명','주요 건물 설명','현재 상태를 소개하는 비교용 설명입니다.'] for n in ['경복궁','창덕궁','창경궁','경희궁','덕수궁']]
cases={'one':{'rows':rows,'widths':[8,12,20,25,35],'style':{'size':14}},'split':{'rows':[['번호','설명']]+[[str(i),'자동 분할 검증'] for i in range(25)],'widths':[1,4]},'merged':{'rows':[[{'text':'병합된 제목','col_span':2}],['구분','설명'],['예시','병합 셀 너비 검증']],'widths':[1,4]}}
results={}
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for name,table in cases.items():
 folder=a.out/name;folder.mkdir(exist_ok=True)
 data={'format':'pptx','title':'표 배치 검증','filename':name,'slides':[{'title':'궁궐 비교' if name=='one' else '표 생성 검사','table':table}]}
 (folder/'input.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
 start=time.monotonic();req=urllib.request.Request(a.url+'/v1/documents',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=100) as response: result=json.load(response)
 elapsed=time.monotonic()-start
 assert not result.get('warning'),result
 for file in result.pop('files'):(folder/file['name']).write_bytes(base64.b64decode(file['data']))
 report=result['presentation'];assert report['visually_verified'] is False
 with zipfile.ZipFile(folder/'document.pptx') as z:
  slides=[f for f in z.namelist() if __import__('re').fullmatch(r'ppt/slides/slide\d+\.xml',f)]
  assert len(slides)==report['slide_count']
  for slide,rows_count in zip(report['tables'][0]['slide_numbers'],report['tables'][0]['rows_per_slide']):
   xml=ET.fromstring(z.read(f'ppt/slides/slide{slide}.xml'));tbl=xml.find('.//a:tbl',ns)
   assert len(tbl.findall('a:tr',ns))==rows_count
   widths=[int(c.attrib['w'])/914400 for c in tbl.findall('a:tblGrid/a:gridCol',ns)]
   assert widths==report['tables'][0]['column_widths_inches']
 assert report['tables'][0]['split']==(name=='split')
 if name!='split':assert report['slide_count']==1
 pdfinfo=subprocess.check_output(['pdfinfo',str(folder/'document.pdf')],text=True)
 pages=int(next(line.split(':')[1] for line in pdfinfo.splitlines() if line.startswith('Pages:')))
 assert pages==report['slide_count']
 result.update(seconds=elapsed,preview_page_count=pages);results[name]=result
 print(name,round(elapsed,3),report,flush=True)
(a.out/'http-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
