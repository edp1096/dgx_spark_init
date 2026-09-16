from pathlib import Path
import json,subprocess,xml.etree.ElementTree as E,zipfile,time,sys
root=Path(sys.argv[1] if len(sys.argv)>1 else '/out')
results=[]
for p in root.glob('*/document.*'):
 if p.suffix not in ('.hwp','.hwpx'):continue
 start=time.monotonic();text='';counts={}
 if p.suffix=='.hwp':
  x=subprocess.run(['/opt/verify/bin/hwp5proc','xml','--no-validate-wellformed',str(p)],capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode();xml=E.fromstring(x.stdout);text=''.join(xml.itertext());counts={k:sum(1 for e in xml.iter() if e.tag.rsplit('}',1)[-1]==k) for k in ['TableControl','FootNote','EndNote','ShapeComponent']};(p.parent/'independent.xml').write_bytes(x.stdout)
 else:
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   for n in z.namelist():
    if n.endswith(('.xml','.hpf')):
     root=E.fromstring(z.read(n))
     if n.startswith('Contents/section'):
      text+=''.join(root.itertext());
      for e in root.iter():
       k=e.tag.rsplit('}',1)[-1];counts[k]=counts.get(k,0)+1
 if 'rich-page-' in str(p):assert '페이지 나눔 뒤 본문' in text
 else:
  for expected in ['병합 헤더','각주 내용 확인','미주 내용 확인','글상자 검증']:assert expected in text,(str(p),expected)
 results.append({'file':str(p),'passed':True,'seconds':time.monotonic()-start,'text_length':len(text),'object_counts':counts});print(str(p),'PASS',flush=True)
(root/'hancom-independent.json').write_text(json.dumps(results,indent=2))
assert len(results)==4
