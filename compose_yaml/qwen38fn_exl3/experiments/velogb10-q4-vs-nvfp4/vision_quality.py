"""Identical image inputs for vision-tower regression checks."""
import base64
import io
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw,ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'velogb10-yarn'))
from compare import stream,color_image


def run(base,model,out):
    out.mkdir(exist_ok=True)
    image=Image.new('RGB',(1024,640),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',64)
    for y,text in [(60,'코드: AB-391'),(180,'해상도: 864×480'),(300,'속도: 24 FPS')]:draw.text((60,y),text,font=font,fill='black')
    data=io.BytesIO();image.save(data,format='PNG');(out/'ocr.png').write_bytes(data.getvalue())
    ocr='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
    photo=Path('/home/edp1096/.cache/model-download-jobs/velogb10-talk-512k-20261010/exllama/image.png').read_bytes()
    penguin='data:image/png;base64,'+base64.b64encode(photo).decode()
    rows=[]
    cases=[('quadrants',color_image(),'그림의 좌상단, 우상단, 좌하단, 우하단 색을 순서대로 영어 소문자 JSON 배열만 답해라. 색 이름은 red, green, blue, yellow 중에서 골라라.',lambda r:json.loads(r['text'])==['red','green','blue','yellow']),
           ('korean-ocr',ocr,'이미지의 텍스트를 읽고 코드, 해상도, FPS 값을 순서대로 답해라.',lambda r:all(s in r['text'] for s in ['AB-391','864','480','24'])),
           ('natural-image',penguin,'사진에 보이는 동물 종류를 영어 소문자 한 단어로만 답해라.',lambda r:r['text'].strip().strip('.').lower()=='penguin')]
    for name,url,prompt,check in cases:
        r=stream(base,model,[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':url}}]}],192)
        try:r['pass']=bool(check(r))
        except (ValueError,KeyError):r['pass']=False
        r['name']=name;rows.append(r);(out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        print('VISION',name,r['pass'],r['text'][:200],flush=True)
    return dict(passed=sum(r['pass'] for r in rows),total=len(rows))
