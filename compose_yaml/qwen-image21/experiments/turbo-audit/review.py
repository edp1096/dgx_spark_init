"""Review contact sheets preserve matched prompts/seeds, not pixel identity."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
profiles=['baseline','turbo-nvfp4','turbo-selective','turbo-bf16','turbo-fp8'];labels=['Current UC NVFP4 / 40 steps','Turbo NVFP4 / 8 steps','Turbo selective NVFP4 / 8 steps','Turbo BF16 DiT / 8 steps','Turbo FP8 DiT / 8 steps']
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for name in ['penguin','hands','counting','korean','english','alpha','counting-seed2','korean-seed2','edit','multiref','multiref-seed2','photo-edit']:
 available=[]
 if name=='photo-edit':available.append((r/'inputs/hands-reference.png','Original reference'))
 for p,label in zip(profiles,labels):
  candidates=[r/p/'output'/(name+'.png'),r/('editing-check-'+p)/'output'/(name+'.png'),r/('joint-'+p)/'output'/(name+'.png')]
  path=next((x for x in candidates if x.exists()),None)
  if path is not None:available.append((path,label))
 if len(available)<2:continue
 w=512;sheet=Image.new('RGB',(w*len(available),w+44),'white');draw=ImageDraw.Draw(sheet)
 for i,(path,label) in enumerate(available):
  image=Image.open(path).convert('RGBA');image.thumbnail((w,w));bg=Image.new('RGBA',image.size,'white');bg.alpha_composite(image);sheet.paste(bg.convert('RGB'),(i*w,44));draw.text((i*w+8,8),label,fill='black',font=font)
 (r/'review').mkdir(exist_ok=True);sheet.save(r/'review'/(name+'.jpg'),quality=94)
