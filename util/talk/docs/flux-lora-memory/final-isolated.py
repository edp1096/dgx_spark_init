import measure as m
from PIL import Image
src=m.dataurl(Image.open(m.ROOT/'baseline-generate.png').convert('RGB'))
small=m.dataurl(Image.open(m.ROOT/'baseline-generate.png').convert('RGB').resize((768,768)))
jobs=[('outpaint','outpaint',dict(anypaint_image=small,outpaint_left=128,outpaint_right=128,outpaint_top=128,outpaint_bottom=128)),('object','object_remove',dict(anypaint_image=src,mask_box=[200,280,530,800])),('background','background_cleanup',dict(source_image=src)),('inpaint','inpaint',dict(anypaint_image=src,mask_box=[200,280,530,800])),('lora-rembg','background_remove',dict(source_image=src,background_method='lora_rembg'))]
for name,op,kw in jobs:
 m.restart();m.run('no-pins-isolated-'+name,op,**kw)
print('COMPLETE',flush=True)
