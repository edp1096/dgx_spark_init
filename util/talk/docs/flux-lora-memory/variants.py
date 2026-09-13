import measure as m
import subprocess,json
from pathlib import Path
from PIL import Image
root=m.ROOT
original=(root/'base_start.original.sh').read_text()
src=m.dataurl(Image.open(root/'baseline-generate.png').convert('RGB'))
small=m.dataurl(Image.open(root/'baseline-generate.png').convert('RGB').resize((768,768)))
for variant,flags in [('classic','--cache-classic'),('no-pins','--disable-pinned-memory')]:
 text=original.replace('--disable-auto-launch \\', '--disable-auto-launch '+flags+' \\')
 assert text != original
 path=root/'base_start.variant.sh';path.write_text(text);path.chmod(0o755)
 subprocess.run(['docker','cp',str(path),'flux2-klein-nvfp4-api:/opt/nvfp4-api/base_start.sh'],check=True)
 m.restart()
 m.run(variant+'-base','identity_edit',source_image=src)
 for cycle in range(3):
  m.run(f'{variant}-{cycle}-outpaint','outpaint',anypaint_image=small,outpaint_left=128,outpaint_right=128,outpaint_top=128,outpaint_bottom=128)
  m.run(f'{variant}-{cycle}-object','object_remove',anypaint_image=src,mask_box=[200,280,530,800])
  m.run(f'{variant}-{cycle}-background','background_cleanup',source_image=src)
 print(variant+' COMPLETE',flush=True)
print('COMPLETE',flush=True)
