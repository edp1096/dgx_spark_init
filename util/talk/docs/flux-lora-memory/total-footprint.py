import measure as m
import subprocess,time,threading,json
from PIL import Image
subprocess.run(['docker','stop','flux2-klein-nvfp4-api'],check=True);time.sleep(2)
samples=[];stop=threading.Event()
def loop():
 while not stop.is_set():samples.append(m.mem());stop.wait(.05)
before=m.mem();t=threading.Thread(target=loop);t.start()
try:
 subprocess.run(['docker','start','flux2-klein-nvfp4-api'],check=True);m.wait();time.sleep(1)
 idle=m.mem()
 src=m.dataurl(Image.open(m.ROOT/'baseline-generate.png').convert('RGB'))
 m.run('trial4-total-inpaint','inpaint',anypaint_image=src,mask_box=[200,280,530,800])
finally:stop.set();t.join()
result={'before_stopped':before,'api_idle':idle,'peak_used_gib':max(d['used_gib'] for d in samples),'total_increment_gib':max(d['used_gib'] for d in samples)-before['used_gib'],'after':m.mem()}
(m.ROOT/'total-footprint.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
subprocess.run(['docker','restart','flux2-klein-nvfp4-api'],check=True);m.wait()
