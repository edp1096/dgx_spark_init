"""Official Hugging Face downloader, Xet enabled, one file and transfer at a time."""
import os,sys,json,time,hashlib,fcntl
from pathlib import Path
venv=Path.home()/'.cache/model-tools/qwen38fn-huihui-venv/bin/python'
if Path(sys.executable)!=venv:
    os.execv(str(venv),[str(venv),str(Path(__file__).resolve())])
os.environ['HF_HUB_DISABLE_XET']='0'
os.environ['HF_XET_HIGH_PERFORMANCE']='0'
os.environ['HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS']='1'
os.environ['HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY']='1'
os.environ['HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY']='1'
os.environ['HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._runtime import is_xet_available
import hf_xet
job=Path.home()/'.cache/model-download-jobs/glm53-huihui'
lock=(job/'huihui-download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert is_xet_available(),'Xet unavailable; refusing HTTP fallback'
c=hf_xet.XetConfig()
assert c.get('client.ac_max_download_concurrency')==1
m=json.loads((job/'huihui-manifest.json').read_text())
def status(phase,**kw):
    p=job/'xet-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'phase':phase,'time':time.time(),'engine':'official huggingface_hub + hf_xet','max_download_concurrency':1,**kw},indent=2));t.replace(p)
    print(phase,kw,flush=True)
status('started',revision=m['sha'])
for entry in m['siblings']:
    name=entry['rfilename'];status('downloading',file=name)
    p=Path(hf_hub_download(repo_id=m['id'],filename=name,revision=m['sha'],cache_dir=str(Path.home()/'.cache/huggingface/hub')))
    if 'lfs' in entry:
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        expected=entry['lfs']['sha256']
    else:
        b=p.read_bytes();digest=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();expected=entry['blobId']
    if p.stat().st_size!=entry['size'] or digest!=expected:raise ValueError('Downloaded checksum mismatch: '+name)
    status('file_verified',file=name)
(job/'huihui-ranges-complete.json').write_text(json.dumps({'status':'complete','revision':m['sha'],'engine':'hf_xet'}))
status('complete')
