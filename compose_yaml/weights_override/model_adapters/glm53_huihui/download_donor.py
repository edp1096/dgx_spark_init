"""Bounded, resumable range download into a separate staging area.

Only SHA256-verified complete blobs are atomically published to the HF cache.
Existing HF .incomplete files are never modified by this worker.
"""
import hashlib,json,os,time,threading,sys,fcntl
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from inspect_headers import get_range,SOURCES

CHUNK=1<<20
JOB=Path.home()/'.cache/model-download-jobs/glm53-huihui'
SIDE=sys.argv[1] if len(sys.argv)>1 else 'huihui'
REPO_REV={'huihui':SOURCES['huihui'],'nvidia':('nvidia/GLM-5.3-Flash-NVFP4','09b04e5e74bca08ca8549fc736d4cdd8624bfde3')}
CACHE=Path.home()/'.cache/huggingface/hub'/('models--'+REPO_REV[SIDE][0].replace('/','--'))

def run():
    repo,rev=REPO_REV[SIDE];manifest=json.loads((JOB/(SIDE+'-manifest.json')).read_text())
    stage=JOB/('donor-range-stage' if SIDE=='huihui' else 'nvidia-range-stage');stage.mkdir(exist_ok=True)
    files=[]
    for m in manifest['siblings']:
        if 'lfs' not in m:continue
        digest=m['lfs']['sha256'];blob=CACHE/'blobs'/digest
        if blob.exists():
            with blob.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Existing donor blob corrupt')
            continue
        part=stage/(digest+'.partial');bitmap=stage/(digest+'.bitmap');count=(m['size']+CHUNK-1)//CHUNK
        bits=bytearray(bitmap.read_bytes()) if bitmap.exists() else bytearray(count)
        if len(bits)!=count or any(b not in (0,1) for b in bits):raise ValueError('Bad bitmap')
        if not part.exists() and any(bits):raise ValueError('Bitmap without data')
        fd=os.open(part,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,m['size'])
        files.append({'meta':m,'part':part,'bitmap':bitmap,'bits':bits,'fd':fd,'lock':threading.Lock(),'completed':0})
    def save(f):
        tmp=f['bitmap'].with_suffix('.tmp');tmp.write_bytes(f['bits']);tmp.replace(f['bitmap'])
    def fetch(item):
        f,i=item;m=f['meta'];start=i*CHUNK;length=min(CHUNK,m['size']-start)
        data=get_range(repo,rev,m['rfilename'],start,length)
        n=os.pwrite(f['fd'],data,start)
        if n!=length:raise IOError('Short pwrite')
        with f['lock']:
            f['bits'][i]=1;f['completed']+=1
            if f['completed']%128==0:save(f)
    def tasks():
        for i in range(max((len(f['bits']) for f in files),default=0)):
            for f in files:
                if i<len(f['bits']) and not f['bits'][i]:yield f,i
    errors=[];iterator=iter(tasks());last=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=set()
            for _ in range(1):
                item=next(iterator,None)
                if item is not None:pending.add(pool.submit(fetch,item))
            while pending:
                finished,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for future in finished:
                    try:future.result()
                    except Exception as e:errors.append(type(e).__name__)
                    item=next(iterator,None)
                    if item is not None:pending.add(pool.submit(fetch,item))
                if time.monotonic()-last>30:
                    for f in files:
                        with f['lock']:save(f)
                    print('DONOR RANGES GiB',round(sum(sum(f['bits'])*CHUNK for f in files)/2**30,3),'failed_chunks',len(errors),flush=True);last=time.monotonic()
    finally:
        for f in files:
            save(f);os.fsync(f['fd']);os.close(f['fd'])
    if errors:raise RuntimeError(f'{len(errors)} chunks require retry')
    for f in files:
        m=f['meta'];digest=m['lfs']['sha256']
        if not all(f['bits']):raise ValueError('Incomplete bitmap')
        with f['part'].open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:
            f['bitmap'].unlink();raise ValueError('Donor SHA256 mismatch; bitmap reset')
        blob=CACHE/'blobs'/digest;blob.parent.mkdir(parents=True,exist_ok=True);f['part'].replace(blob)
        link=CACHE/'snapshots'/rev/m['rfilename'];link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(os.path.relpath(blob,link.parent))
        print('DONOR BLOB SHA256 VERIFIED',m['rfilename'],flush=True)
    for m in manifest['siblings']:
        if 'lfs' in m:continue
        link=CACHE/'snapshots'/rev/m['rfilename']
        if link.is_file():data=link.read_bytes()
        else:
            response=requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{m["rfilename"]}',timeout=60);response.raise_for_status();data=response.content
        digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if len(data)!=m['size'] or digest!=m['blobId']:raise ValueError('Metadata Git hash mismatch '+m['rfilename'])
        blob=CACHE/'blobs'/digest
        if not blob.exists():
            temp=stage/(digest+'.metadata');temp.write_bytes(data);temp.replace(blob)
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(os.path.relpath(blob,link.parent))
    (JOB/(SIDE+'-ranges-complete.json')).write_text(json.dumps({'status':'complete','revision':rev}))
if __name__=='__main__':
    process_lock=(JOB/(SIDE+'-download.lock')).open('a')
    fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if SIDE=='huihui':
        while not (JOB/'proof-ranges/complete.json').exists():
            time.sleep(2)
    for attempt in range(5):
        try:run();break
        except Exception as e:
            print('RETRY DONOR RANGES',attempt+1,type(e).__name__,str(e),flush=True)
            if attempt==4:raise
            time.sleep(30)
