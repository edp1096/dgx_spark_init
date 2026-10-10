"""Pinned, checksum-verified EXL3 download; hard-link identical existing HF blobs."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

ROOT = Path(os.environ.get('EXL3_DOWNLOAD_ROOT', '/home/edp1096/.cache/model-download-jobs/velogb10-q4-vs-nvfp4-20261010'))
CACHE = Path('/home/edp1096/.cache/huggingface/hub')
META = json.loads((ROOT/os.environ.get('EXL3_DOWNLOAD_METADATA', 'q4-remote-metadata.json')).read_text())
REPO, REV = META['id'], META['sha']
BASE = CACHE/('models--'+REPO.replace('/', '--'))
SNAP = BASE/'snapshots'/REV
LOCK = threading.RLock()
STATE = dict(repo=REPO, revision=REV, started=time.time(), completed={}, active={}, errors=[])


def save():
    with LOCK:
        (ROOT/'download-status.tmp').write_text(json.dumps(STATE, indent=2))
        (ROOT/'download-status.tmp').replace(ROOT/'download-status.json')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(8 << 20): h.update(data)
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def fetch(entry):
    name, size = entry['rfilename'], entry['size']
    sha = entry.get('lfs', {}).get('sha256')
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
    link = SNAP/name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not sha:
        with urllib.request.urlopen(url, timeout=90) as response: data=response.read()
        assert len(data)==size
        link.write_bytes(data)
        STATE['completed'][name] = dict(bytes=size, reused=False)
        save(); return
    final = BASE/'blobs'/sha
    reused = final.exists()
    if not final.exists():
        for other in CACHE.glob('models--*/blobs/'+sha):
            if other.is_file() and other.stat().st_size==size:
                os.link(other, final); reused=True; break
    if not final.exists():
        partial=final.with_suffix('.incomplete')
        fd=os.open(partial, os.O_RDWR|os.O_CREAT, 0o644)
        os.ftruncate(fd, size)
        chunk=64 << 20; done=set()
        progress=ROOT/(name+'.progress.json')
        progress.parent.mkdir(parents=True, exist_ok=True)
        if progress.exists(): done=set(json.loads(progress.read_text())['done'])
        local=threading.Lock()
        def part(i):
            a=i*chunk; b=min(size, a+chunk)-1
            for attempt in range(8):
                try:
                    req=urllib.request.Request(url+f'?audit_chunk={i}', headers={'Range':f'bytes={a}-{b}'})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        assert response.status==206 and response.headers['Content-Range']==f'bytes {a}-{b}/{size}'
                        off=a
                        while data:=response.read(1 << 20):
                            assert off+len(data)<=b+1
                            assert os.pwrite(fd, data, off)==len(data); off+=len(data)
                        assert off==b+1
                    with local:
                        done.add(i)
                        progress.write_text(json.dumps(dict(done=sorted(done), bytes=sum(min(chunk,size-j*chunk) for j in done), total=size)))
                        STATE['active'][name]=dict(bytes=sum(min(chunk,size-j*chunk) for j in done), total=size)
                        save()
                    return
                except Exception as e:
                    print('retry', name, i, attempt, type(e).__name__, flush=True)
                    if attempt==7: raise
                    time.sleep(min(20, 2*(attempt+1)))
        print('DOWNLOAD', name, size, flush=True)
        try:
            with ThreadPoolExecutor(max_workers=int(os.environ.get("EXL3_DOWNLOAD_STREAMS", "4"))) as pool:
                list(pool.map(part, (i for i in range((size+chunk-1)//chunk) if i not in done)))
            os.fsync(fd)
        finally: os.close(fd)
        assert digest(partial)==sha, name
        partial.replace(final)
    else:
        assert final.stat().st_size==size and digest(final)==sha, name
    if not link.exists(): link.symlink_to(os.path.relpath(final, link.parent))
    STATE['completed'][name]=dict(bytes=size, reused=reused, sha256=sha, verified=True)
    STATE['active'].pop(name, None); save()
    print('VERIFIED', name, 'reused' if reused else 'downloaded', flush=True)


def main():
    SNAP.mkdir(parents=True, exist_ok=True); (BASE/'blobs').mkdir(exist_ok=True)
    with (ROOT/'download.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        save()
        try:
            with ThreadPoolExecutor(max_workers=int(os.environ.get("EXL3_DOWNLOAD_FILES", "3"))) as pool: list(pool.map(fetch, META['siblings']))
            (BASE/'refs').mkdir(exist_ok=True); (BASE/'refs'/'main').write_text(REV)
            STATE.update(complete=True, finished=time.time(), snapshot=str(SNAP));save()
        except Exception as e:
            STATE['errors'].append(repr(e));save();raise


if __name__=='__main__':main()
