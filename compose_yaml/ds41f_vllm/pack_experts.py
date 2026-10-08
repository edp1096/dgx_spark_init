"""Prepare immutable rank-local expert files once; resume at layer boundaries."""
import argparse,hashlib,json,os,struct,time
from pathlib import Path
import torch
from expert_store import ExpertStore
from b12x_layout import FORMAT,from_raw,tensors

HEADER_BYTES=65536
p=argparse.ArgumentParser()
p.add_argument('model');p.add_argument('output');p.add_argument('--rank',type=int,required=True)
p.add_argument('--layers',type=int,default=43)
a=p.parse_args()
store=ExpertStore(a.model,rank=a.rank)
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
revision=Path(a.model).name
for layer in range(a.layers):
    family,index,count=('layers',layer,384) if layer<40 else ('mtp',layer-40,128)
    final=root/f'layer-{layer:02d}.bin'
    if final.exists():
        with final.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(size))
        assert (meta['format'],meta['rank'],meta['revision'])==(FORMAT,a.rank,revision)
        assert final.stat().st_size==HEADER_BYTES+meta['record_bytes']*count
        print('SKIP',layer,flush=True);continue
    start=time.monotonic();tmp=final.with_suffix('.partial')
    hashes=[];record_bytes=None;layout=None
    with tmp.open('w+b',buffering=0) as f:
        f.seek(HEADER_BYTES)
        for expert in range(count):
            prepared=from_raw(store.expert(index,expert,family=family))
            parts=[t.cpu().contiguous() for t in tensors(prepared)]
            blobs=[t.numpy().tobytes() for t in parts]
            if record_bytes is None:
                record_bytes=sum(len(b) for b in blobs)
                layout=[{'shape':list(t.shape[1:]),'bytes':len(b),'dtype':str(t.dtype)} for t,b in zip(parts,blobs)]
            assert sum(len(b) for b in blobs)==record_bytes
            digest=hashlib.blake2b(digest_size=16)
            for b in blobs:
                digest.update(b)
                view=memoryview(b)
                while view:
                    written=f.write(view)
                    if not written: raise OSError('Short packed write')
                    view=view[written:]
            hashes.append(digest.hexdigest())
            del prepared,parts,blobs
        meta={'format':FORMAT,'rank':a.rank,'revision':revision,'layer':layer,'experts':count,
              'record_bytes':record_bytes,'layout':layout,'hashes':hashes}
        payload=json.dumps(meta,separators=(',',':')).encode()
        assert len(payload)+8<=HEADER_BYTES
        f.seek(0);f.write(struct.pack('<Q',len(payload))+payload)
        os.fsync(f.fileno())
    os.replace(tmp,final)
    print(json.dumps({'layer':layer,'seconds':time.monotonic()-start,'bytes':final.stat().st_size,'rank':a.rank}),flush=True)
print('PACK_COMPLETE',flush=True)
