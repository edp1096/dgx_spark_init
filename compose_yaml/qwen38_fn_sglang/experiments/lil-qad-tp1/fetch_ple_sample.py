"""Fetch tiny, exact tensor slices from a pinned safetensors checkpoint."""
import hashlib,json,pathlib,struct,sys,urllib.request
base='https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/resolve/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd/'
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
def fetch(file,lo,hi):
    # Separate URL avoids caches that do not vary their objects on Range.
    request=urllib.request.Request(base+file+f'?slice={lo}-{hi}',headers={'Range':f'bytes={lo}-{hi}'})
    with urllib.request.urlopen(request,timeout=45) as r:
        if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {lo}-'):
            raise RuntimeError('Server did not honor byte range')
        data=r.read(hi-lo+2)
    if len(data)!=hi-lo+1:raise RuntimeError('Truncated tensor slice')
    return data
prefix='model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
manifest={}
for file,items in [('model-00002-of-00036.safetensors',[('shard_0.weight','weights.bin',80),('shard_0.weight_scale','scales.bin',10)]),('model-00034-of-00036.safetensors',[('weight_scale_2','global.bin',4)])]:
    n=struct.unpack('<Q',fetch(file,0,7))[0]
    if n>8*2**20:raise RuntimeError('Unexpected safetensors header size')
    header=json.loads(fetch(file,8,7+n))
    for suffix,name,stride in items:
        entry=header[prefix+suffix];offset=8+n+entry['data_offsets'][0]
        rows=[0,1,100,entry['shape'][0]-1] if suffix!='weight_scale_2' else [0]
        data=b''.join(fetch(file,offset+r*stride,offset+(r+1)*stride-1) for r in rows)
        (out/name).write_bytes(data);manifest[name]={'source':file,'tensor':prefix+suffix,'rows':rows,'sha256':hashlib.sha256(data).hexdigest(),'tensor_shape':entry['shape'],'dtype':entry['dtype']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
