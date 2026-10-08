"""Fetch public, pinned Xet content identities; never persist bearer tokens or URLs.

Equal (xorb hash, chunk range) with equal unpacked length proves equal bytes.
Unmatched regions must still be read and compared, never assumed unchanged.
"""
import json,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from inspect_headers import SOURCES

def metadata(side,file_index,part,out,span=10*2**30):
    repo,rev=SOURCES[side];file=f'UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-{file_index:05d}-of-00006.gguf'
    target=out/f'{side}-{file_index}-{part}.json'
    if target.exists():return
    tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/{rev}/UD-Q4_K_XL?expand=true',timeout=30))
    entry=next(t for t in tree if t['path']==file)
    start=part*span;end=min(entry['size'],start+span)-1
    if start>end:return
    for attempt in range(5):
        try:
            token=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/xet-read-token/{rev}',timeout=30))
            req=urllib.request.Request(token['casUrl']+'/v2/reconstructions/'+entry['xetHash'],headers={'Authorization':'Bearer '+token['accessToken'],'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=90) as r:result=json.load(r)
            offset=start-result['offset_into_first_range'];terms=[]
            for t in result['terms']:
                if t['unpacked_length']<=0 or t['range']['start']>=t['range']['end']:raise ValueError('Invalid Xet term')
                terms.append({**t,'offset':offset});offset+=t['unpacked_length']
            if offset<end+1:raise ValueError('Incomplete reconstruction coverage')
            record={'repo':repo,'revision':rev,'file':file,'xet_hash':entry['xetHash'],'file_size':entry['size'],'start':start,'end':end+1,'terms':terms}
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record));tmp.replace(target)
            print('XET TERMS',side,file_index,part,len(terms),flush=True);return
        except Exception as e:
            print('RETRY XET',side,file_index,part,type(e).__name__,flush=True)
            if attempt==4:raise
            time.sleep(2**attempt)
def run(out):
    out.mkdir(parents=True,exist_ok=True)
    jobs=[(side,i,part) for side in SOURCES for i in range(2,7) for part in range(5)]
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda args:metadata(*args,out),jobs))
if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]))
