"""Compare all GGUF tensors against the streamed, revision-pinned original audit."""
import hashlib,json,os,time
from pathlib import Path
from audit_original import nbytes
from inspect_headers import SOURCES,get_range

def hash_region(path,offset,length):
    h=hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while length:
            b=f.read(min(length,8<<20))
            if not b:raise EOFError(str(path))
            h.update(b);length-=len(b)
    return h.hexdigest()

def compare(headers,hashes,donor,out):
    records=[];seen=set();original_files={}
    for i in range(1,7):
        file=f'GLM-5.3-Flash-UD-Q4_K_XL-{i:05d}-of-00006.gguf'
        a=json.loads((headers/('original-'+file+'.json')).read_text())
        b=json.loads((headers/('huihui-'+file+'.json')).read_text())
        report=json.loads((hashes/('original-'+file+'.hashes.json')).read_text())
        if report['status']!='complete' or report['revision']!=SOURCES['original'][1]:raise ValueError('Original audit incomplete or stale')
        if not report.get('file_sha256'):raise ValueError('Original full-file checksum missing')
        original_files[a['file']]=report['file_sha256']
        ah={t['name']:t for t in report['tensors']};bt={t['name']:t for t in b['tensors']}
        if set(ah)!=set(bt):raise ValueError('Tensor names differ')
        path=donor/b['file']
        if path.stat().st_size!=b['size']:raise ValueError('Donor size mismatch')
        for t in a['tensors']:
            name=t['name'];d=bt[name];r=ah[name]
            if name in seen:raise ValueError('Duplicate tensor')
            seen.add(name)
            if t['shape']!=d['shape'] or t['type']!=d['type'] or nbytes(t)!=r['bytes']:raise ValueError('Tensor layout mismatch')
            digest=hash_region(path,d['offset'],r['bytes'])
            rec={'name':name,'pair':b['file'],'shape':t['shape'],'type':t['type'],'bytes':r['bytes'],'original_offset':t['offset'],'huihui_offset':d['offset'],'original_sha256':r['sha256'],'huihui_sha256':digest,'changed':digest!=r['sha256']}
            records.append(rec)
            if rec['changed']:print('CHANGED',name,'type',t['type'],flush=True)
    result={'status':'complete','sources':SOURCES,'original_file_sha256':original_files,'tensor_count':len(records),'changed_count':sum(r['changed'] for r in records),'tensors':records}
    out.write_text(json.dumps(result,indent=2));return result

def fetch_changed(report,out):
    out.mkdir(parents=True,exist_ok=True)
    for r in report['tensors']:
        if not r['changed']:continue
        path=out/(r['name']+'.bin')
        if path.exists() and hash_region(path,0,path.stat().st_size)==r['original_sha256']:continue
        tmp=path.with_suffix('.partial');h=hashlib.sha256()
        with tmp.open('wb') as f:
            for pos in range(0,r['bytes'],4<<20):
                b=get_range(*SOURCES['original'],r['pair'],r['original_offset']+pos,min(4<<20,r['bytes']-pos));f.write(b);h.update(b)
        if h.hexdigest()!=r['original_sha256']:raise ValueError('Range hash mismatch '+r['name'])
        tmp.replace(path);print('RANGE VERIFIED',r['name'],flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--headers',type=Path,required=True);p.add_argument('--hashes',type=Path,required=True);p.add_argument('--donor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ranges',type=Path,required=True);a=p.parse_args()
    report=compare(a.headers,a.hashes,a.donor,a.out);fetch_changed(report,a.ranges)
