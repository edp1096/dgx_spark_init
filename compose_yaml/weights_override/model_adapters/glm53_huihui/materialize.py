"""Reconstruct original tensor bytes from proven shared donor ranges plus fetched gaps."""
import bisect,hashlib,json
from pathlib import Path
from audit_original import nbytes
from inspect_headers import SOURCES

class Original:
    def __init__(self,headers,proof,donor):
        plan=proof/'proof-plan.json';receipt=json.loads((proof/'complete.json').read_text())
        if receipt['status']!='complete' or hashlib.sha256(plan.read_bytes()).hexdigest()!=receipt['plan_sha256']:raise ValueError('Incomplete proof ranges')
        self.files={};self.headers=headers
        for rec in json.loads(plan.read_text())['files']:
            name=rec['file'];segments=[]
            for a,b in rec['shared']:segments.append((a,b,donor/name,rec['shift']))
            receipts=sorted(proof.glob(f'{rec["file_index"]}-*.json'))
            for path in receipts:
                m=json.loads(path.read_text());p=path.with_suffix('.bin')
                if m['file']!=name or p.stat().st_size!=m['end']-m['start']:raise ValueError('Gap layout mismatch')
                with p.open('rb') as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()!=m['sha256']:raise ValueError('Gap checksum mismatch')
                segments.append((m['start'],m['end'],p,-m['start']))
            segments.sort();h=json.loads((headers/('original-'+Path(name).name+'.json')).read_text());pos=h['data_start']
            for a,b,p,delta in segments:
                if a!=pos or b<=a:raise ValueError('Gap or overlap in reconstructed original')
                pos=b
            if pos!=h['size']:raise ValueError('Incomplete original')
            self.files[name]=segments
    def blocks(self,file,offset,length):
        segments=self.files[file];starts=[s[0] for s in segments];index=bisect.bisect_right(starts,offset)-1
        while length:
            a,b,path,delta=segments[index]
            if not a<=offset<b:raise ValueError('Uncovered tensor byte')
            take=min(length,b-offset)
            with path.open('rb') as f:
                f.seek(offset+delta);remaining=take
                while remaining:
                    data=f.read(min(8<<20,remaining))
                    if not data:raise EOFError('Truncated reconstruction source')
                    yield data;remaining-=len(data)
            offset+=take;length-=take;index+=1
    def audit(self,out):
        out.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((self.headers.parent/'original-manifest.json').read_text())
        if manifest['sha']!=SOURCES['original'][1]:raise ValueError('Original manifest revision mismatch')
        expected={m['rfilename']:m for m in manifest['siblings']}
        for header in sorted(self.headers.glob('original-*.gguf.json')):
            h=json.loads(header.read_text());records=[]
            old_path=out/(header.stem+'.hashes.json')
            old={t['name']:t['sha256'] for t in json.loads(old_path.read_text())['tensors']} if old_path.exists() else {}
            prefix=header.with_suffix('.header.bin').read_bytes()
            if len(prefix)!=min(h['data_start'],h['size']):raise ValueError('Original header size mismatch')
            full=hashlib.sha256(prefix);cursor=len(prefix)
            for t in sorted(h['tensors'],key=lambda t:t['offset']):
                if t['offset']<cursor:raise ValueError('Overlapping original tensors')
                if t['offset']>cursor:
                    for data in self.blocks(h['file'],cursor,t['offset']-cursor):full.update(data)
                size=nbytes(t);digest=hashlib.sha256()
                for data in self.blocks(h['file'],t['offset'],size):digest.update(data);full.update(data)
                if t['name'] in old and old[t['name']]!=digest.hexdigest():raise ValueError('Independent streamed hash disagrees')
                records.append({**t,'bytes':size,'sha256':digest.hexdigest()});cursor=t['offset']+size
            if cursor<h['size']:
                for data in self.blocks(h['file'],cursor,h['size']-cursor):full.update(data)
            if full.hexdigest()!=expected[h['file']]['lfs']['sha256']:raise ValueError('Reconstructed original fails published SHA256 '+h['file'])
            result={'status':'complete','revision':h['revision'],'repo':h['repo'],'file':h['file'],'file_sha256':full.hexdigest(),'method':'Xet shared-range proof + original gaps; full reconstructed file SHA256 matches Hugging Face LFS','tensors':records}
            p=out/(header.stem+'.hashes.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(p)
            print('ORIGINAL FULL SHA256 VERIFIED',h['file'],flush=True)
    def changed(self,report,out):
        out.mkdir(parents=True,exist_ok=True)
        for r in report['tensors']:
            if not r['changed']:continue
            p=out/(r['name']+'.bin');tmp=p.with_suffix('.partial');h=hashlib.sha256()
            with tmp.open('wb') as f:
                for data in self.blocks(r['pair'],r['original_offset'],r['bytes']):f.write(data);h.update(data)
            if h.hexdigest()!=r['original_sha256']:raise ValueError('Original reconstruction changed')
            tmp.replace(p);print('DELTA ORIGINAL VERIFIED',r['name'],flush=True)
