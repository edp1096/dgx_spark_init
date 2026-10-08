"""Resume-safe download/audit gate followed by conversion; never auto-deploys."""
import hashlib,json,subprocess,time,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
JOB=Path.home()/'.cache/model-download-jobs/glm53-huihui'
HF=Path.home()/'.cache/huggingface'
BASE=HF/'hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3'
DONOR=HF/'hub/models--huihui-ai--GLM-5.3-Flash-abliterated-GGUF/snapshots/50d39f500cbd1d0478295da89d417607521d378b'
OUTPUT=HF/'edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4'

def state(phase,**kwargs):
    p=JOB/'pipeline-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'phase':phase,'time':time.time(),**kwargs},indent=2));tmp.replace(p);print(phase,kwargs,flush=True)
def ready(root,side):
    m=json.loads((JOB/(side+'-manifest.json')).read_text())
    return all((root/f['rfilename']).is_file() and (root/f['rfilename']).stat().st_size==f['size'] for f in m['siblings'])
def verify_donor():
    receipt=JOB/'donor-verified.json'
    m=json.loads((JOB/'huihui-manifest.json').read_text())
    results={}
    for f in m['siblings']:
        p=DONOR/f['rfilename']
        if 'lfs' in f:
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=f['lfs']['sha256']:raise ValueError('Donor SHA256 mismatch '+f['rfilename'])
            results[f['rfilename']]=digest
    receipt.write_text(json.dumps(results,indent=2))
def run():
    if OUTPUT.exists():
        m=json.loads((OUTPUT/'transfer-manifest.json').read_text())
        if m['status']!='candidate_verified':raise ValueError('Existing candidate is not verified')
        for name,expected in m['output_shard_hashes'].items():
            with (OUTPUT/name).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('Existing candidate checksum mismatch')
        state('candidate_verified_runtime_pending',output=str(OUTPUT));return
    state('waiting_for_downloads_and_original_audit')
    while True:
        reports=list((JOB/'original-hashes').glob('*.hashes.json'))
        audit_ready=(JOB/'proof-ranges/complete.json').exists()
        if ready(BASE,'nvidia') and ready(DONOR,'huihui') and audit_ready and (JOB/'hub-downloader-cleaned.json').exists():break
        time.sleep(30)
    state('verifying_donor');verify_donor()
    from compare import compare,fetch_changed
    from materialize import Original
    state('hashing_proven_original')
    original=Original(JOB/'headers',JOB/'proof-ranges',DONOR)
    original.audit(JOB/'original-hashes')
    state('comparing_all_tensors')
    audit=JOB/'audit.json'
    report=compare(JOB/'headers',JOB/'original-hashes',DONOR,audit)
    state('delta_audit_complete',changed=report['changed_count'])
    state('building_candidate')
    cmd=['docker','run','--rm','--name','glm53-huihui-convert','--network','none','--memory','6g','--cpus','2','--entrypoint','python3','-e','PYTHONPATH=/work:/deps','-v',str(ROOT)+':/work:ro','-v',str(HF)+':/hf','-v',str(JOB)+':/job:ro','-v',str(Path.home()/'.cache/model-tools/qwen38fn-huihui-venv/lib/python3.12/site-packages/gguf')+':/deps/gguf:ro','dgx-sglang-qwen38-qad:sm121-v5','/work/model_adapters/glm53_huihui/build.py','--base','/hf/'+str(BASE.relative_to(HF)),'--base-manifest','/job/nvidia-manifest.json','--donor','/hf/'+str(DONOR.relative_to(HF)),'--proof-ranges','/job/proof-ranges','--audit','/job/audit.json','--output','/hf/'+str(OUTPUT.relative_to(HF))]
    subprocess.run(cmd,check=True)
    retained=sum(r['bytes'] for r in report['tensors'] if r['changed'])
    if shutil.disk_usage(JOB).free>retained+5*2**30:
        state('retaining_only_original_delta_tensors',bytes=retained)
        original.changed(report,JOB/'original-ranges')
        receipt={'status':'complete','audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),'bytes':retained}
        (JOB/'original-ranges/status.json').write_text(json.dumps(receipt,indent=2))
        for p in (JOB/'proof-ranges').glob('*.bin'):p.unlink()
        (JOB/'proof-ranges/data-evicted-after-candidate.json').write_text(json.dumps(receipt,indent=2))
    state('candidate_verified_runtime_pending',output=str(OUTPUT))
if __name__=='__main__':
    try:run()
    except Exception as e:state('failed',error=str(e));raise
