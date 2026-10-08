"""Transfer an audited GGUF delta to NVIDIA GLM NVFP4; keep all other bytes.

A candidate is never promoted until complete structural and byte-level validation.
No activation calibration or claim of equivalence to Huihui BF16 is made.
"""
import argparse,hashlib,json,math,os,shutil,sys,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import gguf
import modelopt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from weights_core.safetensors_io import Checkpoint
from model_adapters.glm53_huihui.model_card import render as render_model_card
from model_adapters.qwen38_lil_huihui.build import decode as decode_quant,encode,NVFP4QTensor
from model_adapters.qwen38_lil_huihui.refine_scales import hash_regions

MAP={'attn_output.weight':'self_attn.o_proj','ffn_down.weight':'mlp.down_proj','ffn_down_shexp.weight':'mlp.shared_experts.down_proj'}
DT={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'U8':torch.uint8,'F8_E4M3':torch.float8_e4m3fn}

def targets(r):
    parts=r['name'].split('.')
    if len(parts)<3 or parts[0]!='blk':raise ValueError('Unsupported changed tensor '+r['name'])
    layer=int(parts[1]);suffix='.'.join(parts[2:])
    if not 0<=layer<45:raise ValueError('Unexpected layer')
    if suffix=='ffn_down_exps.weight':
        if r['shape']!=[2048,4096,288]:raise ValueError('Expert layout mismatch')
        return [f'model.language_model.layers.{layer}.mlp.experts.{e}.down_proj' for e in range(288)]
    if suffix not in MAP:raise ValueError('Unsupported changed tensor '+r['name'])
    return [f'model.language_model.layers.{layer}.'+MAP[suffix]]

def raw(t):return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
def read(cp,name):
    t=cp.tensors[name];return torch.frombuffer(bytearray(b''.join(cp.blocks(name))),dtype=DT[t['dtype']]).reshape(t['shape'])
def decode(cp,prefix):
    t=cp.tensors[prefix+'.weight']
    if t['dtype'] in ('BF16','F16','F32'):return read(cp,prefix+'.weight').float(),t['dtype']
    return decode_quant(cp,prefix)

def norm2(value):
    flat=value.reshape(-1);total=0.
    for pos in range(0,flat.numel(),1<<20):total+=float(flat[pos:pos+(1<<20)].double().square().sum())
    return total

def dot(a,b):
    x=a.reshape(-1);y=b.reshape(-1);total=0.
    for pos in range(0,x.numel(),1<<20):total+=float((x[pos:pos+(1<<20)].double()*y[pos:pos+(1<<20)].double()).sum())
    return total

def encode_best(target,cp,prefix):
    fresh,restored=encode(target,'nvfp4')
    q,scale,global_scale=NVFP4QTensor.quantize(target.to(torch.bfloat16),16,weights_scaling_factor=read(cp,prefix+'.weight_scale'),weights_scaling_factor_2=read(cp,prefix+'.weight_scale_2'))
    fixed=q.dequantize(dtype=torch.float32,scale=scale,double_scale=global_scale,block_sizes={-1:16})
    if not torch.isfinite(fixed).all():raise ValueError('Nonfinite fixed-scale reconstruction')
    error=lambda x:norm2(x-target)
    fresh_error=error(restored)
    if error(fixed)<fresh_error:return {'weight':q._quantized_data,'weight_scale':scale,'weight_scale_2':global_scale},fixed,True,fresh_error
    return fresh,restored,False,fresh_error

@contextmanager
def original_stream(a,r,proof):
    if proof is None:
        with (a.ranges/(r['name']+'.bin')).open('rb') as f:yield f
        return
    class Reader:
        def __init__(self):self.it=iter(proof.blocks(r['pair'],r['original_offset'],r['bytes']));self.buf=b'';self.pos=0
        def read(self,n):
            out=bytearray()
            while n:
                if self.pos==len(self.buf):
                    self.buf=next(self.it,b'');self.pos=0
                    if not self.buf:break
                k=min(n,len(self.buf)-self.pos);out.extend(self.buf[self.pos:self.pos+k]);self.pos+=k;n-=k
            return bytes(out)
    yield Reader()

def run(a):
    torch.set_num_threads(2)
    if any(a.output.resolve()==p.resolve() or p.resolve() in a.output.resolve().parents or a.output.resolve() in p.resolve().parents for p in [p for p in [a.base,a.donor,a.ranges,a.proof_ranges] if p is not None]):raise ValueError('Output overlaps input')
    work=a.output.with_name('.'+a.output.name+'.partial')
    if work.exists() or a.output.exists():raise FileExistsError('Output exists')
    report=json.loads(a.audit.read_text())
    if report['status']!='complete' or report['tensor_count']!=1412 or len(report['tensors'])!=1412:raise ValueError('Incomplete full audit')
    changed=[r for r in report['tensors'] if r['changed']]
    if not changed:raise ValueError('No ablit delta found')
    proof=None
    if a.proof_ranges is not None:
        from materialize import Original
        proof=Original(a.proof_ranges.parent/'headers',a.proof_ranges,a.donor)
    elif a.ranges is None:raise ValueError('Original ranges or Xet proof ranges required')
    source_manifest=json.loads(a.base_manifest.read_text())
    cp=Checkpoint(a.base);config=json.loads((a.base/'config.json').read_text())
    if config['architectures']!=['Glm5NextForConditionalGeneration']:raise ValueError('Wrong architecture')
    tc=config['text_config']
    if [tc[k] for k in ['hidden_size','num_hidden_layers','n_routed_experts']]!=[4096,45,288]:raise ValueError('Wrong GLM structure')
    for r in changed:
        for prefix in targets(r):
            if prefix+'.weight' not in cp.tensors:raise ValueError('Target missing '+prefix)
        if proof is None:
            p=a.ranges/(r['name']+'.bin')
            with p.open('rb') as f:
                if p.stat().st_size!=r['bytes'] or hashlib.file_digest(f,'sha256').hexdigest()!=r['original_sha256']:raise ValueError('Original range mismatch')
    files=sorted(set(t['shard'] for t in cp.tensors.values()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.output.parent).free<sum((a.base/f).stat().st_size for f in files)+5*2**30:raise ValueError('Insufficient disk space')
    manifest={'status':'building','runtime_validated':False,'method':'DQ(NVIDIA NVFP4) + DQ(Huihui GGUF) - DQ(Unsloth GGUF); original dtype or ModelOpt requantization','sources':report['sources'],'original_file_sha256':report.get('original_file_sha256',{}),'base_revision':source_manifest['sha'],'base_repo':source_manifest['id'],'quantizer_version':modelopt.__version__,'activation_scales':'preserved; no fresh calibration','scale_policy':'minimum reconstruction error of refreshed versus original NVIDIA NVFP4 scales','audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'started':time.time(),'tensors':[],'source_shard_hashes':{}}
    work.mkdir()
    def save():
        p=work/'transfer-manifest.json';p.with_suffix('.tmp').write_text(json.dumps(manifest,indent=2));p.with_suffix('.tmp').replace(p)
    for name in files:
        h=hashlib.sha256()
        with (a.base/name).open('rb') as src,(work/name).open('xb') as dst:
            while b:=src.read(8<<20):h.update(b);dst.write(b)
        expected=next(f for f in source_manifest['siblings'] if f['rfilename']==name)
        if h.hexdigest()!=expected['lfs']['sha256']:raise ValueError('NVIDIA source hash mismatch '+name)
        manifest['source_shard_hashes'][name]=h.hexdigest();save();print('COPIED',name,flush=True)
    metadata=[]
    for entry in source_manifest['siblings']:
        name=entry['rfilename']
        if name.endswith('.safetensors'):continue
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe metadata path')
        data=(a.base/name).read_bytes()
        digest=hashlib.sha256(data).hexdigest() if 'lfs' in entry else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        expected=entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId']
        if len(data)!=entry['size'] or digest!=expected:raise ValueError('NVIDIA metadata hash mismatch '+name)
        (work/name).parent.mkdir(parents=True,exist_ok=True);(work/name).write_bytes(data);metadata.append(name)
    mutations=set()
    for r in changed:
        names=targets(r);qtype=gguf.GGMLQuantizationType(r['type']);ha=hashlib.sha256();hb=hashlib.sha256()
        stats={'base_norm2':0.,'delta_norm2':0.,'error_norm2':0.,'target_norm2':0.,'original_match_error2':0.,'original_norm2':0.,'effective_change_norm2':0.,'delta_dot_effective':0.,'original_scales_selected':0,'fresh_error_norm2':0.}
        with original_stream(a,r,proof) as fa,(a.donor/r['pair']).open('rb') as fb:
            fb.seek(r['huihui_offset'])
            for prefix in names:
                value,kind=decode(cp,prefix)
                shape=list(reversed(r['shape']))[-2:]
                if list(value.shape)!=shape:raise ValueError('Matrix shape mismatch '+prefix)
                length=r['bytes']//len(names);ba=fa.read(length);bb=fb.read(length)
                if len(ba)!=length or len(bb)!=length:raise EOFError('Short delta matrix')
                ha.update(ba);hb.update(bb)
                qa=torch.from_numpy(gguf.dequantize(np.frombuffer(ba,dtype=np.uint8),qtype).copy()).reshape(value.shape)
                qb=torch.from_numpy(gguf.dequantize(np.frombuffer(bb,dtype=np.uint8),qtype).copy()).reshape(value.shape)
                delta=qb-qa;target=value+delta
                if not torch.isfinite(target).all():raise ValueError('Nonfinite target')
                # Catch an incompatible basis, tensor ordering or different source model.
                mismatch=float(torch.linalg.vector_norm(value-qa)/torch.linalg.vector_norm(qa).clamp_min(1e-30))
                if mismatch>0.20:raise ValueError(f'Base/GGUF mismatch {prefix}: {mismatch}')
                if kind in DT:
                    parts={'weight':target.to(DT[kind])};restored=parts['weight'].float()
                elif kind=='nvfp4':
                    parts,restored,fixed,fresh_error=encode_best(target,cp,prefix)
                    stats['original_scales_selected']+=int(fixed);stats['fresh_error_norm2']+=fresh_error
                else:parts,restored=encode(target,kind)
                for key,x in parts.items():
                    name=prefix+'.'+key;t=cp.tensors[name];b=raw(x)
                    if list(x.shape)!=t['shape'] or x.dtype!=DT[t['dtype']] or len(b)!=t['data_offsets'][1]-t['data_offsets'][0]:raise ValueError('Encoded layout mismatch '+name)
                    with (work/t['shard']).open('r+b') as f:f.seek(t['start']+t['data_offsets'][0]);f.write(b)
                    mutations.add(name)
                for k,v in [('base_norm2',value),('delta_norm2',delta),('error_norm2',target-restored),('target_norm2',target),('original_match_error2',value-qa),('original_norm2',qa),('effective_change_norm2',restored-value)]:stats[k]+=norm2(v)
                stats['delta_dot_effective']+=dot(delta,restored-value)
        if ha.hexdigest()!=r['original_sha256'] or hb.hexdigest()!=r['huihui_sha256']:raise ValueError('Delta audit hash mismatch')
        stats['requant_relative_l2']=math.sqrt(stats['error_norm2']/max(stats['target_norm2'],1e-30))
        stats['delta_effective_cosine']=stats['delta_dot_effective']/max(math.sqrt(stats['delta_norm2']*stats['effective_change_norm2']),1e-30)
        if stats['requant_relative_l2']>0.20:raise ValueError('Excessive requantization error')
        manifest['tensors'].append({'gguf_name':r['name'],'matrices':len(names),**stats});save();print('PATCHED',r['name'],flush=True)
    if not any(r['delta_norm2']>0 for r in manifest['tensors']):raise ValueError('No numerical ablit delta')
    output=Checkpoint(work)
    if cp.tensors!=output.tensors:raise ValueError('Output tensor structure changed')
    manifest['output_shard_hashes']={}
    for shard in files:
        intervals=sorted((t['start']+t['data_offsets'][0],t['start']+t['data_offsets'][1]) for n,t in cp.tensors.items() if n in mutations and t['shard']==shard)
        ah=hash_regions(a.base/shard,intervals);bh=hash_regions(work/shard,intervals)
        if ah['full']!=manifest['source_shard_hashes'][shard] or ah['outside']!=bh['outside']:raise ValueError('Source or non-target bytes changed')
        manifest['output_shard_hashes'][shard]=bh['full'];print('BYTE VERIFIED',shard,flush=True)
    for name in metadata:
        if (a.base/name).read_bytes()!=(work/name).read_bytes():raise ValueError('Metadata changed')
    manifest.update(status='candidate_verified',changed_tensors=sorted(mutations),finished=time.time());save()
    shutil.copyfile(a.audit,work/'gguf-audit.json')
    shutil.copyfile(work/'README.md',work/'README.nvidia.md')
    (work/'README.md').write_text(render_model_card((work/'README.nvidia.md').read_text()))
    work.rename(a.output);print('CANDIDATE VERIFIED',a.output,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['base','base-manifest','donor','audit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ranges',type=Path);p.add_argument('--proof-ranges',type=Path)
    run(p.parse_args())
