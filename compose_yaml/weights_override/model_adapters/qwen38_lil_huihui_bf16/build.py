"""Transfer audited BF16 deltas into the pinned step-5500 hybrid QAD.

Writes an independent, unqualified candidate. All other checkpoint bytes remain
unchanged. Serving configuration and promotion are separate operations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import torch
from weights_core.safetensors_io import Checkpoint
from quantize import DTYPES, blob, decode, encode, read
from fetch_audit import atomic_json

QAD_REVISION = '6909a5bed089a48fa07e956d3915af2537de9368'
DONOR_REVISION = '298f94632b784e26a7fe576114f82066689d5baa'
ORIGINAL_REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
OUTPUT = re.compile(r'model\.language_model\.layers\.(\d+)\.(linear_attn\.out_proj\.weight|self_attn\.o_proj\.weight|mlp\.shared_expert\.down_proj\.weight|mlp\.experts\.down_proj)')


def file_hash(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def complement_hash(path, intervals):
    """Hash exact bytes outside approved, disjoint mutation ranges."""
    h = hashlib.sha256(); cursor = 0
    with path.open('rb') as f:
        for start, end in sorted(intervals) + [(path.stat().st_size, path.stat().st_size)]:
            if not cursor <= start <= end <= path.stat().st_size:
                raise ValueError('Overlapping/out-of-bounds mutation ranges')
            f.seek(cursor)
            left = start-cursor
            while left:
                b = f.read(min(left, 4<<20))
                if not b: raise ValueError('Truncated shard')
                h.update(b); left -= len(b)
            cursor = end
    return h.hexdigest()


def matrices(row):
    match = OUTPUT.fullmatch(row['name'])
    if not match or not 0 <= int(match[1]) < 48 or row['dtype'] != 'BF16':
        raise ValueError('Unaudited transfer target: ' + row['name'])
    if match[2] == 'mlp.experts.down_proj':
        if row['shape'] != [512,2560,640]: raise ValueError('Unexpected fused expert geometry')
        return [(f'model.language_model.layers.{match[1]}.mlp.experts.{i}.down_proj', [2560,640]) for i in range(512)]
    if len(row['shape']) != 2 or row['shape'][0] != 2560:
        raise ValueError('Unexpected output projection geometry')
    return [(row['name'].removesuffix('.weight'), row['shape'])]


def run(args):
    torch.set_num_threads(args.threads)
    deadline=time.monotonic()+12*3600
    selection_path=args.audit.parent/'xet-audit.json'
    streaming=args.stream_inputs and not args.audit.exists()
    audit = json.loads((selection_path if streaming else args.audit).read_text())
    if audit['sources'] != {'original':ORIGINAL_REVISION, 'huihui':DONOR_REVISION}:
        raise ValueError('Unexpected BF16 source revisions')
    if streaming:
        if audit['status']!='metadata_audit_complete' or any(not r.get('changed_confirmed') for r in audit['tensors'] if r['status']!='equal'):
            raise ValueError('Streaming requires a complete selection audit and confirmed changes')
        records=[dict(r,changed=True) for r in audit['tensors'] if r['status']!='equal']
    else:
        if audit['status']!='complete':raise ValueError('A complete pinned BF16 audit is required')
        records=[r for r in audit['tensors'] if r['changed']]
    receipt = json.loads((args.base/'download-receipt.json').read_text())
    if receipt['status'] != 'verified' or receipt['revision'] != QAD_REVISION:
        raise ValueError('Unexpected QAD base')
    if not records: raise ValueError('No changed BF16 tensors')
    for r in records:
        matrices(r)
    output = args.output.resolve(); base = args.base.resolve()
    if output == base or output in base.parents or base in output.parents:
        raise ValueError('Output and source must be independent')
    work = output.with_name('.'+output.name+'.partial')
    if output.exists() or work.exists(): raise FileExistsError('Refusing existing output/partial')
    files = [r for r in receipt['files'] if r['rfilename'].endswith('.safetensors')]
    if shutil.disk_usage(output.parent).free < sum(r['size'] for r in files) + (4<<30):
        raise ValueError('Insufficient disk space for an independent candidate')
    cp = Checkpoint(base)
    # Validate every destination before making an output directory.
    for r in records:
        for prefix, shape in matrices(r):
            spec = cp.tensors[prefix+'.weight']
            logical = list(spec['shape'])
            if spec['dtype'] == 'U8': logical[-1] *= 2
            if logical != shape: raise ValueError('Source/QAD geometry mismatch: '+prefix)
    work.mkdir()
    report = dict(status='copying', runtime_validated=False, base_revision=QAD_REVISION,
        sources=audit['sources'], audit_sha256=None if streaming else file_hash(args.audit),
        selection_audit_sha256=file_hash(selection_path), streaming_inputs=streaming,
        method='ModelOpt(BF16(DQ(QAD5500) + Huihui_BF16 - Qwen_BF16)); native HF tensor order',
        activation_scales='unchanged; no fresh calibration',
        scale_policy='MXFP8 refreshed; NVFP4 minimum target reconstruction error of original versus refreshed scales',
        converter_sha256=file_hash(Path(__file__)), cpu_threads=args.threads, started=time.time(),
        source_shard_hashes={}, output_shard_hashes={}, tensors=[], changed_tensors=[])
    def save(): atomic_json(work/'transfer-manifest.json',report)
    save()
    for r in files:
        src, dst = base/r['rfilename'], work/r['rfilename']
        h = hashlib.sha256()
        with src.open('rb') as fi, dst.open('xb') as fo:
            for b in iter(lambda:fi.read(4<<20),b''): h.update(b); fo.write(b)
        if h.hexdigest() != r['lfs']['sha256'] or os.path.samefile(src,dst):
            raise ValueError('Base differs from pinned SHA256 or output is aliased')
        report['source_shard_hashes'][r['rfilename']] = h.hexdigest()
        save(); print('COPIED',r['rfilename'],flush=True)
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith('.safetensors') and p.name not in ['README.md','download-receipt.json','export-manifest.json','HYBRID.json']:
            shutil.copyfile(p,work/p.name)
    for name in ['export-manifest.json','HYBRID.json']:
        if (base/name).exists():shutil.copyfile(base/name,work/('upstream-'+name))
    config=json.loads((base/'config.json').read_text())
    ple_weights=[spec for name,spec in cp.tensors.items() if 'ngram_embedding.shard_' in name and name.endswith('.weight')]
    if len(ple_weights)!=128 or any(t['dtype']!='U8' or t['shape'][1]!=80 for t in ple_weights):
        raise ValueError('Unexpected PLE representation; cannot annotate loader metadata')
    before=config['text_config'].get('ple_embedding_dtype')
    if before not in (None,'nvfp4'):raise ValueError('Contradictory PLE dtype metadata')
    if before is None:
        shutil.copyfile(base/'config.json',work/'upstream-config.json')
        config['text_config']['ple_embedding_dtype']='nvfp4'
        atomic_json(work/'config.json',config)
        report['metadata_adjustments']={'text_config.ple_embedding_dtype':{
            'before':None,'after':'nvfp4','reason':'Upstream hybrid export omits the dtype tag while retaining 128 packed NVFP4 PLE shards.'}}
    index=json.loads((base/'model.safetensors.index.json').read_text())
    payload=sum(t['data_offsets'][1]-t['data_offsets'][0] for t in cp.tensors.values())
    old_size=index.get('metadata',{}).get('total_size')
    if old_size!=payload:
        index.setdefault('metadata',{})['total_size']=payload
        atomic_json(work/'model.safetensors.index.json',index)
        report.setdefault('metadata_adjustments',{})['index.metadata.total_size']={
            'before':old_size,'after':payload,'reason':'Recomputed tensor payload bytes; tensor map, shapes and dtypes unchanged.'}
    report['status']='converting';save()
    intervals = {}; changed = set()
    for r in records:
        if streaming:
            selected=args.audit.parent/'selected-receipts'/(hashlib.sha256(r['name'].encode()).hexdigest()+'.json')
            if not selected.exists():
                report['status']='waiting_for_input';save();print('WAITING INPUT',r['name'],flush=True)
            while not selected.exists():
                if time.monotonic()>deadline:raise TimeoutError('Input wait exceeded 12 hours')
                time.sleep(5)
            row=json.loads(selected.read_text())
            if not row['changed'] or any(row[k]!=r[k] for k in ['name','shape','dtype','bytes']):
                raise ValueError('Downloaded tensor contradicts selection audit')
            r=row;report['status']='converting';save()
        for side in ['original','huihui']:
            path=args.audit.parent/side/(r['name']+'.bf16')
            if path.stat().st_size!=r['bytes']:raise ValueError('Wrong BF16 tensor length')
        input_hashes={side:hashlib.sha256() for side in ['original','huihui']}
        stats = {k:0.0 for k in ['base_norm2','delta_norm2','target_norm2','error_norm2','effective_change_norm2','delta_dot_effective']}
        stats.update(matrices=0, original_scales_selected=0, skipped_identical_matrices=0)
        with (args.audit.parent/'original'/(r['name']+'.bf16')).open('rb') as fa, (args.audit.parent/'huihui'/(r['name']+'.bf16')).open('rb') as fb:
            for prefix,shape in matrices(r):
                n = math.prod(shape)*2
                aa,bb=fa.read(n),fb.read(n)
                if len(aa)!=n or len(bb)!=n:raise ValueError('Truncated BF16 tensor')
                input_hashes['original'].update(aa);input_hashes['huihui'].update(bb)
                original=torch.frombuffer(bytearray(aa),dtype=torch.bfloat16).reshape(shape).float()
                donor=torch.frombuffer(bytearray(bb),dtype=torch.bfloat16).reshape(shape).float()
                if not torch.isfinite(original).all() or not torch.isfinite(donor).all():raise ValueError('Nonfinite BF16 source')
                if torch.equal(original,donor):
                    stats['skipped_identical_matrices']+=1
                    continue
                value,kind=decode(cp,prefix)
                delta=donor-original;target=value+delta
                fixed=(read(cp,prefix+'.weight_scale'),read(cp,prefix+'.weight_scale_2')) if kind=='nvfp4' else None
                parts,restored,choice=encode(target,kind,fixed)
                for suffix,tensor in parts.items():
                    name=prefix+'.'+suffix;spec=cp.tensors[name];raw=blob(tensor)
                    if list(tensor.shape)!=spec['shape'] or tensor.dtype!=DTYPES[spec['dtype']] or len(raw)!=spec['data_offsets'][1]-spec['data_offsets'][0]:
                        raise ValueError('Quantized layout changed: '+name)
                    start,end=(spec['start']+x for x in spec['data_offsets'])
                    with (work/spec['shard']).open('r+b') as f:f.seek(start);f.write(raw)
                    changed.add(name);intervals.setdefault(spec['shard'],[]).append((start,end))
                for k,x in [('base_norm2',value),('delta_norm2',delta),('target_norm2',target),('error_norm2',target-restored),('effective_change_norm2',restored-value)]:
                    stats[k]+=float(x.double().square().sum())
                stats['delta_dot_effective']+=float((delta.double()*(restored-value).double()).sum())
                stats['matrices']+=1;stats['original_scales_selected']+=choice=='original'
                if stats['matrices']%64==0:print('MATRICES',r['name'],stats['matrices'],flush=True)
            if fa.read(1) or fb.read(1):raise ValueError('Unconsumed BF16 tensor bytes')
        if any(input_hashes[side].hexdigest()!=r[side+'_sha256'] for side in input_hashes):
            raise ValueError('Consumed tensor bytes differ from verified download receipt')
        stats['requant_relative_l2']=math.sqrt(stats['error_norm2']/max(stats['target_norm2'],1e-300))
        stats['delta_relative_l2']=math.sqrt(stats['delta_norm2']/max(stats['base_norm2'],1e-300))
        stats['delta_effective_cosine']=stats['delta_dot_effective']/max(math.sqrt(stats['delta_norm2']*stats['effective_change_norm2']),1e-300)
        if stats['requant_relative_l2']>.2:raise ValueError('Excessive reconstruction error')
        report['tensors'].append(dict(name=r['name'],source_tensor_sha256={s:input_hashes[s].hexdigest() for s in input_hashes},**stats));save()
        print('CONVERTED',r['name'],stats['requant_relative_l2'],flush=True)
    while not args.audit.exists():
        if time.monotonic()>deadline:raise TimeoutError('Final audit did not complete')
        time.sleep(5)
    completed=json.loads(args.audit.read_text())
    if completed['status']!='complete' or completed['sources']!=report['sources'] or {r['name'] for r in completed['tensors'] if r['changed']}!={r['name'] for r in records}:
        raise ValueError('Final audit does not match the transfer plan')
    completed_rows={r['name']:r for r in completed['tensors']}
    for converted in report['tensors']:
        if any(converted['source_tensor_sha256'][s]!=completed_rows[converted['name']][s+'_sha256'] for s in ['original','huihui']):
            raise ValueError('Final audit hashes differ from consumed source bytes')
    report['audit_sha256']=file_hash(args.audit)
    if Checkpoint(work).tensors!=cp.tensors:raise ValueError('Checkpoint headers/index changed')
    report['status']='verifying';save()
    for r in files:
        name=r['rfilename'];spans=intervals.get(name,[])
        if complement_hash(base/name,spans)!=complement_hash(work/name,spans):
            raise ValueError('Untargeted bytes changed: '+name)
        if file_hash(base/name)!=report['source_shard_hashes'][name]:raise ValueError('Source mutated')
        report['output_shard_hashes'][name]=file_hash(work/name)
        with (work/name).open('rb') as f:os.fsync(f.fileno())
        print('VERIFIED',name,flush=True)
    report.update(status='candidate_verified',finished=time.time(),changed_tensors=sorted(changed))
    save();shutil.copyfile(args.audit,work/'bf16-audit.json')
    work.rename(output)
    print('CANDIDATE VERIFIED',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['base','audit','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--stream-inputs',action='store_true',help='Consume completed, hash-verified tensor pairs while remaining pairs download')
    run(p.parse_args())
