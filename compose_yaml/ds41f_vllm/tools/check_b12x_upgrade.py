"""Compare packed ABI and real expert outputs across pinned b12x images."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,hashlib,json,time
from pathlib import Path
import torch
from b12x_layout import from_raw,tensors
import b12x_slots
from expert_store import ExpertStore
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model');p.add_argument('--rank',type=int,required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--write-reference',action='store_true');p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.manual_seed(918)
store=ExpertStore(a.model,rank=a.rank);rows=[];outputs={}
expected={} if a.write_reference else torch.load(a.reference,weights_only=True)
with torch.inference_mode():
    for layer,topk in ((0,6),(40,3)):
        cache=b12x_slots.SlotLayer(store,layer,topk);b12x_slots._layers[layer]=cache
        # Independently repack source weights and compare every byte to disk.
        for expert in (0,17):
            prepared=from_raw(store.expert(layer if layer<40 else layer-40,expert,'layers' if layer<40 else 'mtp'))
            digest=hashlib.blake2b(digest_size=16)
            for t,desc in zip(tensors(prepared),cache.meta['layout']):
                assert list(t.shape[1:])==desc['shape'] and str(t.dtype)==desc['dtype']
                digest.update(memoryview(t[0].cpu().contiguous().view(torch.uint8).numpy()))
            assert digest.hexdigest()==cache.meta['hashes'][expert],('PACKED_ABI_CHANGED',layer,expert)
        for n in (1,5,6,16,512,2048):
            x=(torch.randn(n,5120,dtype=torch.bfloat16)*.1).cuda()
            weights=torch.rand(n,topk).cuda();weights/=weights.sum(-1,keepdim=True)
            ids=((torch.arange(n,dtype=torch.int32)[:,None]+torch.arange(topk,dtype=torch.int32))%24).cuda()
            mapped=cache.ensure(ids.flatten().tolist())
            output=cache.execute(x,weights,mapped,ids.shape).clone()
            key=f'{layer}:{n}';outputs[key]=output.cpu()
            for _ in range(3): cache.execute(x,weights,mapped,ids.shape)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(30): cache.execute(x,weights,mapped,ids.shape)
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/30
            row={'layer':layer,'tokens':n,'execution_ms':ms,'finite':bool(output.isfinite().all())}
            assert row['finite']
            if not a.write_reference:
                target=expected[key].cuda()
                row['max_abs_error']=(output-target).abs().max().item()
                row['relative_l2_error']=((output-target).float().norm()/target.float().norm()).item()
                try:
                    torch.testing.assert_close(output,target,rtol=1e-5,atol=1e-5);row["numeric_close"]=True
                except AssertionError: row["numeric_close"]=False
            rows.append(row);print(json.dumps(row),flush=True)
    if a.write_reference: torch.save(outputs,a.reference)
    a.output.write_text(json.dumps({'rank':a.rank,'write_reference':a.write_reference,'rows':rows,'packed_abi_identical':True},indent=2)+'\n')
    assert all(r.get('numeric_close',True) for r in rows),'NUMERICAL_EQUIVALENCE_FAILED'
    print('B12X_UPGRADE_VALIDATION_PASS',flush=True)
