"""Exercise the real native-BF16 delta path on a bounded set of downloaded weights."""
import argparse
import json
import math
from pathlib import Path
import torch
from weights_core.safetensors_io import Checkpoint
from quantize import decode,encode,read
from fetch_audit import atomic_json


def source(path,shape,offset=0):
    with path.open('rb') as f:
        f.seek(offset);raw=f.read(math.prod(shape)*2)
    if len(raw)!=math.prod(shape)*2:raise ValueError('Missing BF16 matrix')
    return torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(shape).float()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);cp=Checkpoint(a.base);results=[]
    cases=[]
    for layer in [0,1]:
        name=f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        for expert in [0,1,127,511]:
            cases.append((name,f'model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj',[2560,640],expert*2560*640*2))
    name='model.language_model.layers.0.linear_attn.out_proj.weight'
    cases.append((name,name.removesuffix('.weight'),[2560,6144],0))
    for name,prefix,shape,offset in cases:
        original=source(a.audit_root/'original'/(name+'.bf16'),shape,offset)
        donor=source(a.audit_root/'huihui'/(name+'.bf16'),shape,offset)
        base,kind=decode(cp,prefix);delta=donor-original;target=base+delta
        fixed=(read(cp,prefix+'.weight_scale'),read(cp,prefix+'.weight_scale_2')) if kind=='nvfp4' else None
        _,restored,choice=encode(target,kind,fixed)
        effect=restored-base
        values={'target_relative_error':((target-restored).norm()/target.norm()).item(),
                'delta_relative_l2':(delta.norm()/base.norm()).item(),
                'delta_effective_cosine':(delta.double().flatten().dot(effect.double().flatten())/(delta.double().norm()*effect.double().norm())).item(),
                'scale_choice':choice}
        if not all(math.isfinite(v) for v in values.values() if isinstance(v,float)) or values['target_relative_error']>.2:
            raise ValueError('Real numerical probe failed')
        results.append(dict(tensor=prefix,kind=kind,**values));print(prefix,values,flush=True)
    atomic_json(a.output,{'status':'passed','scope':'numerical reconstruction only; not an inference-quality benchmark','results':results})
