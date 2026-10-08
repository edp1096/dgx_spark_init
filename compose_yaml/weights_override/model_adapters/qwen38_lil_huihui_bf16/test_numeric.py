"""Independent ModelOpt decoder and immutable-output checks (CPU only)."""
import hashlib
from pathlib import Path
import tempfile
import torch
from safetensors.torch import save_file
from weights_core.safetensors_io import Checkpoint
from quantize import decode, encode
from build import complement_hash, matrices


def main():
    torch.set_num_threads(2);torch.manual_seed(5500)
    for kind in ['mxfp8','nvfp4']:
        for shape in [(3,64),(64,128)]:
            x=torch.randn(shape)*.03
            parts,reference,_=encode(x,kind)
            with tempfile.TemporaryDirectory() as td:
                save_file({'p.'+k:v for k,v in parts.items()},str(Path(td)/'model.safetensors'))
                actual,detected=decode(Checkpoint(td),'p')
                if detected!=kind:raise AssertionError('Wrong quantization detection')
                torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            if kind=='nvfp4':
                target=x+torch.randn_like(x)*.001
                _,fresh,_=encode(target,kind)
                _,best,_=encode(target,kind,(parts['weight_scale'],parts['weight_scale_2']))
                assert (best-target).double().square().sum() <= (fresh-target).double().square().sum()
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'data';p.write_bytes(b'0123456789')
        assert complement_hash(p,[(2,4),(6,9)])==hashlib.sha256(b'01459').hexdigest()
        try:complement_hash(p,[(1,5),(4,7)])
        except ValueError:pass
        else:raise AssertionError('Overlapping writes accepted')
    row={'name':'model.language_model.layers.0.mlp.experts.down_proj','dtype':'BF16','shape':[512,2560,640]}
    items=matrices(row)
    assert len(items)==512 and items[0][0].endswith('experts.0.down_proj') and items[-1][0].endswith('experts.511.down_proj')
    row['name']='model.language_model.layers.0.mlp.experts.gate_up_proj'
    try:matrices(row)
    except ValueError:pass
    else:raise AssertionError('Unaudited projection accepted')
    print('PASS: ModelOpt decoder equivalence, scale error bound, immutable regions, native fused-expert mapping')


if __name__=='__main__':main()
