import json,pathlib,sys
import torch
from safetensors import safe_open
from b12x.gemm.bf16_gemv import mm
p=pathlib.Path(sys.argv[1])
idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
with safe_open(p/idx['lm_head.weight'],framework='pt',device='cpu') as f:
    weight=f.get_tensor('lm_head.weight').cuda()
assert weight.shape==(248320,2560) and weight.dtype==torch.bfloat16
torch.manual_seed(19)
for m in (1,4):
    x=torch.randn((m,2560),device='cuda',dtype=torch.bfloat16)
    expected=x @ weight.T
    actual=mm(x,weight,output_dtype=torch.bfloat16)
    relative=((actual.float()-expected.float()).norm()/expected.float().norm()).item()
    assert relative<0.005 and torch.isfinite(actual).all(),relative
    print(f'Actual TP1 LM head M={m}: B12X/BF16 matmul relative L2={relative}',flush=True)
