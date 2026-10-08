
# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
import torch
from safetensors import safe_open
from vllm.models.deepseek_v4_1.common.engram import DiskEngramTable
from pathlib import Path
import json
model=Path(sys.argv[1])
mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
for layer, start in [(1,0),(1,192000000),(14,0),(14,192000000)]:
    table=DiskEngramTable(str(model),layer,256,32,row_start=start,num_rows=100)
    indices=torch.tensor([0,31,99],dtype=torch.int64,device='cpu')
    actual=table.gather_dequant(indices,torch.tensor([True,True,True],device='cpu'))
    expected=[]
    for i in indices.tolist():
        pair=[]
        for kind in ('weight','scale'):
            name=f'layers.{layer}.engram.embed.{kind}'
            with safe_open(str(model/mapping[name]),framework='pt',device='cpu') as f:
                t=f.get_slice(name)[start+i:start+i+1]
                pair.append(t)
        w,s=pair
        sf=torch.exp2(s.view(torch.uint8).float()-127)
        expected.append((w.float().reshape(1,8,32)*sf.unsqueeze(-1)).flatten(1).to(torch.bfloat16))
    assert torch.equal(actual,torch.cat(expected)),(layer,start)
print('PASS: Engram disk lookup at rank-zero and nonzero row offsets, both tables')
