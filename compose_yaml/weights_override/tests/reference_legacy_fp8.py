"""CPU regression: legacy Torch FP8 arithmetic vs unified CLI conversion core."""
import ast
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from weights_core.conversion import convert
from weights_core.numeric import float_to_bf16
from weights_core.safetensors_io import Checkpoint
from test_local import shard

root=Path(__file__).resolve().parents[1]
legacy=root/'quantization/fp8/quantize_fp8.py'
functions=[n for n in ast.parse(legacy.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('dequant_fp8','quantize_fp8')]
ns={'torch':torch}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(legacy),'exec'),ns)
quant,dequant=ns['quantize_fp8'],ns['dequant_fp8']
rng=np.random.default_rng(1096)
formats=[('per_tensor',[1],(16,32),'_scale_inv'),('per_tensor',[],(16,32),'_scale'),('per_channel',[16],(16,32),'.weight_scale'),('per_tensor',[1],(16,32),'_weight_scale'),('block',[2,2],(256,256),'_scale_inv'),('block',[2,1,2,1],(256,256),'_scale_inv'),('block',[2,3],(130,257),'_scale_inv')]
for mode,scale_shape,shape,suffix in formats:
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)
  original=torch.tensor(rng.normal(size=shape),dtype=torch.bfloat16).float()
  donor=(original+.02).to(torch.bfloat16).float()
  q,s=quant(original,mode,128,scale_shape);s=s.reshape(scale_shape)
  expected,expected_scale=quant(dequant(q,s,mode,128)+(donor-original),mode,128,scale_shape)
  name='layer.weight';scale_name=name+suffix
  for label in ('original','donor','base'):
   folder=p/label;folder.mkdir()
   config={'architectures':['Qwen3_5MoeForConditionalGeneration'],'text_config':{'hidden_size':256,'num_hidden_layers':1}}
   tensors={'unchanged':('BF16',[2],float_to_bf16(np.array([1,2],np.float32)))}
   if label=='base':
    config['quantization_config']={'quant_method':'fp8','weight_block_size':[128,128]}
    tensors[name]=('F8_E4M3',list(shape),q.view(torch.uint8).numpy().tobytes())
    tensors[scale_name]=('F32',scale_shape,s.numpy().tobytes())
   else:tensors[name]=('BF16',list(shape),float_to_bf16((original if label=='original' else donor).numpy()))
   shard(folder/'weights.safetensors',tensors);(folder/'config.json').write_text(json.dumps(config))
  before=[(p/n/'weights.safetensors').read_bytes() for n in ('original','donor','base')]
  report=convert(p/'original',p/'donor',p/'base','fp8',p/'output','preserve')
  cp=Checkpoint(p/'output')
  assert b''.join(cp.blocks(name))==expected.view(torch.uint8).numpy().tobytes(),(mode,scale_shape,'weight mismatch')
  assert b''.join(cp.blocks(scale_name))==expected_scale.reshape(scale_shape).numpy().tobytes(),(mode,scale_shape,'scale mismatch')
  assert before==[(p/n/'weights.safetensors').read_bytes() for n in ('original','donor','base')]
  assert b''.join(cp.blocks('unchanged'))==float_to_bf16(np.array([1,2],np.float32))
  print('PASS',mode,scale_shape,shape,suffix,flush=True)
print('All 7 legacy FP8 layouts match weights and scales byte-for-byte')
