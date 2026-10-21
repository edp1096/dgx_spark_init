"""Check warmup tensor layout against per-head and full-vector normalization."""
import ast,importlib.util,sys,types
from pathlib import Path
import torch
p=Path(importlib.util.find_spec('vllm').origin).parent/'model_executor/warmup/qwen_triton_warmup.py'
node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_warm_layer_norm_kernel')
node.returns=None
for arg in node.args.args:arg.annotation=None
seen=[]
def check(x,weight,bias,eps,**kw):
 assert x.shape[-1]==weight.numel()
 assert kw['z'].shape==x.shape==kw['out'].shape
 assert x.shape[-1]%kw['group_size']==0
 seen.append(tuple(x.shape))
module=types.ModuleType('vllm.third_party.flash_linear_attention.ops.layernorm_guard');module.layer_norm_fwd=check
sys.modules[module.__name__]=module
ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
for width in (128,4096):
 seen.clear();c=types.SimpleNamespace(hv=32,v=128,norm_weight=torch.ones(width),norm_bias=None,norm_eps=1e-6,norm_group_size=128,conv_dtype=torch.bfloat16,norm_before_gate=True,norm_activation='swish')
 ns['_warm_layer_norm_kernel'](torch.device('cpu'),c)
 assert seen==[(n*(4096//width),width) for n in (1,2,16,32,128,1024)]
 print('norm width',width,'passed')
