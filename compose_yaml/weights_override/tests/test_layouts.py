import unittest,tempfile,json
from pathlib import Path
import numpy as np
from model_adapters.fused_experts import aligned_checkpoints
from weights_core.numeric import float_to_bf16
from weights_core.quantization import quantize_nvfp4
from weights_core.conversion import convert
from weights_core.planning import load_profile
from test_local import shard

class ExpertLayoutTests(unittest.TestCase):
 def test_fused_expert_conversion_both_architectures(self):
  for profile_name in ('ornith_15','gemma4_26b'):
   with self.subTest(profile=profile_name),tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);profile=load_profile(profile_name)
    source_prefix='model.language_model.layers.11.'+('mlp.' if profile_name=='ornith_15' else '')+'experts.'
    target_prefix=source_prefix if profile_name=='ornith_15' else source_prefix.replace('.experts.','.moe.experts.')
    gate=np.arange(128,dtype=np.float32).reshape(2,4,16)/64
    down=np.arange(64,dtype=np.float32).reshape(2,2,16)/64
    for name in ('original','donor','base'):
     folder=root/name;folder.mkdir()
     config={'architectures':[profile['architecture']],'text_config':profile['text_config']}
     if name=='base':
      tensors={};config['quantization_config']={'quant_method':'modelopt','quant_algo':'NVFP4'}
      for expert in range(2):
       for proj,weight in [('gate_proj',gate[expert,:2]),('up_proj',gate[expert,2:]),('down_proj',down[expert])]:
        q,s,g,_=quantize_nvfp4(weight);key=f'{target_prefix}{expert}.{proj}.weight'
        tensors[key]=('U8',[2,8],q.tobytes());tensors[key+'_scale']=('F8_E4M3',[2,1],s.tobytes());tensors[key+'_scale_2']=('F32',[],g.tobytes())
     else:
      changed=down.copy()
      if name=='donor':changed[1]*=.5
      tensors={source_prefix+'gate_up_proj':('BF16',[2,4,16],float_to_bf16(gate)),source_prefix+'down_proj':('BF16',[2,2,16],float_to_bf16(changed))}
     shard(folder/'model.safetensors',tensors);(folder/'config.json').write_text(json.dumps(config))
    report=convert(root/'original',root/'donor',root/'base',profile_name,root/'out',activation_scales='preserve')
    self.assertEqual(len(report['changes']),1)
    self.assertEqual(report['changes'][0]['name'],target_prefix+'1.down_proj.weight')
    self.assertEqual(report['changes'][0]['source_tensor'],source_prefix+'down_proj')
    self.assertEqual(report['preserved_base_tensors'],15)
