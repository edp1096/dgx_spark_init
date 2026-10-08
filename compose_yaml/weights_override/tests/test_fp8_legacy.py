import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import numpy as np
from weights_core.conversion import convert, fp8_block_layout
from weights_core.numeric import float_to_bf16
from weights_core.quantization import quantize_fp8, fp8_decode
from weights_core.safetensors_io import Checkpoint
from test_local import shard


class FP8LegacyTests(unittest.TestCase):
    def test_fused_experts_and_raw_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);prefix='model.layers.0.mlp.experts.'
            source=np.ones((2,2,16),np.float32)
            donor=source.copy();donor[1]*=.5
            q,scale,_=quantize_fp8(source[0],[1])
            for label in ('original','donor','base'):
                folder=p/label;folder.mkdir()
                config={'architectures':['Qwen3_5MoeForConditionalGeneration'],'text_config':{'hidden_size':16,'num_hidden_layers':1}}
                tensors={'norm.weight':('F32',[2],np.array([2,2] if label=='donor' else [1,1],dtype='<f4').tobytes())}
                if label=='base':
                    config['quantization_config']={'quant_method':'fp8'}
                    for i in range(2):
                        name=prefix+str(i)+'.down_proj.weight'
                        tensors[name]=('F8_E4M3',[2,16],q.tobytes())
                        tensors[name+'_scale_inv']=('F32',[1],scale.tobytes())
                else:tensors[prefix+'down_proj']=('BF16',[2,2,16],float_to_bf16(donor if label=='donor' else source))
                shard(folder/'weights.safetensors',tensors);(folder/'config.json').write_text(json.dumps(config))
            run=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'convert.py'),'--profile','fp8','--original',str(p/'original'),'--donor',str(p/'donor'),'--base',str(p/'base'),'--output',str(p/'out'),'--activation-scales','preserve'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            report=json.loads((p/'out/conversion-manifest.json').read_text())
            cp=Checkpoint(p/'out');base=Checkpoint(p/'base')
            self.assertEqual(b''.join(cp.blocks('norm.weight')),np.array([2,2],dtype='<f4').tobytes())
            self.assertEqual(cp.digest(prefix+'0.down_proj.weight'),base.digest(prefix+'0.down_proj.weight'))
            self.assertEqual(len(report['changes']),2)
            self.assertEqual(report['preserved_base_tensors'],2)

    def test_partial_blocks_require_metadata(self):
        with self.assertRaises(ValueError):fp8_block_layout([130,257],[2,3],{})
        self.assertEqual(fp8_block_layout([130,257],[2,3],{'weight_block_size':[128,128]}),[128,128])
        with self.assertRaises(ValueError):fp8_block_layout([130,257],[2,1,3,2],{'weight_block_size':[128,128]})
