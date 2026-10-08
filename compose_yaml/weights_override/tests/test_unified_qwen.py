"""Opt-in end-to-end CLI test using tiny real GGUF/safetensors files, CPU Docker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_local import shard

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('WEIGHTS_TEST_DOCKER') == '1', 'requires local offline helper image and gguf')
class UnifiedQwenTests(unittest.TestCase):
    def test_check_convert_and_reject_overwrite(self):
        import gguf
        import numpy as np
        from weights_core.numeric import float_to_bf16, bf16_to_float
        from weights_core.safetensors_io import read_header
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, donor, base, output = (root/n for n in ('original','donor','base','output'))
            values = np.ones((2,32),dtype=np.float32)
            for directory, delta in [(original,0.),(donor,.125)]:
                (directory/'UD-Q4_K_XL').mkdir(parents=True)
                for i in range(1,5):
                    path = directory/f'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-{i:05d}-of-00004.gguf'
                    w = gguf.GGUFWriter(str(path),'qwen3next')
                    if i==1:
                        data=gguf.quantize(values+delta,gguf.GGMLQuantizationType.Q8_0)
                        w.add_tensor('blk.0.attn_output.weight',data,raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                    else:
                        w.add_tensor(f'unchanged_{i}',values)
                    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                path=directory/('mmproj-BF16.gguf' if directory==original else 'mmproj-model-bf16.gguf')
                w=gguf.GGUFWriter(str(path),'clip');w.add_tensor('vision',values)
                w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
            base.mkdir()
            profile=json.loads((ROOT/'profiles/qwen38_huihui.json').read_text())
            (base/'config.json').write_text(json.dumps({'architectures':[profile['architecture']], 'text_config':profile['text_config'],'quantization_config':{'quant_algo':'NVFP4'}}))
            name='model.language_model.layers.0.self_attn.o_proj.weight'
            shard(base/'model.safetensors',{name:('BF16',[2,32],float_to_bf16(values.reshape(-1)))})
            shard(base/'unchanged.safetensors',{'untouched':('BF16',[2,32],float_to_bf16(values.reshape(-1)))})
            (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:'model.safetensors','untouched':'unchanged.safetensors'}}))
            before={p.name:p.read_bytes() for p in base.iterdir()}
            command=[sys.executable,str(ROOT/'convert.py'),'--profile','qwen38_huihui','--original',str(original),'--donor',str(donor),'--base',str(base),'--output',str(output),'--activation-scales','preserve']
            checked=subprocess.run(command+['--check-only'],capture_output=True,text=True)
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
            self.assertFalse(output.exists())
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads((output/'conversion-manifest.json').read_text())
            self.assertFalse(report['runtime_validated'])
            self.assertEqual(report['verification']['status'],'passed')
            self.assertEqual(report['changed_count'],1)
            hd,offset=read_header(output/'model.safetensors')
            with (output/'model.safetensors').open('rb') as f:
                f.seek(offset+hd[name]['data_offsets'][0]);actual=bf16_to_float(f.read(128))
            expected=gguf.dequantize(gguf.quantize(values+.125,gguf.GGMLQuantizationType.Q8_0),gguf.GGMLQuantizationType.Q8_0)-gguf.dequantize(gguf.quantize(values,gguf.GGMLQuantizationType.Q8_0),gguf.GGMLQuantizationType.Q8_0)+values
            np.testing.assert_array_equal(actual,bf16_to_float(float_to_bf16(expected.reshape(-1))))
            self.assertEqual(before,{p.name:p.read_bytes() for p in base.iterdir()})
            self.assertEqual((base/'unchanged.safetensors').read_bytes(),(output/'unchanged.safetensors').read_bytes())
            self.assertFalse(os.path.samefile(base/'unchanged.safetensors',output/'unchanged.safetensors'))
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertIn('Output already exists',second.stderr)


class UnifiedDispatchTests(unittest.TestCase):
    def test_missing_qwen_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'output'
            result=subprocess.run([sys.executable,str(ROOT/'convert.py'),'--profile','qwen38_huihui','--cache',tmp,'--output',str(output),'--activation-scales','preserve'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Missing original GGUF',result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).iterdir()),[])
