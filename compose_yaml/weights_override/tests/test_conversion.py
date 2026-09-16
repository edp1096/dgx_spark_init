import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from weights_core.conversion import convert
from weights_core.numeric import float_to_bf16
from weights_core.quantization import fp8_encode, fp8_decode, quantize_fp8, quantize_nvfp4
from weights_core.safetensors_io import Checkpoint
from test_local import shard
from weights_core.planning import load_profile


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.values = np.linspace(-3, 3, 64, dtype=np.float32).reshape(4,16)

    def setup_models(self, kind='nvfp4'):
        values = self.values
        for role in ('original','donor','base'):
            p = self.root/role
            p.mkdir()
            profile = load_profile('ornith_15')
            config = {'architectures':[profile['architecture']], 'text_config':profile['text_config']}
            tensors = {'unchanged': ('BF16', [1], float_to_bf16(np.array([1],dtype=np.float32)))}
            weight = values if role != 'donor' else values*.8
            # Construct the base from BF16 original, exactly as the public contract.
            from weights_core.numeric import bf16_to_float
            weight = bf16_to_float(float_to_bf16(weight)).reshape(4,16)
            if role != 'base' or kind == 'bf16':
                tensors['linear.weight'] = ('BF16', [4,16], float_to_bf16(weight))
            else:
                config['quantization_config'] = {'quant_method':'modelopt', 'quant_algo': 'NVFP4' if kind=='nvfp4' else 'FP8'}
                if kind == 'nvfp4':
                    q,s,g,_ = quantize_nvfp4(weight)
                    tensors.update({'linear.weight':('U8',[4,8],q.tobytes()),
                                    'linear.weight_scale':('F8_E4M3',[4,1],s.tobytes()),
                                    'linear.weight_scale_2':('F32',[],g.tobytes())})
                else:
                    q,s,_ = quantize_fp8(weight, [1])
                    tensors.update({'linear.weight':('F8_E4M3',[4,16],q.tobytes()),
                                    'linear.weight_scale':('F32',[1],s.tobytes())})
                tensors['linear.input_scale'] = ('F32',[],np.float32(.5).tobytes())
            shard(p/'model.safetensors', tensors)
            (p/'config.json').write_text(json.dumps(config))
        return [self.root/n for n in ('original','donor','base')]

    def test_nvfp4_pipeline_preserves_inputs_and_unmodified_tensors(self):
        inputs = self.setup_models()
        before = [(p/'model.safetensors').read_bytes() for p in inputs]
        out = self.root/'result'
        with patch('socket.socket', side_effect=AssertionError('Network prohibited')):
            report = convert(*inputs, 'ornith_15', out, activation_scales='preserve')
        self.assertEqual(report['status'],'candidate_verified')
        self.assertFalse(report['runtime_validated'])
        self.assertEqual(before, [(p/'model.safetensors').read_bytes() for p in inputs])
        src,dst = Checkpoint(inputs[2]),Checkpoint(out)
        for key in ('unchanged','linear.input_scale'):
            self.assertEqual(src.digest(key),dst.digest(key))
        self.assertNotEqual(src.digest('linear.weight_scale_2'),dst.digest('linear.weight_scale_2'))
        self.assertFalse((out/'.payloads').exists())

    def test_scales_in_separate_shard(self):
        inputs = self.setup_models()
        base = Checkpoint(inputs[2])
        groups = [{}, {}]
        index = {}
        for key, tensor in base.tensors.items():
            group = 1 if 'scale' in key else 0
            groups[group][key] = (tensor['dtype'], tensor['shape'], b''.join(base.blocks(key)))
            index[key] = f'part-{group}.safetensors'
        (inputs[2]/'model.safetensors').unlink()
        for number, tensors in enumerate(groups):
            shard(inputs[2]/f'part-{number}.safetensors',tensors)
        (inputs[2]/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
        out = self.root/'out'
        report = convert(*inputs,'ornith_15',out,activation_scales='preserve')
        self.assertEqual(Checkpoint(out).tensors.keys(),base.tensors.keys())
        self.assertEqual(report['status'],'candidate_verified')
        import os
        self.assertFalse(os.path.samefile(inputs[2]/'part-0.safetensors',out/'part-0.safetensors'))

    def test_dry_run_does_not_write(self):
        inputs = self.setup_models()
        cmd = [sys.executable,str(Path(__file__).resolve().parents[1]/'convert.py'),
               '--profile','ornith_15','--output',str(self.root/'out'),
               '--activation-scales','preserve','--check-only']
        for flag,path in zip(('original','donor','base'),inputs):
            cmd += ['--'+flag,str(path)]
        result = subprocess.run(cmd,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.root/'out').exists())

    def test_fp8_pipeline(self):
        inputs = self.setup_models('fp8')
        report = convert(*inputs,'ornith_15',self.root/'out',activation_scales='preserve')
        self.assertLess(report['metrics'][0]['relative_l2_error'], .05)

    def test_bf16_pipeline(self):
        inputs = self.setup_models('bf16')
        report = convert(*inputs,'ornith_15',self.root/'out')
        self.assertEqual(report['metrics'][0]['relative_l2_error'],0)
        self.assertEqual(Checkpoint(inputs[1]).digest('linear.weight'),Checkpoint(self.root/'out').digest('linear.weight'))

    def test_implicit_activation_policy_rejected_before_output(self):
        inputs = self.setup_models()
        with self.assertRaisesRegex(ValueError,'activation-scales'):
            convert(*inputs,'ornith_15',self.root/'out')
        self.assertFalse((self.root/'out').exists())
        self.assertFalse(list(self.root.glob('.out.*')))

    def test_error_threshold_cleans_partial_output(self):
        inputs = self.setup_models()
        with self.assertRaisesRegex(ValueError,'align with original|error too large'):
            convert(*inputs,'ornith_15',self.root/'out',activation_scales='preserve',max_relative_error=1e-10)
        self.assertFalse((self.root/'out').exists())
        self.assertFalse(list(self.root.glob('.out.*')))

    def test_wrong_quantized_base_rejected(self):
        inputs = self.setup_models()
        # Base produced from the negative of original should never be accepted.
        q,s,g,_ = quantize_nvfp4(-self.values)
        shard(inputs[2]/'model.safetensors', {
            'linear.weight':('U8',[4,8],q.tobytes()),
            'linear.weight_scale':('F8_E4M3',[4,1],s.tobytes()),
            'linear.weight_scale_2':('F32',[],g.tobytes()),
            'unchanged':('BF16',[1],b'\x80\x3f')})
        with self.assertRaisesRegex(ValueError,'align with original'):
            convert(*inputs,'ornith_15',self.root/'out',activation_scales='preserve')
        self.assertFalse((self.root/'out').exists())

    def test_external_profile(self):
        inputs = self.setup_models('bf16')
        path = self.root/'profile.json'
        path.write_text(json.dumps(load_profile('ornith_15')))
        self.assertEqual(convert(*inputs,str(path),self.root/'out')['status'],'candidate_verified')

    def test_existing_output_and_nested_output_rejected(self):
        inputs = self.setup_models('bf16')
        with self.assertRaises(ValueError):
            convert(*inputs,'ornith_15',inputs[2]/'out')
        out = self.root/'out'
        out.mkdir()
        (out/'keep').write_text('safe')
        with self.assertRaises(FileExistsError):
            convert(*inputs,'ornith_15',out)
        self.assertEqual((out/'keep').read_text(),'safe')

    def test_cli_end_to_end_and_repeat_refusal(self):
        inputs = self.setup_models('bf16')
        cmd = [sys.executable,str(Path(__file__).resolve().parents[1]/'convert.py'),
               '--profile','ornith_15','--output',str(self.root/'out')]
        for flag,path in zip(('original','donor','base'),inputs):
            cmd += ['--'+flag,str(path)]
        self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)
        self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


class QuantizerTests(unittest.TestCase):
    def test_fp8_roundtrip_all_finite_codes(self):
        codes = np.array([i for i in range(256) if (i&127)!=127],dtype=np.uint8)
        np.testing.assert_array_equal(fp8_encode(fp8_decode(codes)),codes)

    def test_fp8_ties_to_even_and_saturation(self):
        np.testing.assert_array_equal(fp8_encode(np.array([1.0625,1.1875,500,-500])),[56,58,126,254])

    def test_fp8_channel_scales(self):
        values = np.array([[1,2,-3],[100,200,-300]],dtype=np.float32)
        for shape in ([2],[2,1]):
            codes, scale, restored = quantize_fp8(values,shape)
            self.assertEqual(list(scale.shape),shape)
            np.testing.assert_allclose(restored,values,rtol=.04)
        with self.assertRaises(ValueError):
            quantize_fp8(values,[2,2])

    def test_nvfp4_zeros_finite(self):
        q,s,g,r = quantize_nvfp4(np.zeros((2,16),dtype=np.float32))
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue((q==0).all())
        self.assertGreater(float(g),0)

    def test_nonfinite_rejected(self):
        for quantizer in (lambda x: quantize_fp8(x,[1]), quantize_nvfp4):
            with self.assertRaises(ValueError):
                quantizer(np.full((1,16),np.nan,dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
