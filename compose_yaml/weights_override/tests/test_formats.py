import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from weights_core.formats import validate_format

ROOT=Path(__file__).resolve().parents[1]

class FormatTests(unittest.TestCase):
    def test_auto_and_matching(self):
        for quant,expected in [({'quant_method':'fp8'},'fp8'),({'quant_method':'modelopt','quant_algo':'NVFP4'},'nvfp4'),({'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','quantized_layers':{'expert':{'quant_algo':'W4A16_NVFP4'},'attention':{'quant_algo':'FP8'}}},'nvfp4')]:
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);(p/'config.json').write_text(json.dumps({'quantization_config':quant}))
                self.assertEqual(validate_format(p),expected)
                self.assertEqual(validate_format(p,expected),expected)
                with self.assertRaises(ValueError):validate_format(p,'fp8' if expected=='nvfp4' else 'nvfp4')

    def test_cli_mismatch_before_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir()
            (base/'config.json').write_text('{"quantization_config":{"quant_method":"fp8"}}')
            result=subprocess.run([sys.executable,str(ROOT/'convert.py'),'--profile','fp8','--original',str(root/'missing1'),'--donor',str(root/'missing2'),'--base',str(base),'--format','nvfp4','--output',str(root/'output')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('does not match base format fp8',result.stderr)
            self.assertFalse((root/'output').exists())
