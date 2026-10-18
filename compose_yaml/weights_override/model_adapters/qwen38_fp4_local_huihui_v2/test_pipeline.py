import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch
from modelopt.torch.quantization.qtensor.mxfp8_tensor import MXFP8QTensor

from audit import parse_header, tensor_bytes, complements
from build import q8_decode, restore_gdn, transform, mapped, complement_hash


class PipelineTests(unittest.TestCase):
    def test_gguf_header_and_shape(self):
        name = b'blk.0.ssm_out.weight'
        raw = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
        raw += struct.pack('<Q', len(name)) + name
        raw += struct.pack('<IQQIQ', 2, 32, 2, 8, 0)
        h = parse_header(raw)
        self.assertEqual(h['tensors'][name.decode()]['bytes'], 68)
        self.assertEqual(h['start'] % 32, 0)
        with self.assertRaises(ValueError):
            tensor_bytes([31, 2], 8)

    def test_q8_block_decode(self):
        values = list(range(-16, 16))
        raw = struct.pack('<e32b', .5, *values) + struct.pack('<e32b', 2., *values)
        expected = torch.tensor([values, values], dtype=torch.float32) * torch.tensor([[.5], [2.]])
        self.assertTrue(torch.equal(q8_decode(raw, 2, 32), expected))
        with self.assertRaises(ValueError):
            q8_decode(raw[:-1], 2, 32)

    def test_inverse_gdn_column_layout(self):
        config = {'text_config': {'linear_num_key_heads': 2, 'linear_num_value_heads': 4, 'linear_value_head_dim': 16}}
        native = torch.arange(128).reshape(2, 64)
        gguf = native.reshape(2, 2, 2, 16).transpose(1, 2).contiguous().reshape(2, 64)
        self.assertFalse(torch.equal(native, gguf))
        self.assertTrue(torch.equal(native, restore_gdn(gguf, config)))

    def test_zero_delta_and_nonzero_change(self):
        original = torch.linspace(-1, 1, 128).reshape(4, 32)
        q, scales = MXFP8QTensor.quantize(original.to(torch.bfloat16))
        before = q.dequantize(dtype=torch.float32, scale=scales)
        same, ss, stats = transform(original, original, q._quantized_data, scales, {}, False)
        decoded = (same.float().reshape(4, 1, 32) * torch.exp2(ss.float() - 127)[..., None]).reshape(4, 32)
        self.assertTrue(torch.equal(before, decoded))
        self.assertEqual(stats['delta_norm2'], 0.)
        changed, cs, metrics = transform(original, original + .1, q._quantized_data, scales, {}, False)
        after = (changed.float().reshape(4, 1, 32) * torch.exp2(cs.float() - 127)[..., None]).reshape(4, 32)
        self.assertGreater(metrics['delta_dot_effect'], 0.)
        self.assertLess((after - before - .1).abs().max().item(), .065)

    def test_experts_are_not_mutation_targets(self):
        row = {'name': 'blk.0.ffn_down_exps.weight', 'type': 8, 'shape': [640, 2560, 512], 'bytes': 1}
        with self.assertRaises(ValueError):
            mapped(row)

    def test_untouched_csf_bytes_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            a.write_bytes(b'CSF_before' + b'old-weight' + b'CSF_after')
            b.write_bytes(b'CSF_before' + b'new-weight' + b'CSF_after')
            allowed = [(10, 20)]
            self.assertEqual(complement_hash(a, allowed), complement_hash(b, allowed))
            b.write_bytes(b'CSF_beforX' + b'new-weight' + b'CSF_after')
            self.assertNotEqual(complement_hash(a, allowed), complement_hash(b, allowed))
        self.assertEqual(complements(100, [(10, 20), (20, 80)]), [(0, 10), (80, 100)])
        with self.assertRaises(ValueError):
            complements(100, [(10, 30), (20, 80)])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
