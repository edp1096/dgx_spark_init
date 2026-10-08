import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from weights_core.safetensors_io import Checkpoint, read_header, replace_shard
from weights_core.planning import audit, load_profile


def shard(path, values):
    offset = 0
    header = {}
    payload = b''
    for name, (dtype, shape, data) in values.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + len(data)])
        offset += len(data)
        payload += data
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, 'little') + raw + payload)


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkpoint(self, name, values, profile='ornith_15'):
        path = self.root / name
        path.mkdir()
        shard(path / 'model.safetensors', values)
        p = load_profile(profile)
        (path / 'config.json').write_text(json.dumps({'architectures': [p['architecture']], 'text_config': p['text_config']}))
        return path

    def test_plan_changed_and_preserved_both_profiles(self):
        for profile in ('ornith_15', 'gemma4_26b'):
            with self.subTest(profile=profile):
                values = {'a': ('BF16', [1], b'\x80\x3f'), 'b': ('BF16', [1], b'\x00\x40')}
                original = self.checkpoint(profile+'o', values, profile)
                base = self.checkpoint(profile+'b', values, profile)
                donor = self.checkpoint(profile+'d', {**values, 'a': ('BF16', [1], b'\x40\x40')}, profile)
                # Fail if code attempts network activity during offline audit.
                with patch('socket.socket', side_effect=AssertionError('Network prohibited')):
                    report = audit(original, donor, base, profile)
                self.assertEqual(report['unchanged_source_tensors'], 1)
                self.assertEqual(report['changes'][0]['action'], 'copy_donor_bf16')
                self.assertFalse(report['conversion_ready'])

    def test_packed_base_requires_quantizer(self):
        o = self.checkpoint('o', {'a': ('BF16', [2], b'\0\0\0\0')})
        d = self.checkpoint('d', {'a': ('BF16', [2], b'\0\0\x80\x3f')})
        b = self.checkpoint('b', {'a': ('U8', [1], b'\0')})
        self.assertEqual(audit(o, d, b, 'ornith_15')['changes'][0]['action'], 'requires_quantizer')

    def test_base_mismatch_blocks_direct_copy(self):
        o = self.checkpoint('o', {'a': ('BF16', [1], b'\0\0')})
        d = self.checkpoint('d', {'a': ('BF16', [1], b'\x80\x3f')})
        b = self.checkpoint('b', {'a': ('BF16', [1], b'\0\x40')})
        self.assertEqual(audit(o, d, b, 'ornith_15')['changes'][0]['action'], 'blocked')

    def test_missing_keys_rejected(self):
        o = self.checkpoint('o', {'a': ('BF16', [1], b'\0\0')})
        d = self.checkpoint('d', {'b': ('BF16', [1], b'\0\0')})
        with self.assertRaises(ValueError):
            audit(o, d, o, 'ornith_15')

    def test_copy_does_not_mutate_source_or_unselected_tensor(self):
        source = self.root/'source.safetensors'
        target = self.root/'target.safetensors'
        shard(source, {'a': ('BF16', [1], b'\0\0'), 'b': ('BF16', [1], b'\0\x40')})
        before = source.read_bytes()
        raw = self.root/'replacement'
        raw.write_bytes(b'\x80\x3f')
        replace_shard(source, target, {'a': (raw, 0, 2)})
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(target.read_bytes()[-4:], b'\x80\x3f\0\x40')
        with self.assertRaises(FileExistsError):
            replace_shard(source, source, {'a': (raw, 0, 2)})
        self.assertEqual(source.read_bytes(), before)

    def test_bad_replacement_leaves_no_output(self):
        src = self.root/'source.safetensors'
        dst = self.root/'target.safetensors'
        shard(src, {'a': ('BF16', [1], b'\0\0')})
        with self.assertRaises(ValueError):
            replace_shard(src, dst, {'a': (src, 0, 1)})
        self.assertFalse(dst.exists())

    def test_truncated_payload_rejected(self):
        src = self.root/'source.safetensors'
        shard(src, {'a': ('BF16', [1], b'\0\0')})
        src.write_bytes(src.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            read_header(src)

    def test_chunked_copy_and_failure_cleanup(self):
        src = self.root/'source.safetensors'
        dst = self.root/'target.safetensors'
        raw = self.root/'raw'
        raw.write_bytes(bytes(range(32)))
        shard(src, {'a': ('U8', [32], bytes(32))})
        with patch('weights_core.safetensors_io.CHUNK', 3):
            replace_shard(src, dst, {'a': (raw, 0, 32)})
        self.assertEqual(dst.read_bytes()[-32:], raw.read_bytes())
        dst.unlink()
        with patch('weights_core.safetensors_io.chunks', side_effect=OSError('read failed')):
            with self.assertRaises(OSError):
                replace_shard(src, dst, {'a': (raw, 0, 32)})
        self.assertFalse(dst.exists())

    def test_wrong_profile_rejected(self):
        o = self.checkpoint('o', {'a': ('BF16', [1], b'\0\0')})
        with self.assertRaises(ValueError):
            audit(o, o, o, 'gemma4_26b')

    def test_bad_index_rejected(self):
        p = self.checkpoint('x', {'a': ('BF16', [1], b'\0\0')})
        (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'b': 'model.safetensors'}}))
        with self.assertRaises(ValueError):
            Checkpoint(p)

    def test_index_traversal_rejected(self):
        p = self.root/'bad'
        p.mkdir()
        (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../bad.safetensors'}}))
        with self.assertRaises(ValueError):
            Checkpoint(p)


if __name__ == '__main__':
    unittest.main()
