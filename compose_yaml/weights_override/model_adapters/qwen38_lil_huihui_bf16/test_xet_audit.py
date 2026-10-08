import tempfile
import unittest
from pathlib import Path

from xet_audit import compare


class FakeRemote:
    def __init__(self, prefix=b'a', suffix=b'z', content_hash='a', offset=10):
        self.prefix, self.suffix, self.content_hash, self.offset = prefix, suffix, content_hash, offset
        self.reads = 0

    def file(self, name):
        return {'xet_hash': 'f'*64, 'tensors': {'weight': {'dtype': 'BF16', 'shape': [3], 'data_offsets': [0, 6]}}}

    def gap(self, info, key):
        s = self.offset
        return dict(start=s, end=s+6, gap_start=s+1, gap_end=s+5,
                    nodes=[{'hash': self.content_hash*64, 'size': 4}])

    def bytes(self, name, start, end):
        self.reads += 1
        if (start, end) == (self.offset, self.offset+1): return self.prefix
        if (start, end) == (self.offset+5, self.offset+6): return self.suffix
        raise AssertionError('Unexpected interior read')


class HashAuditTests(unittest.TestCase):
    def comparison(self, a, b):
        with tempfile.TemporaryDirectory() as td:
            return compare(Path(td), a, b, 'one.safetensors')['tensors'][0]

    def test_equal_payload_with_shifted_file_header(self):
        a, b = FakeRemote(offset=10), FakeRemote(offset=42)
        self.assertEqual(self.comparison(a, b)['status'], 'equal')
        self.assertEqual((a.reads, b.reads), (2, 2))

    def test_different_boundary_is_not_certified_equal(self):
        self.assertEqual(self.comparison(FakeRemote(), FakeRemote(suffix=b'x'))['status'], 'requires_tensor_bytes')

    def test_different_interior_is_not_downloaded_or_certified_equal(self):
        a, b = FakeRemote(), FakeRemote(content_hash='b')
        self.assertEqual(self.comparison(a, b)['status'], 'requires_tensor_bytes')
        self.assertEqual((a.reads, b.reads), (0, 0))

    def test_equal_hashes_at_different_relative_positions_do_not_certify(self):
        a, b = FakeRemote(), FakeRemote()
        b.gap = lambda info, key: dict(start=10, end=16, gap_start=12, gap_end=16,
                                     nodes=[{'hash': 'a'*64, 'size': 4}])
        self.assertEqual(self.comparison(a, b)['status'], 'requires_tensor_bytes')


if __name__ == '__main__':
    unittest.main()
