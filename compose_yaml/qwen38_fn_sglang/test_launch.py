import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from launch import server_arguments


class LaunchTests(unittest.TestCase):
    def test_default_and_explicit_map(self):
        args = ['--model-path', '/missing']
        self.assertEqual(server_arguments(args, 'off'), args)
        for extra in [['--speculative-token-map', '/custom.pt'], ['--speculative-token-map=/custom.pt']]:
            self.assertEqual(server_arguments(args + extra, 'ko64k'), args + extra)
        with self.assertRaises(ValueError):
            server_arguments(args, 'typo')

    def test_fingerprint_and_missing_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tokenizer.json').write_bytes(b'known tokenizer')
            (root / 'draft_vocab.json').write_text(json.dumps({'tokenizer_sha256': hashlib.sha256(b'known tokenizer').hexdigest()}))
            args = ['--model-path=' + tmp]
            with self.assertRaisesRegex(ValueError, 'missing'):
                server_arguments(args, 'ko64k', root)
            (root / 'draft_vocab.pt').touch()
            self.assertEqual(server_arguments(args, 'ko64k', root)[-2:], ['--speculative-token-map', str(root / 'draft_vocab.pt')])
            (root / 'tokenizer.json').write_bytes(b'different tokenizer')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                server_arguments(args, 'ko64k', root)


if __name__ == '__main__':
    unittest.main()
