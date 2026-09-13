"""Ensure late profile loading survives warmups and restores hooks on failure."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autotune_window as window


class AutotuneWindowTests(unittest.TestCase):
    def test_late_profile_and_exception_cleanup(self):
        for limit in ('0', '4096'):
            original = lambda runner: (8192, 32)
            module = SimpleNamespace(_flashinfer_autotune_token_counts=original)
            events = []
            with patch.dict(os.environ, {'DSV41_AUTOTUNE_TOKENS': limit}), \
                 patch.dict(sys.modules, {'vllm.model_executor.warmup.kernel_warmup': module}), \
                 patch.object(window, 'load_dense_profile', side_effect=lambda: events.append('profile')):
                with window.bounded_autotune():
                    self.assertEqual(module._flashinfer_autotune_token_counts(None),
                                     (4096,32) if limit == '4096' else (8192,32))
                    self.assertEqual(events, [])
                    events.append('warmup')
                self.assertEqual(events, ['warmup','profile'])
                self.assertIs(module._flashinfer_autotune_token_counts, original)
                events.clear()
                with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
                    with window.bounded_autotune():
                        raise RuntimeError('warmup failed')
                self.assertEqual(events, [])
                self.assertIs(module._flashinfer_autotune_token_counts, original)


if __name__ == '__main__':
    unittest.main()
