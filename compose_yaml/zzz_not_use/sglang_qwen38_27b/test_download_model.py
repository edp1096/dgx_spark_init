import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

# Tests do not require a network, GPU, model download, or HF installation.
hub = types.ModuleType('huggingface_hub')
hub.HfApi = Mock()
hub.hf_hub_download = Mock()
hub.snapshot_download = Mock()
with patch.dict(sys.modules, huggingface_hub=hub):
    spec = importlib.util.spec_from_file_location('download_model', Path(__file__).with_name('download_model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

class DownloadTests(unittest.TestCase):
    def test_reuse_repair_and_complete_marker(self):
        payload = b'weight-data'
        files = [types.SimpleNamespace(rfilename='model.safetensors', size=len(payload), lfs=types.SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest())),
                 types.SimpleNamespace(rfilename='model.safetensors.index.json', lfs=None)]
        info = types.SimpleNamespace(sha='pinned', siblings=files)
        module.HfApi.return_value.model_info.return_value = info
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            weight = dest/'model.safetensors'
            weight.write_bytes(payload)
            def fetch(repo, name, **kw):
                if name.endswith('.index.json'):
                    (dest/name).write_text(json.dumps({'weight_map': {'a': 'model.safetensors'}}))
                else:
                    self.assertTrue(kw['force_download'])
                    weight.write_bytes(payload)
            module.hf_hub_download.reset_mock()
            module.hf_hub_download.side_effect = fetch
            module.download('repo', 'pinned', dest, None)
            self.assertEqual(module.hf_hub_download.call_count, 1)  # existing weight reused
            module.hf_hub_download.reset_mock()
            module.download('repo', 'pinned', dest, None)
            module.hf_hub_download.assert_not_called()  # completed run reused
            weight.write_bytes(b'broken-data')
            module.download('repo', 'pinned', dest, None)
            self.assertEqual(weight.read_bytes(), payload)  # corrupt shard repaired

    def test_revision_mismatch_rejected(self):
        module.HfApi.return_value.model_info.return_value = types.SimpleNamespace(sha='different')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                module.download('repo', 'pinned', Path(tmp), None)

if __name__ == '__main__':
    unittest.main()
