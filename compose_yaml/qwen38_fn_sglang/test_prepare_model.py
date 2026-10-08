import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('standalone_prepare', Path(__file__).with_name('prepare_model.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def checkpoint(root, label='new'):
    root.mkdir(parents=True, exist_ok=True)
    (root/'config.json').write_text(json.dumps({'label': label}))
    header=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'1234')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'model.safetensors'}}))


class StandalonePreparationTests(unittest.TestCase):
    def test_empty_download_then_offline_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'model';calls=[]
            def download(**kw):
                calls.append(kw);checkpoint(Path(kw['local_dir']))
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=download)}):
                self.assertTrue(mod.prepare('test/model','a'*40,root,Path(d)/'hub'))
                self.assertTrue(mod.prepare('test/model','a'*40,root,Path(d)/'hub',True))
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0]['revision'],'a'*40)
    def test_structurally_complete_old_revision_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint(root)
            identity={'config.json':hashlib.sha256((root/'config.json').read_bytes()).hexdigest()}
            self.assertTrue(mod.complete(root,identity))
            checkpoint(root,'old')
            self.assertTrue(mod.complete(root))
            self.assertFalse(mod.complete(root,identity))
            (root/'model.safetensors').write_bytes(b'broken')
            self.assertFalse(mod.complete(root))
    def test_huihui_revision_cannot_drift_from_release(self):
        release=json.loads(Path(__file__).with_name('checkpoint.huihui-lil.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                mod.prepare(release['repo'],'f'*40,Path(d),Path(d)/'hub',True)
            self.assertFalse(mod.prepare(release['repo'],release['revision'],Path(d),Path(d)/'hub',True))
    def test_radixark_missing_cache_never_downloads(self):
        release=json.loads(Path(__file__).with_name('checkpoint.radixark.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            from unittest.mock import Mock
            download=Mock(side_effect=AssertionError('download attempted'))
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=download)}):
                self.assertFalse(mod.prepare(release['repo'],release['revision'],Path(d),Path(d)/'hub'))
                download.assert_not_called()

    def test_snapshot_download_uses_same_pinned_runtime_path(self):
        with tempfile.TemporaryDirectory() as d:
            hub=Path(d)/'hub';root=hub/'models--test--model'/'snapshots'/('b'*40)
            calls=[]
            def download(**kw):calls.append(kw);checkpoint(root)
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=download)}):
                self.assertTrue(mod.prepare('test/model','b'*40,root,hub))
            self.assertEqual(calls[0]['cache_dir'],hub)
            self.assertNotIn('local_dir',calls[0])

if __name__=='__main__':unittest.main()
