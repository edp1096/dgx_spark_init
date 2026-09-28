import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

src = Path(__file__).resolve().parents[2] / 'internal/orchestrator/assets/model-prepare/prepare.py'
spec = importlib.util.spec_from_file_location('prepare', src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    (root/'config.json').write_text('{}')
    header=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'1234')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'model.safetensors'}}))


class PreparationTests(unittest.TestCase):
    def test_empty_download_then_reuse_without_network_or_local_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'model';item={'repo':'test/model','path':str(root)};calls=[]
            def download(**kwargs):
                calls.append(kwargs);checkpoint(Path(kwargs['local_dir']))
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=download,hf_hub_download=download)}):
                mod.prepare(item,'token')
                self.assertTrue(mod.complete(root,item))
                mod.prepare(item,'token')
                self.assertEqual(len(calls),1)
    def test_missing_truncated_and_escaping_shards_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint(root)
            self.assertTrue(mod.complete(root,{}))
            data=(root/'model.safetensors').read_bytes()
            (root/'model.safetensors').write_bytes(data[:-1])
            self.assertFalse(mod.complete(root,{}))
            checkpoint(root)
            (root/'model.safetensors.index.json').write_text('{"weight_map":{"w":"../outside"}}')
            self.assertFalse(mod.complete(root,{}))
            (root/'model.safetensors.index.json').write_text('{"weight_map":{"w":"missing"}}')
            self.assertFalse(mod.complete(root,{}))
    def test_gguf_magic_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'a.gguf';item={'files':['a.gguf']}
            p.write_bytes(b'error page');self.assertFalse(mod.complete(root,item))
            p.write_bytes(b'GGUFdata');self.assertTrue(mod.complete(root,item))

if __name__=='__main__':unittest.main()
