import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

assets = Path(__file__).resolve().parents[2] / 'internal/orchestrator/assets/qwen38fn_exl3'


def module(name):
    spec = importlib.util.spec_from_file_location(name, assets / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launch = module('launch')
release = module('release_weight_cache')


def checkpoint(root):
    root.mkdir()
    cfg = {'text_config': {'max_position_embeddings': 262144, 'rope_parameters':
        {'rope_type': 'default', 'rope_theta': 10000000, 'partial_rotary_factor': .25,
         'mrope_section': [11, 11, 10], 'mrope_interleaved': True}}}
    (root / 'config.json').write_text(json.dumps(cfg))
    for name in [f'model-{i:05d}-of-00007.safetensors' for i in range(1, 8)] + ['ngram_embedding.safetensors', 'tokenizer.json']:
        (root / name).write_bytes(b'fixture')


class Qwen38FNEXL3Tests(unittest.TestCase):
    def test_yarn_view_preserves_original_files_and_rope_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);source = base / 'original';checkpoint(source)
            original = (source / 'config.json').read_bytes()
            path = launch.prepare_runtime(source, base / 'runtime')
            view = base / 'runtime/models' / launch.MODEL
            cfg = json.loads((view / 'config.json').read_text());rope = cfg['text_config']['rope_parameters']
            self.assertEqual(cfg['max_position_embeddings'], 1048576)
            self.assertEqual(rope['factor'], 4)
            self.assertEqual(rope['original_max_position_embeddings'], 262144)
            self.assertEqual(rope['mrope_section'], [11, 11, 10])
            self.assertEqual(rope['partial_rotary_factor'], .25)
            self.assertEqual((source / 'config.json').read_bytes(), original)
            self.assertEqual((view / 'ngram_embedding.safetensors').resolve(), (source / 'ngram_embedding.safetensors').resolve())
            self.assertTrue(path.is_file())
            launch.prepare_runtime(source, base / 'runtime')
            with self.assertRaises(ValueError):launch.prepare_runtime(source, base / 'other', 2097152)

    @unittest.skipUnless(hasattr(os, 'posix_fadvise'), 'Linux file cache advice')
    def test_open_checkpoint_rejects_advice_and_ple_is_never_advised(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);source = base / 'original';checkpoint(source)
            process = base / 'proc';process.mkdir();(process / 'fd').mkdir();(process / 'maps').write_text('')
            descriptor = process / 'fd/1';descriptor.symlink_to(source / 'model-00001-of-00007.safetensors')
            with patch.object(release.os, 'posix_fadvise') as advise:
                with self.assertRaises(RuntimeError):release.release_cache(source, process)
                advise.assert_not_called()
                descriptor.unlink()
                descriptor.symlink_to(source / 'ngram_embedding.safetensors')
                result = release.release_cache(source, process)
                self.assertEqual(advise.call_count, 7)
                self.assertNotIn('ngram_embedding.safetensors', result['files'])


if __name__ == '__main__':unittest.main()
