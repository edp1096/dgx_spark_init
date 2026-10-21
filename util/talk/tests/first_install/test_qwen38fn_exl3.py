import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

assets = Path(__file__).resolve().parents[2] / 'internal/orchestrator/assets/qwen38fn_exl3'


def module(name):
    spec = importlib.util.spec_from_file_location(name, assets / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launch = module('launch')


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
            view = base / 'runtime/model-yarn4'
            cfg = json.loads((view / 'config.json').read_text());rope = cfg['text_config']['rope_parameters']
            self.assertEqual(cfg['max_position_embeddings'], 1048576)
            self.assertEqual(rope['factor'], 4)
            self.assertEqual(rope['original_max_position_embeddings'], 262144)
            self.assertEqual(rope['mrope_section'], [11, 11, 10])
            self.assertEqual(rope['partial_rotary_factor'], .25)
            self.assertEqual((source / 'config.json').read_bytes(), original)
            self.assertEqual((view / 'ngram_embedding.safetensors').resolve(), (source / 'ngram_embedding.safetensors').resolve())
            self.assertTrue(path.is_dir())
            launch.prepare_runtime(source, base / 'runtime')
            with self.assertRaises(ValueError):launch.prepare_runtime(source, base / 'other', 2097152)

    def test_incomplete_checkpoint_and_unexpected_view_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);source = base / 'original';checkpoint(source)
            view = launch.prepare_runtime(source, base / 'runtime')
            target = view / 'tokenizer.json';target.unlink();target.write_text('user data')
            with self.assertRaises(RuntimeError):launch.prepare_runtime(source, base / 'runtime')
            self.assertEqual(target.read_text(), 'user data')
            (source / 'ngram_embedding.safetensors').unlink()
            with self.assertRaises(RuntimeError):launch.prepare_runtime(source, base / 'missing')

    @unittest.skipUnless(os.name == 'posix', 'Linux runtime lock')
    def test_restart_recovers_stale_build_lock_without_stealing_live_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory);locks = cache / 'torch-extensions/fixture';locks.mkdir(parents=True)
            stale = locks / 'lock';stale.touch()
            first = launch.acquire_runtime(cache)
            self.assertFalse(stale.exists())
            stale.touch()
            try:
                with self.assertRaises(RuntimeError):launch.acquire_runtime(cache)
                self.assertTrue(stale.exists())
            finally:first.close()
            with launch.acquire_runtime(cache):self.assertFalse(stale.exists())


if __name__ == '__main__':unittest.main()
