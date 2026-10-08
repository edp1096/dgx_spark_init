import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("prepare_models", Path(__file__).with_name("prepare_models.py"))
prepare_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_models)


class CheckpointCacheTest(unittest.TestCase):
    def test_verified_startup_cache_is_advised_without_skipping_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.safetensors"
            path.write_bytes(b"qualified fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files = [("qualified/repo", "pinned-revision", path.name, digest, "unused")]
            with patch.object(prepare_models, "FILES", files), patch.object(prepare_models, "try_to_load_from_cache", return_value=str(path)), patch.object(prepare_models.os, "posix_fadvise") as advice:
                rows = prepare_models.prepare(link=False, release_file_cache=True)
                self.assertEqual(rows[0]["sha256"], digest)
                advice.assert_called_once()
                advice.reset_mock()
                path.write_bytes(b"corrupted fixture")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    prepare_models.prepare(link=False, release_file_cache=True)
                advice.assert_not_called()


if __name__ == "__main__":
    unittest.main()
