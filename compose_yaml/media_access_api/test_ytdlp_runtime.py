import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ytdlp_runtime as runtime


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base.json"
        self.base.write_text(json.dumps({"sha256": "old", "tag": "2026.07.04"}))
        self.patches = [patch.object(runtime, "ROOT", self.root), patch.object(runtime, "BASE_LOCK", self.base)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_update_and_rollback_to_image(self):
        release = self.root / "new"
        release.mkdir()
        lock = {"sha256": "new", "tag": "2026.08.19"}
        (release / "lock.json").write_text(json.dumps(lock))
        with patch.object(runtime, "latest_lock", return_value=lock), patch.object(runtime, "install", return_value=release):
            self.assertTrue(runtime.mutate("update")["changed"])
        self.assertEqual(runtime.current_lock(), lock)
        self.assertTrue(runtime.status()["rollback_available"])
        self.assertEqual(runtime.mutate("rollback")["current"]["sha256"], "old")

    def test_failed_install_keeps_current(self):
        with patch.object(runtime, "latest_lock", return_value={"sha256": "new"}), patch.object(runtime, "install", side_effect=ValueError("bad hash")):
            with self.assertRaises(ValueError):
                runtime.mutate("update")
        self.assertEqual(runtime.current_lock()["sha256"], "old")
        self.assertFalse((self.root / "current").exists())

    def test_same_version_does_not_install(self):
        with patch.object(runtime, "latest_lock", return_value={"sha256": "old"}), patch.object(runtime, "install") as install:
            self.assertFalse(runtime.mutate("update")["changed"])
            install.assert_not_called()

    def test_concurrent_update_rejected(self):
        with (self.root / "update.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                runtime.mutate("update")

    def test_hash_failure_never_runs_pip(self):
        from io import BytesIO
        lock = {"tag": "2026.08.19", "version": "2026.8.19", "url": "https://files.pythonhosted.org/test.whl", "sha256": "bad"}
        with patch.object(runtime, "urlopen", return_value=BytesIO(b"invalid")), patch.object(runtime.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                runtime.install(lock)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
