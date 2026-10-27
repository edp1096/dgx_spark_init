import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

import music


def audio(seconds, quiet_tail=0):
    data = io.BytesIO()
    with wave.open(data, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
        f.writeframes(b"\x01\x01" * int(8000 * (seconds-quiet_tail)) + b"\0\0" * int(8000 * quiet_tail))
    return base64.b64encode(data.getvalue()).decode()


class MusicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.project = SimpleNamespace(session_id="conversation")
        self.plan = {"music": {"kind":"generated", "model":"ace-turbo", "prompt":"soft piano", "seed":42, "bpm":80}}

    def test_short_generation_fails_without_padding_or_synthesis(self):
        def bridge(operation, request):
            return {"audio":audio(1), "metadata":{"model":"ace-turbo"}}
        with self.assertRaisesRegex(ValueError, "길이가 부족"):
            music.prepare(self.project, self.plan, self.folder, None, 10, bridge)
        self.assertFalse((self.folder/"bgm.wav").exists())

    def test_reuses_exact_request_but_regenerates_changed_prompt(self):
        calls = []
        def bridge(operation, request):
            calls.append(request)
            return {"audio":audio(10), "metadata":{"model":"ace-turbo"}}
        first = music.prepare(self.project, self.plan, self.folder, None, 10, bridge)
        revised = self.folder/"revision"; revised.mkdir()
        second = music.prepare(self.project, self.plan, revised, self.folder, 10, bridge)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.plan["music"]["prompt"] = "bright jazz piano"
        third = music.prepare(self.project, self.plan, revised, self.folder, 10, bridge)
        self.assertNotEqual(first["fingerprint"], third["fingerprint"])
        self.assertEqual(len(calls), 2)

    def test_backend_failure_propagates(self):
        def bridge(operation, request):
            raise RuntimeError("CUDA failure")
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            music.prepare(self.project, self.plan, self.folder, None, 10, bridge)
        self.assertFalse((self.folder/"bgm.wav").exists())

    def test_full_length_file_with_early_music_ending_is_rejected_and_retried(self):
        calls = []
        def bridge(operation, request):
            calls.append(request)
            return {"audio":audio(10, 4 if len(calls)==1 else 0), "metadata":{"model":"ace-turbo"}}
        record = music.prepare(self.project, self.plan, self.folder, None, 10, bridge)
        self.assertEqual(len(calls), 2)
        self.assertFalse(record["attempts"][0]["accepted"])
        self.assertTrue(record["attempts"][1]["accepted"])
        self.assertEqual(record["request"]["seed"], 43)
        self.assertEqual(record["request"]["duration_seconds"], 15)
        self.assertEqual(record["quality"]["trailing_quiet_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
