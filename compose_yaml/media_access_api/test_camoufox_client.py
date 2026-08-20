import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from camoufox_client import CamoufoxClient, CamoufoxError, RESULT_PREFIX


class Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def configured_client(tmp_path: Path) -> CamoufoxClient:
    python = tmp_path / "python"
    worker = tmp_path / "worker.py"
    xvfb_run = tmp_path / "xvfb-run"
    python.touch()
    worker.touch()
    xvfb_run.touch()
    client = CamoufoxClient()
    client.python = str(python)
    client.worker = str(worker)
    client.xvfb_run = str(xvfb_run)
    return client


class CamoufoxClientTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.client = configured_client(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_inspect_reads_prefixed_result(self):
        result = {"ok": True, "data": {"title": "ready"}}
        with patch("camoufox_client.subprocess.run", return_value=Completed(
            stdout="browser noise\n" + RESULT_PREFIX + json.dumps(result) + "\n"
        )):
            self.assertEqual(
                self.client.inspect("https://example.com/video"), {"title": "ready"}
            )

    def test_inspect_runs_worker_in_an_isolated_x_display(self):
        result = {"ok": True, "data": {"title": "ready"}}
        with patch("camoufox_client.subprocess.run", return_value=Completed(
            stdout=RESULT_PREFIX + json.dumps(result) + "\n"
        )) as run:
            self.client.inspect("https://example.com/video")
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [
            self.client.xvfb_run,
            "-a",
            "-s",
            "-screen 0 1920x1080x24 -nolisten tcp +extension GLX +render",
        ])
        self.assertEqual(command[-2:], [self.client.python, self.client.worker])

    def test_inspect_reports_worker_error(self):
        result = {"ok": False, "error": "access verification blocked"}
        with patch("camoufox_client.subprocess.run", return_value=Completed(
            stdout=RESULT_PREFIX + json.dumps(result) + "\n", returncode=1
        )):
            with self.assertRaisesRegex(CamoufoxError, "access verification blocked"):
                self.client.inspect("https://example.com/video")

    def test_resolve_keeps_worker_cookies_and_headers(self):
        state = {
            "url": "https://example.com/video",
            "userAgent": "Camoufox/135",
            "videos": ["https://cdn.example.com/movie.mp4"],
            "cookies": [{"name": "cf_clearance", "value": "token"}],
        }
        with patch.object(self.client, "inspect", return_value=state):
            candidates, cookies, headers = self.client.resolve("https://example.com/video")
        self.assertEqual(candidates, ["https://cdn.example.com/movie.mp4"])
        self.assertEqual(cookies[0]["name"], "cf_clearance")
        self.assertEqual(headers, {
            "User-Agent": "Camoufox/135",
            "Referer": "https://example.com/video",
        })
