import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from direct_camoufox_client import (
    DirectCamoufoxClient,
    RESULT_PREFIX,
    source_state,
)
from direct_camoufox_worker import user_agent


class Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class DirectCamoufoxClientTest(unittest.TestCase):
    def test_user_agent_ignores_sandbox_pid_prefix(self):
        output = "[10271] Sandbox warning\nCamoufox Camoufox 152.0.4-beta.28\n"
        with patch("direct_camoufox_worker.subprocess.run", return_value=Completed(stdout=output)):
            self.assertEqual(
                user_agent(),
                "Mozilla/5.0 (X11; Linux x86_64; rv:152.0) Gecko/20100101 Firefox/152.0",
            )

    def test_source_state_extracts_supjav_servers(self):
        state = source_state({
            "url": "https://supjav.com/1.html",
            "source": '''<html><head><title>Movie</title></head><body>
              <a class="btn-server active" data-link="tv-token">TV</a>
              <a class="btn-server" data-link="st-token">ST</a>
            </body></html>''',
        })
        self.assertEqual(state["title"], "Movie")
        self.assertEqual(state["servers"], [
            {"index": 0, "name": "TV", "link": "tv-token"},
            {"index": 1, "name": "ST", "link": "st-token"},
        ])

    def test_source_state_extracts_parts(self):
        state = source_state({
            "source": '''<button class="btn-cd">1</button><button class="btn-cd">2</button>
              <div class="cd-server"><a class="btn-server" data-link="a">ST</a></div>
              <div class="cd-server"><a class="btn-server" data-link="b">DS</a></div>''',
        })
        self.assertEqual([part["label"] for part in state["parts"]], ["1", "2"])
        self.assertEqual(state["parts"][1]["sources"][0]["link"], "b")

    def test_inspect_reads_worker_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = DirectCamoufoxClient()
            client.python = str(root / "python")
            client.worker = str(root / "worker")
            client.xvfb_run = str(root / "xvfb-run")
            for path in (client.python, client.worker, client.xvfb_run):
                Path(path).touch()
            payload = {"ok": True, "data": {
                "url": "https://supjav.com/1.html",
                "source": "<html><title>Ready</title></html>",
            }}
            with patch("direct_camoufox_client.subprocess.run", return_value=Completed(
                stdout=RESULT_PREFIX + json.dumps(payload)
            )):
                self.assertEqual(client.inspect("https://supjav.com/1.html")["title"], "Ready")


if __name__ == "__main__":
    unittest.main()
