"""Exercise CLI and container paths after directory changes."""
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PathTests(unittest.TestCase):
    def test_cli_outside_repository(self):
        with tempfile.TemporaryDirectory() as cwd:
            for relative in ('convert.py', 'tools/local_audit.py'):
                result = subprocess.run([sys.executable, str(ROOT / relative), '--help'],
                                        cwd=cwd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--profile', result.stdout)

    def test_qwen_container_code_mounts(self):
        script = ROOT / 'model_adapters/qwen38_huihui/run_quant_container.py'
        with patch('subprocess.run') as run, patch.object(sys, 'argv', [str(script), '--probe-only']):
            runpy.run_path(str(script), run_name='__main__')
        command = run.call_args.args[0]
        mounts = {}
        for i, value in enumerate(command):
            if value == '--mount':
                fields = dict(part.split('=', 1) for part in command[i+1].split(',') if '=' in part)
                mounts[fields['dst']] = Path(fields['src'])
        self.assertEqual(mounts['/work'], ROOT / 'model_adapters/qwen38_huihui')
        self.assertEqual(mounts['/weights_core'], ROOT / 'weights_core')
        self.assertTrue((mounts['/work'] / 'patch_experts.py').is_file())
        self.assertTrue((mounts['/modelopt'] / 'modelopt').is_dir())
        self.assertEqual(command[-1], '--probe-only')
