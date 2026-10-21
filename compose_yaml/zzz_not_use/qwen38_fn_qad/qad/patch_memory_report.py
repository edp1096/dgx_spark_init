"""Expose the QAD receipt through the existing server_info request."""
import ast
from pathlib import Path

p = Path('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py')
s = p.read_text()
anchor = '        ret["startup_time"] = self.startup_time\n'
if 'ret["qad_memory"]' not in s:
    if s.count(anchor) != 1:
        raise RuntimeError('Unexpected scheduler source for QAD memory receipt')
    s = s.replace(anchor, '''        from memory_report import report as qad_memory_report
        ret["qad_memory"] = qad_memory_report(self, torch)
''' + anchor)
    ast.parse(s)
    p.write_text(s)
