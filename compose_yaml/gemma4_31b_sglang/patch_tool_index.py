"""Backport SGLang PR #25075 (a60207960d04f5c4d2e7487c9dbb190230217f71)."""
import re
from pathlib import Path

path = Path('/sgl-workspace/sglang/python/sglang/srt/function_call/gemma4_detector.py')
source = path.read_text()
if 'tool_index=self.current_tool_id' in source and '_tool_indices' not in source:
    print('Gemma4 tool-call index patch already applied')
else:
    replacements = [
        ('        self._tool_indices: Optional[dict] = None\n', ''),
        ('            tool_indices = self._get_tool_indices(tools)\n', ''),
        ('            for func_name, args_str in matches:', '            for i, (func_name, args_str) in enumerate(matches):'),
        ('tool_index=tool_indices.get(func_name, -1)', 'tool_index=i'),
        ('        if self._tool_indices is None:\n            self._tool_indices = self._get_tool_indices(tools)\n', ''),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected Gemma4 parser source: {old!r}')
        source = source.replace(old, new)
    source, count = re.subn(r'tool_index=self\._tool_indices\.get\(\s*(?:func_name|self.current_func_name), -1\s*\)', 'tool_index=self.current_tool_id', source)
    if count != 2:
        raise RuntimeError('Expected two streaming tool index sites')
    compile(source, str(path), 'exec')
    path.write_text(source)
    print('Applied Gemma4 sequential tool-call indices')
