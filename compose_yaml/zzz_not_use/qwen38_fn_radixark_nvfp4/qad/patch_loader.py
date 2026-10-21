"""Apply the indexed MTP file filter before either native iterator path."""
import ast
from pathlib import Path
p=Path('/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py')
s=p.read_text()
old='''        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
'''
assert s.count(old)==1
s=s.replace(old,'''        if use_safetensors:
            from qad_loader import mtp_files
            hf_weights_files = mtp_files(hf_folder, hf_weights_files, source.model_config)

'''+old)
ast.parse(s)
p.write_text(s)
