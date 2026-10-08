FROM eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262
# Transformers' YaRN validation reads the enclosing config's native context
# before PretrainedConfig consumes kwargs. Mirror the text config beforehand.
RUN python3 - <<'PY'
from pathlib import Path
import ast
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/config.py')
s=p.read_text()
old='        self.rope_parameters = rope_parameters or getattr(\n'
assert s.count(old)==1
s=s.replace(old,'        self.max_position_embeddings = self.text_config.max_position_embeddings\n'+old)
ast.parse(s)
p.write_text(s)
PY
