FROM dgx-sglang-qwen38-qad:sm121-v5
COPY qad/qad_gdn.py /opt/qad-tp1/qad_gdn.py
RUN python3 - <<'PY'
from pathlib import Path
import ast
p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/logits_processor.py')
s=p.read_text().replace('import logging\n', 'import logging\nimport os\n');old='and lm_head.weight.shape == (248320, 2560)):'
assert s.count(old)==1
s=s.replace(old,'''and (lm_head.weight.shape == (248320, 2560)
                     or (lm_head.weight.shape == (65536, 2560)
                         and hidden_states.shape[0] == 1
                         and os.environ.get("SGLANG_QAD_DRAFT_GEMV") == "1"))):''')
ast.parse(s);p.write_text(s)
PY
