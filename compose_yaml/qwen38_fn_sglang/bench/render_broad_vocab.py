"""Run CPU-only inside the serving image, using the installed model tokenizer."""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
out=Path('/out')
model='/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9'
tokenizer=AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
tasks=json.loads((out/'tasks.json').read_text())['tasks']
for t in tasks:
    text=tokenizer.apply_chat_template(t['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=t['thinking'])
    ids=tokenizer.encode(text,add_special_tokens=False)
    t['input_ids']=ids
    t['rendered_prompt']=text
    assert len(ids)+t['max_tokens']<65536
    print(t['id'],len(ids),t['max_tokens'],flush=True)
payload=dict(tokenizer=model,eos_token_id=tokenizer.eos_token_id,tasks_sha256=hashlib.sha256((out/'tasks.json').read_bytes()).hexdigest(),tasks=tasks)
(out/'prepared.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
