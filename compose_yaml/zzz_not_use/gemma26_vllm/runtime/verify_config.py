"""Exercise the pinned vLLM HF override loader with actual local model configs."""
import json,sys
from pathlib import Path
from launch import command
from vllm.transformers_utils.config import get_config
for family,path in zip(('ornith35','gemma26'),sys.argv[1:]):
 for context in (262144,524288,1048576):
  args=command({'MODEL_FAMILY':family,'MODEL_PATH':path,'SERVED_MODEL_NAME':'validation','CONTEXT_LENGTH':str(context),'RUNTIME_VIEW_ROOT':'/tmp/runtime-views'})
  config=get_config(args[2],trust_remote_code=False)
  text=config.get_text_config()
  assert text.max_position_embeddings==context
  assert text.num_hidden_layers==(40 if family=='ornith35' else 30)
  rope=text.rope_parameters if family=='ornith35' else text.rope_parameters['full_attention']
  assert rope.get('factor',1)==context/262144
  assert rope['partial_rotary_factor']==.25
  print(family,context,'runtime config loaded correctly',flush=True)
