#!/usr/bin/env python3
"""Create public release metadata without including local paths or host details."""
import json,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
NAME='edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
root=Path(__file__).resolve().parent;reports=root/'docs';model=Path.home()/'.cache/huggingface'/NAME
license_file=hf_hub_download('Qwen/Qwen3.8-Flash-Next','LICENSE',revision='de4b8e4d43b917e7706784d8bb445c9af86a3540');shutil.copyfile(license_file,model/'LICENSE')
m=json.loads((model/'transfer-manifest.json').read_text());v=json.loads((reports/'verification.json').read_text());s=json.loads((reports/'runtime-summary.json').read_text())
provenance={'model_id':NAME,'method':m['method'],'gdn_layout':m['gdn_layout'],'expert_layers':[x['layer'] for x in m['expert_layers']],'expert_quantizer':m['expert_quantizer'],'activation_calibration':m['activation_calibration'],'sources':{'RadixArk/Qwen3.8-Flash-Next-NVFP4':'7b719225242aacd3dbd3f9407468c2ee9a9d2594','Qwen/Qwen3.8-Flash-Next':'de4b8e4d43b917e7706784d8bb445c9af86a3540','unsloth/Qwen3.8-Flash-Next-GGUF':'38bb39ee97821de2c9009abb7e93950eec396e66','huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF':'7e3bfc316b880fefeb049596f11c49d6a18e05fb'},'verification':{k:v[k] for k in ['status','target_tensors','target_tensors_with_changed_bytes','unchanged_tensors','modified_shards']},'shards':v['shards']}
(model/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
card=(reports/'MODEL_CARD.md').read_text()
(model/'README.md').write_text(card);(reports/'MODEL_CARD.md').write_text(card)
print('Prepared release card, license and sanitized provenance:',NAME)
