import os,json
from pathlib import Path
model='/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9'
rank=os.environ['RANK'];context=int(os.environ.get('CONTEXT','1048576'))
assert rank in ('0','1') and context in (262144,524288,1048576)
# Transformers standardizes the wrapper's YaRN fields before extra kwargs.
# The wrapper must expose the text model's position limit before super().__init__.
config_source=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/config.py')
source=config_source.read_text()
anchor='        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)'
assert source.count(anchor)==1
config_source.write_text(source.replace(anchor,'        self.max_position_embeddings = self.text_config.max_position_embeddings\n'+anchor))
cfg=json.loads(Path(model,'config.json').read_text());text=cfg.get('text_config',cfg).copy();rope=text['rope_parameters'].copy()
rope.update(rope_type='yarn',factor=context/262144,original_max_position_embeddings=262144)
text.update(rope_parameters=rope,max_position_embeddings=context)
# The fork deliberately does not propagate dict hf-overrides to MTP.
# Give target and draft the same ephemeral config, retaining read-only weights.
if 'text_config' in cfg: cfg['text_config']=text
else: cfg=text
runtime=Path('/tmp/qwen38-yarn-model');runtime.mkdir(exist_ok=True)
for source in Path(model).iterdir():
 if source.name != 'config.json': (runtime/source.name).symlink_to(source)
(runtime/'config.json').write_text(json.dumps(cfg))
model=str(runtime)
override={}

args=['vllm','serve',model,'--served-model-name','qwen38-b12x-tp2','--host','127.0.0.1','--port','8013','--tensor-parallel-size','2','--nnodes','2','--node-rank',rank,'--master-addr','10.200.0.1','--master-port','29880','--block-size','16','--max-model-len',str(context),'--hf-overrides',json.dumps(override),'--dtype','bfloat16','--kv-cache-dtype','fp8','--load-format','b12x','--no-async-scheduling','--mamba-cache-mode','align','--enable-prefix-caching','--enable-chunked-prefill','--max-num-seqs','1','--max-num-batched-tokens','1024','--gpu-memory-utilization','0.70','--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':3}),'--linear-backend','b12x','--moe-backend','b12x','--gdn-decode-kernel','b12x','--no-enable-flashinfer-autotune','--reasoning-parser','qwen3','--tool-call-parser','qwen3_xml','--enable-auto-tool-choice','--mm-encoder-tp-mode','data']
if rank=='1':args+=['--headless']
print('B12X_PROBE_ARGS '+json.dumps(args),flush=True)
os.execvp(args[0],args)
