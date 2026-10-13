import hashlib,json,os,shutil,tempfile
from pathlib import Path
def config_view(source,config,root):
 source=source.resolve();root=root.resolve()
 if source==root or source in root.parents:raise ValueError('Runtime view root must be outside the source model')
 entries=sorted(source.iterdir(),key=lambda p:p.name)
 encoded=json.dumps(config,sort_keys=True,indent=2)+'\n'
 identity=hashlib.sha256((str(source)+'\n'+encoded+'\n'.join(p.name for p in entries)).encode()).hexdigest()
 root.mkdir(parents=True,exist_ok=True);target=root/identity
 if target.exists():
  if (target/'config.json').read_text()!=encoded:raise ValueError('Runtime config view mismatch')
  return target
 stage=Path(tempfile.mkdtemp(prefix='.config-',dir=root))
 try:
  for item in entries:
   if item.name!='config.json':(stage/item.name).symlink_to(item,target_is_directory=item.is_dir())
  (stage/'config.json').write_text(encoded)
  try:stage.rename(target)
  except FileExistsError:
   if (target/'config.json').read_text()!=encoded:raise ValueError('Concurrent runtime config mismatch')
  return target
 finally:
  if stage.exists():shutil.rmtree(stage)

if __name__ == '__main__':
 source=Path(os.environ['MODEL_PATH']);context=int(os.environ.get('CONTEXT_LENGTH','1048576'))
 if context not in (262144,524288,1048576):raise ValueError('Unsupported context length')
 cfg=json.loads((source/'config.json').read_text());cfg['text_config']['max_position_embeddings']=context
 cfg['text_config']['rope_parameters']['full_attention']['factor']=context/262144
 view=config_view(source,cfg,Path('/root/.cache/sglang/runtime-models'))
 args=['python3','-m','sglang.launch_server','--model-path',str(view),'--served-model-name',os.environ['SERVED_MODEL_NAME'],'--host','0.0.0.0','--port','30000','--tp-size','1','--context-length',str(context),'--mem-fraction-static','0.70','--max-total-tokens',str(context+4096),'--swa-full-tokens-ratio','0.03125','--max-running-requests','1','--attention-backend','triton','--chunked-prefill-size','4096','--disable-prefill-cuda-graph','--cuda-graph-max-bs-decode','2','--kv-cache-dtype','fp8_e4m3','--tool-call-parser','gemma4','--reasoning-parser','gemma4','--chat-template','/opt/gemma4/chat_template.jinja','--enable-strict-thinking','--mm-feature-transport','cpu','--enable-metrics','--skip-server-warmup']
 quant=json.dumps(cfg.get('quantization_config',{})).lower()
 if 'nvfp4' in quant or ('"num_bits": 4' in quant and '"type": "float"' in quant):args+=['--moe-runner-backend','flashinfer_cutlass']
 count=int(os.environ.get('MTP_TOKENS','1'))
 if count not in (0,1,3):raise ValueError('Unsupported draft count')
 if os.environ.get('DRAFT_VOCAB','off')!='off':raise ValueError('SGLang Gemma uses the full draft vocabulary')
 if count:
  draft=Path(os.environ['DRAFT_MODEL_PATH']);dc=json.loads((draft/'config.json').read_text());dc['text_config']['max_position_embeddings']=context;dc['text_config']['rope_parameters']['full_attention']['factor']=context/262144
  dv=config_view(draft,dc,Path('/root/.cache/sglang/runtime-drafts'))
  args+=['--speculative-algorithm','NEXTN','--speculative-draft-model-path',str(dv),'--speculative-num-steps',str(count),'--speculative-eagle-topk','1','--speculative-num-draft-tokens',str(count+1),'--speculative-draft-kv-cache-dtype','fp8_e4m3']
 print('Gemma26 SGLang:',args,flush=True);os.execvp(args[0],args)
