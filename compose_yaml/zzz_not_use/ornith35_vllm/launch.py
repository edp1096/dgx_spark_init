"""Offline TP1 launcher. Immutable weights with a separate runtime config view."""
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

def command(env):
 family=env['MODEL_FAMILY'];path=Path(env['MODEL_PATH'])
 if family != 'ornith35':raise ValueError('Unsupported model family')
 if not (path/'config.json').is_file():raise ValueError(f'Local model is missing: {path}')
 config=json.loads((path/'config.json').read_text());text=config['text_config']
 context=int(env.get('CONTEXT_LENGTH','1048576'))
 if context not in (262144,524288,1048576):raise ValueError('Unsupported context length')
 original=262144;factor=context/original
 rope={**text['rope_parameters']}
 if factor>1:rope.update(rope_type='yarn',factor=factor,original_max_position_embeddings=original)
 text['rope_parameters']=rope
 reasoning,tools='qwen3','qwen3_xml'
 text['max_position_embeddings']=context
 view=config_view(path,config,Path(env.get('RUNTIME_VIEW_ROOT','/root/.cache/vllm/runtime-models')))
 args=['vllm','serve',str(view),'--served-model-name',env['SERVED_MODEL_NAME'],
       '--host','0.0.0.0','--port','8000','--tensor-parallel-size','1',
       '--max-model-len',str(context),
       '--max-num-seqs','1','--gpu-memory-utilization',env.get('GPU_MEMORY_UTILIZATION','0.70'),
       '--kv-cache-memory-bytes',env.get('KV_CACHE_BYTES',str(32*1024**3)),
       '--kv-cache-dtype','fp8','--enable-prefix-caching','--enable-chunked-prefill',
       '--max-num-batched-tokens',env.get('PREFILL_CHUNK','1024'),
       '--enable-auto-tool-choice','--tool-call-parser',tools,'--reasoning-parser',reasoning,
       '--moe-backend',env.get('MOE_BACKEND','b12x'),
       '--limit-mm-per-prompt','{"image":4,"video":0}','--mm-processor-cache-gb','0.5']
 tokens=int(env.get('MTP_TOKENS','0'))
 if tokens not in (0,1,3):raise ValueError('MTP_TOKENS must be 0, 1 or 3')
 if env.get('DRAFT_VOCAB','off') not in ('off','ko64k'):raise ValueError('DRAFT_VOCAB must be off or ko64k')
 if tokens:
  draft_context=int(env.get('DRAFT_CONTEXT_LENGTH',str(context)))
  if draft_context<1 or draft_context>context:raise ValueError('Draft context exceeds target context')
  spec={'method':'mtp','num_speculative_tokens':tokens,'max_model_len':draft_context}
  spec['moe_backend']='flashinfer_cutlass'
  args+=['--speculative-config',json.dumps(spec)]
 if env.get('ENFORCE_EAGER','0')=='1':args.append('--enforce-eager')
 return args

def draft_vocab_path(family,model_path,vocab_root):
 if family != 'ornith35':raise ValueError('Unsupported model family')
 prefix='ornith'
 digest=hashlib.sha256((Path(model_path)/'tokenizer.json').read_bytes()).hexdigest()
 matches=[p for p in Path(vocab_root).glob(prefix+'-*ko64k.json')
          if json.loads(p.read_text()).get('tokenizer_sha256')==digest]
 if len(matches)!=1:
  raise ValueError(f'Expected one verified {prefix} ko64k list for tokenizer {digest}; found {len(matches)}. Set DRAFT_VOCAB=off to use the full draft vocabulary.')
 return matches[0]

if __name__=='__main__':
 family=os.environ['MODEL_FAMILY']
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']='1'
 if int(os.environ.get('MTP_TOKENS','0')) and os.environ.get('DRAFT_VOCAB','off')=='ko64k':
  vocab_root=Path(__file__).parent/'draft_vocab'
  os.environ['DRAFT_SHORTLIST']=str(draft_vocab_path(family,os.environ['MODEL_PATH'],vocab_root))
  os.environ['PYTHONPATH']=str(vocab_root)+(':'+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
 else:
  os.environ.pop('DRAFT_SHORTLIST',None)
 args=command(os.environ)
 print('TP1 model:',family,'context:',os.environ.get('CONTEXT_LENGTH','1048576'),flush=True)
 os.execvp(args[0],args)
