#!/usr/bin/env python3
"""Publish only after the explicit TP1/TP2 release-validation gate passes."""
import hashlib,json,time,os
from pathlib import Path
# Keep upload staging separate from root-owned caches created by model containers.
os.environ.setdefault('HF_XET_CACHE',str(Path.home()/'.cache/model-upload-jobs/huihui-qwen38fn/xet'))
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE','0')
os.environ.setdefault('HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION','1')
os.environ.setdefault('HF_XET_FIXED_UPLOAD_CONCURRENCY','2')
from huggingface_hub import HfApi,CommitOperationCopy
NAME='edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
root=Path(__file__).resolve().parent;reports=root/'docs';model=Path.home()/'.cache/huggingface'/NAME
status_path=reports/'publication-status.json'

def save(**values):status_path.write_text(json.dumps({'model_id':NAME,'time':time.time(),**values},indent=2))

gate=json.loads((reports/'release-validation.json').read_text());assert gate['status']=='passed'
for filename,sha in gate['evidence_sha256'].items():assert hashlib.sha256((reports/filename).read_bytes()).hexdigest()==sha,filename
api=HfApi();account=api.whoami();assert account['name']=='edp1096','Wrong HF account'
verification=json.loads((reports/'verification.json').read_text());assert verification['status']=='passed'
for filename,v in verification['shards'].items():assert (model/filename).stat().st_size==v['size']
metadata=['README.md','LICENSE','PROVENANCE.json','config.json','hf_quant_config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja','preprocessor_config.json','processor_config.json','video_preprocessor_config.json','special_tokens_map.json','added_tokens.json']
allow=list(verification['shards'])+[f for f in metadata if (model/f).is_file()]
assert len(verification['shards'])==206
save(status='uploading',files=len(allow))
try:
 # This 135 GB release exceeds the free account's private-storage allowance.
 # Keep an explicit incomplete-upload notice until all remote hashes pass.
 api.create_repo(NAME,repo_type='model',private=False,exist_ok=True)
 card=(model/'README.md').read_text()
 staging=card.replace('\n# Huihui-', '\n> Upload in progress. Do not download until this notice is removed.\n\n# Huihui-',1)
 api.upload_file(repo_id=NAME,path_in_repo='README.md',path_or_fileobj=staging.encode(),commit_message='Prepare validated model release; upload in progress')
 api.upload_file(repo_id=NAME,path_in_repo='LICENSE',path_or_fileobj=model/'LICENSE',commit_message='Include upstream model license')
 previous={f.rfilename:f for f in api.model_info(NAME,files_metadata=True).siblings}
 unchanged=[]
 for filename,v in verification['shards'].items():
  old=previous.get(filename)
  if not v['modified'] and not (old is not None and old.lfs is not None and old.lfs.sha256==v['sha256']):unchanged.append(filename)
 for first in range(0,len(unchanged),40):
  batch=unchanged[first:first+40]
  operations=[CommitOperationCopy(src_path_in_repo=f,path_in_repo=f,src_repo_id='RadixArk/Qwen3.8-Flash-Next-NVFP4',src_revision='7b719225242aacd3dbd3f9407468c2ee9a9d2594',src_repo_type='model') for f in batch]
  try:
   api.create_commit(repo_id=NAME,repo_type='model',operations=operations,commit_message='Reuse unchanged, hash-verified RadixArk weight shards')
   print('SERVER_COPIED',first+len(batch),'/',len(unchanged),flush=True)
  except Exception as e:
   print('Server copy unavailable; remaining files will use upload:',type(e).__name__,flush=True);break
 previous={f.rfilename:f for f in api.model_info(NAME,files_metadata=True).siblings}
 remaining=[]
 for filename in allow:
  if filename=='README.md':continue
  remote_file=previous.get(filename)
  if filename in verification['shards'] and remote_file is not None and remote_file.lfs is not None:
   expected=verification['shards'][filename]
   if remote_file.size==expected['size'] and remote_file.lfs.sha256==expected['sha256']:continue
  remaining.append(filename)
 for first in range(0,len(remaining),12):
  batch=remaining[first:first+12]
  api.upload_folder(repo_id=NAME,folder_path=model,repo_type='model',allow_patterns=batch,commit_message=f'Upload validated model files ({first+1}-{first+len(batch)} of {len(remaining)})')
  save(status='uploading',remaining_at_start=len(remaining),files_committed_this_run=first+len(batch))
  print('COMMITTED',first+len(batch),'/',len(remaining),flush=True)
 info=api.model_info(NAME,files_metadata=True);remote={f.rfilename:f for f in info.siblings}
 for filename,v in verification['shards'].items():
  f=remote[filename];assert f.size==v['size'],filename
  lfs=f.lfs
  sha=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']
  assert sha==v['sha256'],filename
 for filename in metadata:
  if (model/filename).is_file():assert filename in remote
 commit=api.upload_file(repo_id=NAME,path_in_repo='README.md',path_or_fileobj=model/'README.md',commit_message='Complete release after all 206 shard hashes verified')
 info=api.model_info(NAME,revision=commit.oid)
 save(status='complete',revision=info.sha,verified_shards=206,url='https://huggingface.co/'+NAME)
 print('PUBLISHED',NAME,info.sha,flush=True)
except BaseException as e:
 save(status='failed',error_type=type(e).__name__)
 raise
