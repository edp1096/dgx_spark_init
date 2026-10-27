"""Run the pinned GLM FP4-Local profile without changing checkpoint files."""
from pathlib import Path
import os,json,subprocess,sys
e=os.environ
for key in ['NODE_RANK','MODEL_HOST_PATH','HEAD_RAIL_IP','GLM53_HOST_IP','NCCL_IB_HCA']:e[key]
if not e.get('NCCL_IB_GID_INDEX'):
 wanted='0000:0000:0000:0000:0000:ffff:'+''.join(f'{int(x):02x}' for x in e['GLM53_HOST_IP'].split('.')[:2])+':'+''.join(f'{int(x):02x}' for x in e['GLM53_HOST_IP'].split('.')[2:])
 p=Path('/sys/class/infiniband')/e['NCCL_IB_HCA']/'ports/1'
 for f in (p/'gids').iterdir():
  if f.read_text().strip()==wanted and 'v2' in (p/'gid_attrs/types'/f.name).read_text():e['NCCL_IB_GID_INDEX']=f.name;break
 if not e.get('NCCL_IB_GID_INDEX'):raise ValueError('No matching RoCE v2 GID')
if e.get('MAX_MODEL_LEN','1048576')!='1048576' or e.get('MAX_NUM_SEQS','1')!='1':raise ValueError('Qualified profile requires 1M context and one request')
mtp=e.get('MTP_TOKENS','5')
if mtp not in ['1','5']:raise ValueError('Qualified MTP depths: 1 or 5')
subprocess.run([sys.executable,'/opt/glm53/check_models.py',e['MODEL_HOST_PATH']],check=True)
subprocess.run([sys.executable,'/opt/glm53/prepare_chat_template.py',e['MODEL_HOST_PATH'],'/root/.cache/glm53-chat.jinja'],check=True)
defaults={'SERVED_MODEL_NAME':'glm-5.3-flash','API_BIND':'0.0.0.0','API_PORT':'8000','MASTER_PORT':'29521'}
args=json.loads(Path('/opt/glm53/serve-args.json').read_text())
args=[e.get(x[1:-1],defaults.get(x[1:-1],'')) if x.startswith('@') and x.endswith('@') else x for x in args]
i=args.index('--speculative-config')+1
spec=json.loads(args[i]);spec['num_speculative_tokens']=int(mtp);args[i]=json.dumps(spec)
if e['NODE_RANK']!='0':args.append('--headless')
os.execvp('vllm',['vllm',*args])
