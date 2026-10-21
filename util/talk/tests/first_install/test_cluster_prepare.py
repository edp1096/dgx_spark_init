"""Exercise empty-host lifecycle scripts with recording CLIs, never real GPUs/SSH."""
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CLI='''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name;a=sys.argv[1:]
with open(os.environ['AUDIT_TRACE'],'a') as f:f.write(json.dumps([name,a])+'\\n')
state=Path(os.environ['AUDIT_STATE'])
if name=='docker':
 if a[:2]==['image','inspect']:
  if not state.exists():sys.exit(1)
  print('sha256:test')
 elif a and a[0]=='build':state.touch()
 elif a and a[0]=='save':sys.stdout.write('test image')
 elif a and a[0]=='run' and '-i' in a:sys.stdin.read()
elif name=='ssh':
 if 'image inspect' in ' '.join(a):sys.exit(1)
 if a[-2:]==['docker','load'] or 'tar -xzf -' in ' '.join(a):sys.stdin.read()
'''

@unittest.skipUnless(os.name == "posix", "Linux cluster shell recipes")
class ClusterPreparationTests(unittest.TestCase):
 def test_qwen_cluster_prepares_missing_images_and_weights(self):
  for kind in ['qwen38-tp2']:
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
    root=Path(d);recipe=root/'recipe';recipe.mkdir();tools=root/'bin';tools.mkdir()
    with tarfile.open(ROOT/f'internal/orchestrator/assets/recipes/{kind}.tar.gz') as t:t.extractall(recipe,filter='data')
    for name in ['docker','ssh','rsync']:
     p=tools/name;p.write_text(CLI);p.chmod(0o755)
    trace=root/'trace';env={**os.environ,'PATH':str(tools)+':/usr/bin:/bin','AUDIT_TRACE':str(trace),'AUDIT_STATE':str(root/'image'),'HF_CACHE':str(root/'hf'),'WORKER_HF_CACHE':str(root/'worker-hf'),'WORKER_HOST':'test@worker','REMOTE_COMPOSE_DIR':str(root/'remote'),'DSV41_IMAGE':'test:ds41','QWEN_TP2_IMAGE':'test:qwen','HF_TOKEN':'test-private-token'}
    r=subprocess.run(['bash',str(recipe/'prepare.sh'),'setup'],env=env,capture_output=True,text=True,timeout=10)
    self.assertEqual(r.returncode,0,r.stderr)
    text=trace.read_text();self.assertNotIn('test-private-token',text)
    rows=[json.loads(l) for l in text.splitlines()]
    self.assertTrue(any(n=='docker' and a[0]=='build' for n,a in rows))
    self.assertTrue(any(n=='docker' and a[0]=='save' for n,a in rows))
    self.assertTrue(any(n=='rsync' for n,a in rows))
    self.assertIn('snapshot_download',text)
    self.assertNotIn('--gpus',text)

 def test_ds41_setup_validates_prepared_release_without_downloading_or_starting(self):
  # The pinned TensorFold release has a separate preparation/backup contract.
  # Setup must refuse missing assets and must never revive the retired download.
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);recipe=root/'recipe';recipe.mkdir();tools=root/'bin';tools.mkdir()
   with tarfile.open(ROOT/'internal/orchestrator/assets/recipes/ds41.tar.gz') as t:t.extractall(recipe,filter='data')
   for name in ['docker','ssh','rsync']:
    p=tools/name;p.write_text(CLI);p.chmod(0o755)
   trace=root/'trace';state=root/'image';state.touch()
   hf=root/'hf'
   (recipe/'.env').write_text('DSV41_IMAGE_ID=sha256:test\n')
   env={**os.environ,'PATH':str(tools)+':/usr/bin:/bin','AUDIT_TRACE':str(trace),'AUDIT_STATE':str(state),'HF_CACHE':str(hf),'WORKER_HF_CACHE':str(root/'worker-hf'),'WORKER_HOST':'test@worker','REMOTE_COMPOSE_DIR':str(root/'remote')}
   command=['bash',str(recipe/'prepare.sh'),'setup']
   missing=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
   self.assertNotEqual(missing.returncode,0)
   self.assertIn('prerequisite missing',missing.stderr)
   base=hf/'tensorfold/deepseek-v41-flash-1017'
   files=['models/abliterated/config.json','models/abliterated/model.safetensors.index.json','models/abliterated/ABLIT_META.json','engram/rank0/engram-l1-r0of2.bin','engram/rank0/engram-l14-r0of2.bin','assets/rank0/rope.bin','assets/rank0/engram-host.bin','assets/rank0/aot/aot.json','verification/pack.verified.json']
   for name in files:
    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
   prepared=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
   self.assertEqual(prepared.returncode,0,prepared.stderr)
   text=trace.read_text();rows=[json.loads(l) for l in text.splitlines()]
   self.assertIn('check-rank 1',text)
   self.assertNotIn('snapshot_download',text)
   self.assertNotIn('pack_experts.py',text)
   self.assertTrue(all(a[:2]==['image','inspect'] for n,a in rows if n=='docker'))

if __name__=='__main__':unittest.main()
