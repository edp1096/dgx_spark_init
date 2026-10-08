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
 if a[-2:]==['docker','load']:sys.stdin.read()
'''

@unittest.skipUnless(os.name == "posix", "Linux cluster shell recipes")
class ClusterPreparationTests(unittest.TestCase):
 def test_both_cluster_recipes_prepare_missing_images_and_weights(self):
  for kind in ['ds41','qwen38-tp2']:
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
    if kind=='ds41':
     self.assertIn('pack_experts.py',text)
     self.assertIn('/packed/rank0',text)
     self.assertIn('/packed/rank1',text)
    else:
     self.assertNotIn('--gpus',text)

if __name__=='__main__':unittest.main()
