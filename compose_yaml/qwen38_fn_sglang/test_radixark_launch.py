import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from radixark_launch import mtp_files
class RadixArkDraftFiles(unittest.TestCase):
 def test_only_mtp_shards_and_missing_shard_fails(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'model.w':'main.safetensors','mtp.w':'draft.safetensors'}}))
   cfg=SimpleNamespace(is_draft_model=True,hf_config=SimpleNamespace(model_type='qwen4_exp'),quantization='modelopt_fp4')
   files=[str(root/'main.safetensors'),str(root/'draft.safetensors')]
   self.assertEqual(mtp_files(d,files,cfg),files[1:])
   with self.assertRaises(ValueError):mtp_files(d,files[:1],cfg)
   cfg.is_draft_model=False;self.assertEqual(mtp_files(d,files,cfg),files)
if __name__=='__main__':unittest.main()
