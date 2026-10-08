"""Independent ModelOpt decode comparison and inverse GDN layout checks."""
import tempfile,hashlib
from pathlib import Path
import torch
from safetensors.torch import save_file
from model_adapters.qwen38_lil_huihui.build import decode,encode,restore_columns
from weights_core.safetensors_io import Checkpoint
from model_adapters.qwen38_lil_huihui.refine_scales import hash_regions

def main():
 torch.set_num_threads(2);torch.manual_seed(731)
 for kind in ('mxfp8','nvfp4'):
  for shape in ((3,64),(64,128)):
   x=torch.randn(shape)*.03
   parts,reference=encode(x,kind)
   with tempfile.TemporaryDirectory() as td:
    save_file({'p.'+k:v for k,v in parts.items()},str(Path(td)/'model.safetensors'))
    actual,detected=decode(Checkpoint(td),'p')
    assert detected==kind
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    assert torch.isfinite(actual).all()
 config={'text_config':{'linear_num_key_heads':2,'linear_num_value_heads':6,'linear_value_head_dim':4}}
 hf=torch.arange(3*24).reshape(3,24)
 # GGUF stores value-head group before key-head, HF stores key-head first.
 gguf=hf.reshape(3,2,3,4).transpose(1,2).contiguous().reshape(3,24)
 assert torch.equal(restore_columns(gguf,config),hf)
 with tempfile.TemporaryDirectory() as td:
  file=Path(td)/'bytes';file.write_bytes(b'0123456789')
  for intervals,expected in [([],b'0123456789'),([(2,4),(6,9)],b'01459'),([(0,10)],b'')]:
   h=hash_regions(file,intervals)
   assert h['full']==hashlib.sha256(b'0123456789').hexdigest()
   assert h['outside']==hashlib.sha256(expected).hexdigest()
  try:hash_regions(file,[(1,5),(4,7)])
  except AssertionError:pass
  else:raise AssertionError('Overlapping mutation ranges accepted')
 print('PASS: both checkpoint decoders match ModelOpt exactly; GDN columns restored')
if __name__=='__main__':main()
