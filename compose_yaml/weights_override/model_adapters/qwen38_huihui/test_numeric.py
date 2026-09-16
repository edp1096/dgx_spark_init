import unittest
import numpy as np
from build import restore_gdn_columns,bf16_to_float,float_to_bf16

class NumericChecks(unittest.TestCase):
 def test_gdn_roundtrip(self):
  original=np.arange(2*16*3*128,dtype=np.float32).reshape(2,16,3,128)
  gguf=original.transpose(0,2,1,3).copy().reshape(-1)
  config={'text_config':{'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_value_head_dim':128}}
  np.testing.assert_array_equal(restore_gdn_columns(gguf,2,config),original.reshape(-1))
 def test_bf16_ties_even(self):
  values=np.array([1,1+1/256,1+3/256,-1-1/256],dtype=np.float32)
  np.testing.assert_array_equal(bf16_to_float(float_to_bf16(values)),np.array([1,1,1+2/128,-1],dtype=np.float32))
 def test_reject_nonfinite(self):
  for value in [np.inf,-np.inf,np.nan]:
   with self.assertRaises(ValueError):float_to_bf16(np.array([value],dtype=np.float32))
if __name__=='__main__':unittest.main()
