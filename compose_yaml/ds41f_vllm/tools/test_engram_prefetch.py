"""Forecast boundaries: partial prefix reuse, mixed batches, final chunk and exclusion."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engram_prefetch import next_chunk
class Boundaries(unittest.TestCase):
 def test_prompt_only_and_longest_remaining(self):
  b=S(req_ids=['a','b','c'],is_prefilling_np=[True,True,False],idx_mapping_np=[2,0,1],
      num_computed_prefill_tokens_np=[4096,0,9000],num_scheduled_tokens=[4096,8192,6])
  states=S(prompt_len=S(np=[20000,9000,12000]))
  self.assertEqual(next_chunk(b,states,8192),(0,8192,8192))
  self.assertEqual(next_chunk(b,states,8192,{'b'}),(2,8192,3808))
  self.assertIsNone(next_chunk(b,states,8192,{'a','b'}))
  b.num_computed_prefill_tokens_np=[8192,16384,9000]
  b.num_scheduled_tokens=[3808,3616,6]
  self.assertIsNone(next_chunk(b,states,8192))
if __name__=='__main__': unittest.main()
