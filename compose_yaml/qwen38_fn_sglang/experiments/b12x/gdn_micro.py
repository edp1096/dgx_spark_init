"""Decode+gated norm comparison, no MTP rollback integration claim."""
import sys,json,statistics,torch
from pathlib import Path
sys.path.insert(0,'/source')
from benchmarks.benchmark_gdn_decode import BenchmarkCase,build_case,check_correctness,_reference
from b12x.sequence import gdn_decode as b
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
rows=[]
for batch in (1,4):
 row={'sequences':batch,'tokens_per_sequence':1,'key_heads':8,'value_heads':24,'state_dtype':'float32'}
 try:
  buffers=build_case(BenchmarkCase('sglang-race',(1,)*batch,8,24),device=torch.device('cuda:0'),seed=815,capacity_seqs=batch,capacity_columns=1)
  oracle=check_correctness(buffers);expected_output,expected_state=_reference(buffers)
  row['b12x_oracle']=oracle.__dict__
  binding=buffers.binding;oldstate=buffers.initial_state.clone();kernel=TritonGDNKernel();slots=binding.state_indices[:,0].contiguous()
  def old():
   out=kernel.packed_decode(binding.mixed_qkv,binding.a,binding.b,A_log=binding.A_log,dt_bias=binding.dt_bias,scale=128**-.5,ssm_states=oldstate,cache_indices=slots,num_v_heads=24,head_v_dim=128)
   return rms_norm_gated(x=out.reshape(batch,24,128),weight=binding.norm_weight,bias=None,z=binding.z,eps=1e-6,norm_before_gate=True,is_rms_norm=True,activation='sigmoid')
  def new():return b.run(binding)
  oldstate.copy_(buffers.initial_state);binding.recurrent_state.copy_(buffers.initial_state);ref=old().clone();got=new().clone();torch.cuda.synchronize()
  torch.testing.assert_close(ref,expected_output,rtol=1e-2,atol=2e-2)
  torch.testing.assert_close(oldstate,expected_state,rtol=1e-5,atol=2e-5)
  row['sglang_oracle_passed']=True
  row.update(output_relative_l2=float((ref.float()-got.float()).norm()/ref.float().norm()),state_relative_l2=float((oldstate-binding.recurrent_state).norm()/oldstate.norm()),finite=bool(got.isfinite().all()),error_code=int(binding.error_code.item()))
  for name,fn,state in [('sglang',old,oldstate),('b12x',new,binding.recurrent_state)]:
   for _ in range(3):state.copy_(buffers.initial_state);fn()
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):fn()
   times=[]
   for _ in range(21):
    state.copy_(buffers.initial_state);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();times.append(start.elapsed_time(end))
   row[name+'_graph_ms']=statistics.median(times)
 except Exception as e:row['error']=repr(e)
 rows.append(row);print(json.dumps(row),flush=True);Path('/results/gdn-micro.json').write_text(json.dumps({'rows':rows},indent=2))
