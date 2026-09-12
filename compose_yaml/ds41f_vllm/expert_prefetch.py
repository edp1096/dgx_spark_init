"""Opt-in early-router probe and controlled expert-prefetch experiments.

Predictions never replace real routing. A control epoch clears only expert
cache metadata between completed benchmark requests; weights/graphs stay put.
"""
import json,os,time
from collections import Counter
from pathlib import Path
import torch

ENABLED=os.environ.get('DSV41_PREFETCH_TEST')=='1'
CONTROL=Path('/opt/ds41/prefetch-control.json')
config={'mode':'off','epoch':0}
predictions={}
trace=None
trace_path=None
step=0

def refresh():
    global config,trace,trace_path,step
    if not CONTROL.exists(): return
    candidate=json.loads(CONTROL.read_text())
    if candidate==config: return
    if candidate.get('mode','off') not in ('off','probe','early'):
        raise ValueError('Unsupported expert-prefetch mode')
    if candidate.get('epoch',0)!=config.get('epoch',0):
        import b12x_slots
        for cache in b12x_slots._layers.values():
            cache.reset()
        predictions.clear();step=0
        for key in b12x_slots._stats: b12x_slots._stats[key]=0
    config=candidate
    if trace is not None: trace.close();trace=None
    trace_path=None
    print('EXPERT_PREFETCH_CONTROL '+json.dumps(config),flush=True)

def before_attention(decoder,x,input_ids,residual,post_mix,res_mix,attn_pre):
    global step
    owner=decoder.ffn
    layer=owner.experts.layer_index
    if layer==0:
        refresh();step+=1
    mode=config.get('mode')
    if mode not in ('probe','early') or not 0<=layer<40 or not 0<x.shape[0]<=16:
        predictions.pop(layer,None)
        return
    from vllm.forward_context import get_forward_context
    if get_forward_context().attn_metadata is None: return
    if mode=='early' and layer not in config.get('layers',list(range(40))): return
    def predict(value):
        logits,_=owner.gate(value)
        if owner.scoring_func!='sqrtsoftplus':
            raise ValueError('Probe qualified for V4.1 sqrtsoftplus routing only')
        scores=torch.nn.functional.softplus(logits.float()).sqrt()
        scores=scores+owner.gate.e_score_correction_bias
        return scores.topk(32 if mode=='probe' else int(config.get('width',6)),dim=-1).indices.tolist()
    predicted={'attention_input':predict(x)} if mode=='probe' else {}
    # V4.1 changes the residual-stream collapse between attention and FFN.
    # Respect the known residual mixing while approximating attention as zero.
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    known=mhc_post_tilelang(torch.zeros_like(x),residual,post_mix,res_mix)
    collapsed=(known.float()*attn_pre.unsqueeze(-1)).sum(1).to(x.dtype)
    predicted['residual_input']=predict(decoder.ffn_norm(collapsed))
    if mode=='probe':
        predictions[layer]=predicted
    else:
        import b12x_slots
        cache=b12x_slots._layers.get(layer)
        if cache is None: return
        scores=Counter()
        for row in predicted['residual_input']:
            for rank,expert in enumerate(row):
                if expert not in cache.used: scores[expert]+=1/(rank+1)
        chosen=[e for e,_ in scores.most_common(int(config.get('budget',2)))]
        cache.start_prefetch(chosen)

def observe(cache,needed,tokens):
    global trace,trace_path
    if config.get('mode')!='probe' or cache.layer not in predictions: return
    from vllm.distributed import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank()!=0: return
    if trace is None:
        trace_path=Path('/cache')/config.get('trace','expert-routing-probe.jsonl')
        trace=trace_path.open('a')
    record={'step':step,'layer':cache.layer,'tokens':tokens,
            'predicted':predictions.pop(cache.layer),'needed':needed,
            'resident':list(cache.used)}
    trace.write(json.dumps(record,separators=(',',':'))+'\n')
    if cache.layer==39: trace.flush()
