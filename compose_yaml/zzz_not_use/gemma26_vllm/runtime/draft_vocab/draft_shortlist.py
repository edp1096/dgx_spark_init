"""TP1 draft-head-only projection. Preserve full token IDs for verification."""
import hashlib,json,os,pathlib
import torch
import torch.nn.functional as F

def prepare(model, filename):
    if getattr(model.lm_head, 'tp_size', 1) != 1:
        raise ValueError('Draft shortlist supports TP1 only')
    if getattr(model, 'masked_embedding', None) is not None:
        raise ValueError('Do not combine centroid masking and a static shortlist')
    data=json.loads(pathlib.Path(filename).read_text())
    tokenizer=pathlib.Path(os.environ['MODEL_PATH'])/'tokenizer.json'
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest()!=data['tokenizer_sha256']:
        raise ValueError('Draft shortlist tokenizer hash mismatch')
    weight=model.lm_head.weight
    vocab=model.logits_processor.org_vocab_size
    ids=data['ids']
    if not ids or ids!=sorted(set(ids)) or ids[0]<0 or ids[-1]>=vocab:
        raise ValueError('Invalid draft token IDs')
    if weight.ndim!=2:
        raise ValueError('Draft shortlist requires a matrix output head')
    projection_dtype=getattr(model.lm_head,'params_dtype',weight.dtype)
    if model.logits_processor.head_dtype not in (None,projection_dtype):
        raise ValueError('Different head_dtype is not supported in this experiment')
    index=torch.tensor(ids,device=weight.device,dtype=torch.long)
    model.register_buffer('_shortlist_ids',index,persistent=False)
    if weight.dtype in (torch.float16,torch.bfloat16,torch.float32):
        model.register_buffer('_shortlist_weight',weight.index_select(0,index).contiguous(),persistent=False)
    else:
        prepare_nvfp4(model,index)
    print(f'Draft shortlist prepared: {len(ids)}/{vocab}; source head dtype={weight.dtype}',flush=True)
    model._shortlist_vocab=vocab

def logits(model, hidden_states):
    if not hasattr(model,'_shortlist_ids'):
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Initialize shortlist before CUDA graph capture')
        prepare(model,os.environ['DRAFT_SHORTLIST'])
    if hasattr(model,'_shortlist_head'):
        head=model._shortlist_head
        values=head.quant_method.apply(head,hidden_states)
    else:
        values=F.linear(hidden_states,model._shortlist_weight)
    processor=model.logits_processor
    if processor.soft_cap is not None:
        values=torch.tanh(values/processor.soft_cap)*processor.soft_cap
    if processor.scale!=1.0:
        values=values*processor.scale
    result=values.new_full((*values.shape[:-1],model._shortlist_vocab),-float('inf'))
    result.index_copy_(-1,model._shortlist_ids,values)
    suppressed=getattr(model,'_suppress_token_ids',None)
    if suppressed:result[...,suppressed]=-float('inf')
    return result


def prepare_nvfp4(model,index):
    from safetensors import safe_open
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    original=model.lm_head
    method=original.quant_method
    if 'NvFp4' not in type(method).__name__:
        raise ValueError('Unsupported quantized shortlist head: '+type(method).__name__)
    root=pathlib.Path(os.environ['MODEL_PATH'])
    mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    keys={name:mapping['lm_head.'+name] for name in ['weight','weight_scale','weight_scale_2']}
    with torch.device(original.weight.device):
        head=ParallelLMHead(len(index),original.embedding_dim,params_dtype=original.params_dtype,
                            quant_config=method.quant_config,prefix='lm_head')
    if type(head.quant_method) is not type(method):
        raise ValueError('Shortlist head quantization differs from target')
    cpu_ids=index.cpu()
    with torch.no_grad():
        for name,shard in keys.items():
            with safe_open(str(root/shard),framework='pt',device='cpu') as f:
                data=f.get_tensor('lm_head.'+name)
            if data.ndim:
                dtype=data.dtype
                if dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
                    data=data.float().index_select(0,cpu_ids).to(dtype)
                else:data=data.index_select(0,cpu_ids)
            getattr(head,name).copy_(data.to(original.weight.device))
        if hasattr(head,'input_scale'):
            scale=getattr(original,'input_global_scale',None)
            if scale is None:head.input_scale.fill_(1)
            else:head.input_scale.copy_(scale)
        head.quant_method.process_weights_after_loading(head)
        probe=torch.arange(original.embedding_dim,device=original.weight.device,dtype=torch.float32).sin().to(original.params_dtype).reshape(1,-1)
        expected=method.apply(original,probe).index_select(-1,index)
        actual=head.quant_method.apply(head,probe)
        torch.testing.assert_close(actual,expected,rtol=0.01,atol=0.03)
    model.add_module('_shortlist_head',head)
