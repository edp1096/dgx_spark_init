"""Read-only views for fused HF experts exported as individual ModelOpt linears."""
import re
from weights_core.safetensors_io import Checkpoint


def expand_experts(checkpoint, base, prefix_mode="identity"):
    tensors = {}
    for name, tensor in checkpoint.tensors.items():
        match = re.fullmatch(r'(.+\.experts\.)(gate_up_proj|down_proj)(?:\.weight)?', name)
        if not match or name in base.tensors:
            tensors[name] = tensor
            continue
        shape = tensor['shape']
        if tensor['dtype'] != 'BF16' or len(shape) != 3:
            raise ValueError(f'Unsupported fused expert layout: {name}')
        experts, rows, cols = shape
        if match[2] == 'gate_up_proj' and rows % 2:
            raise ValueError(f'Odd fused gate/up row count: {name}')
        prefix=match[1]
        if prefix_mode=='gemma4':prefix=re.sub(r'(\.layers\.\d+)\.experts\.$',r'\1.moe.experts.',prefix)
        elif prefix_mode!='identity':raise ValueError('Unsupported expert prefix mode')
        parts = [('gate_proj', 0, rows//2), ('up_proj', rows//2, rows//2)] if match[2]=='gate_up_proj' else [('down_proj', 0, rows)]
        for expert in range(experts):
            for projection, first, count in parts:
                key = f'{prefix}{expert}.{projection}.weight'
                if key in tensors or key in checkpoint.tensors:raise ValueError(f'Duplicate expanded tensor: {key}')
                if key not in base.tensors:
                    raise ValueError(f'Missing expanded destination: {key}')
                begin = tensor['data_offsets'][0] + (expert*rows + first)*cols*2
                tensors[key] = {**tensor, 'shape':[count,cols],
                                'data_offsets':[begin,begin+count*cols*2],
                                'source_tensor':name,'expert_index':expert}
    if len(tensors) < len(checkpoint.tensors):
        raise ValueError('Expert expansion lost tensors')
    checkpoint.tensors = tensors
    return checkpoint


def aligned_checkpoints(original, donor, base, profile):
    original, donor, base = [Checkpoint(path) for path in (original,donor,base)]
    adapter = profile.get('tensor_adapter')
    if adapter == 'fused_experts':
        expand_experts(original,base,profile.get('expert_prefix_mode','identity'));expand_experts(donor,base,profile.get('expert_prefix_mode','identity'))
    elif adapter is not None:
        raise ValueError(f'Unsupported tensor adapter: {adapter}')
    return original,donor,base
