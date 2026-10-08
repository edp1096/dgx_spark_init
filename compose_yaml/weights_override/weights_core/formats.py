"""Read-only base precision checks; never change checkpoint precision."""
import json
from pathlib import Path


def detect_format(base):
    config=json.loads((Path(base)/'config.json').read_text())
    quant=config.get('quantization_config',{})
    algorithms=set()
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('quant_algo','quant_method') and isinstance(item,str):algorithms.add(item.lower())
                elif key not in ('kv_cache_scheme','input_activations','ignore'):visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(quant)
    if algorithms & {'nvfp4','w4a16_nvfp4'}:return 'nvfp4'
    if algorithms & {'fp8','fp8_per_channel_per_token','fbgemm_fp8','auto_fp8'}:return 'fp8'
    # Older FP8 checkpoints may have no quantization metadata.
    from .safetensors_io import read_header
    dtypes=set()
    for file in Path(base).glob('*.safetensors'):
        header,_=read_header(file)
        dtypes.update(t['dtype'] for k,t in header.items() if k!='__metadata__' and not any(s in k for s in ('scale','zero_point')))
    if 'F8_E4M3' in dtypes and not (dtypes & {'U8','I8','I32'}):return 'fp8'
    return None


def validate_format(base, requested='auto'):
    detected=detect_format(base)
    if requested!='auto' and requested!=detected:
        raise ValueError(f'--format {requested} does not match base format {detected or "unknown"}')
    return detected
