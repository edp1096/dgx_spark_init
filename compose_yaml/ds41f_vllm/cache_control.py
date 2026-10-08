"""Opt-in benchmark reset, executed before the model and outside CUDA graphs."""
import json,os
from pathlib import Path
epoch=None
graphs_enabled=True
expert_io_mode=os.environ.get('DSV41_EXPERT_IO','serial')
kernel_tokens=int(os.environ.get("DSV41_KERNEL_TOKENS","512"))
shared_buffers=os.environ.get("DSV41_SHARED_BUFFERS","0")=="1"
routed_pipeline=os.environ.get('DSV41_ROUTED_PIPELINE','0')=='1'
record_routes=False
final_decoder_rows=os.environ.get('DSV41_FINAL_DECODER_ROWS','0')=='1'
validate_final_decoder=False
def update():
    global epoch,graphs_enabled,expert_io_mode,record_routes,kernel_tokens,shared_buffers,routed_pipeline,final_decoder_rows,validate_final_decoder
    path=Path('/opt/ds41/graph-control.json')
    if not path.exists(): return
    control=json.loads(path.read_text())
    value=control['epoch']
    if value==epoch: return
    import b12x_slots
    for cache in b12x_slots._layers.values(): cache.reset()
    for key in b12x_slots._stats: b12x_slots._stats[key]=0
    kernel_tokens=int(control.get("kernel_tokens",os.environ.get("DSV41_KERNEL_TOKENS","512")))
    shared_buffers=bool(control.get("shared_buffers",os.environ.get("DSV41_SHARED_BUFFERS","0")=="1"))
    if kernel_tokens not in (512,1024,2048): raise ValueError("Invalid expert kernel capacity")
    routed_pipeline=bool(control.get('routed_pipeline',os.environ.get('DSV41_ROUTED_PIPELINE','0')=='1'))
    final_decoder_rows=bool(control.get('final_decoder_rows',os.environ.get('DSV41_FINAL_DECODER_ROWS','0')=='1'))
    validate_final_decoder=bool(control.get('validate_final_decoder',False))
    epoch=value
    graphs_enabled=bool(control.get('graphs',True))
    expert_io_mode=control.get('expert_io',os.environ.get('DSV41_EXPERT_IO','serial'))
    record_routes=bool(control.get('record_routes',False))
    from expert_io import MODES
    if expert_io_mode not in MODES: raise ValueError('Invalid benchmark expert I/O mode')
    print('EXPERT_CACHE_RESET '+str(epoch),flush=True)
