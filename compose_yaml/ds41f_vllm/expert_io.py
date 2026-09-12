"""Actual-route expert I/O scheduling; no predicted or additional reads."""
import json,os
from concurrent.futures import ThreadPoolExecutor

import torch

MODES = ('serial', 'overlap', 'batch', 'batch_overlap')
DEFAULT_MODE = os.environ.get('DSV41_EXPERT_IO', 'serial')
if DEFAULT_MODE not in MODES:
    raise ValueError(f'Invalid DSV41_EXPERT_IO: {DEFAULT_MODE}')
BENCH_CONTROL = os.environ.get('DSV41_BENCH_CONTROL') == '1'
_streams = {}
_executors = {}
_trace_file = None
_trace_epoch = None
_submitter = ThreadPoolExecutor(max_workers=1, thread_name_prefix='expert-batch')


def mode():
    if BENCH_CONTROL:
        import cache_control
        return cache_control.expert_io_mode
    return DEFAULT_MODE


def overlaps():
    return mode() in ('overlap', 'batch_overlap')


def read_stream():
    """Fence prior slot consumers before queuing independent GPU work."""
    device = torch.cuda.current_device()
    if device not in _streams:
        _streams[device] = torch.cuda.Stream(device=device)
    stream = _streams[device]
    stream.wait_stream(torch.cuda.current_stream())
    return stream.cuda_stream


def batch_executor():
    from b12x.loader._native import load
    device = torch.cuda.current_device()
    if device not in _executors:
        native = load()
        workers = int(os.environ.get('DSV41_READ_THREADS', '4'))
        _executors[device] = (native, native.batch_executor(device, workers))
    return _executors[device]


def submit_batch(records, stream):
    native, executor = batch_executor()
    return _submitter.submit(native.batch_execute, executor, records, stream)


def trace_routes(layer, tokens, capacity, needed, rank):
    global _trace_file, _trace_epoch
    if not BENCH_CONTROL:
        return
    import cache_control
    if not cache_control.record_routes:
        if _trace_file is not None:
            _trace_file.close()
            _trace_file = None
        return
    if _trace_file is None or _trace_epoch != cache_control.epoch:
        if _trace_file is not None:
            _trace_file.close()
        _trace_epoch = cache_control.epoch
        path = f'/cache/route-trace-{_trace_epoch}-rank{rank}.jsonl'
        _trace_file = open(path, 'w', buffering=1)
        print('EXPERT_ROUTE_TRACE ' + path, flush=True)
    _trace_file.write(json.dumps({'layer':layer,'tokens':tokens,
        'capacity':capacity,'needed':needed})+'\n')
