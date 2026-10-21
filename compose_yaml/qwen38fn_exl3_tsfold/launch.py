"""Serve the EXL3 checkpoint with cached vision and a YaRN context view through TensorFold's own HTTP API."""
import argparse
from pathlib import Path

import json
import hashlib
import os

MODEL = 'qwen38fn_exl3'
MTP_DEPTH = 5
MTP_CONFIDENCE = 0.4
EXPERT_PROFILE = 'compact-parallel-v1'


def prepare_runtime(source, cache, limit=1048576):
    if limit not in (262144, 524288, 1048576):
        raise ValueError('Unsupported context window')
    required = [f'model-{i:05d}-of-00007.safetensors' for i in range(1, 8)] + ['ngram_embedding.safetensors', 'tokenizer.json']
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError('Original Huihui EXL3 checkpoint is incomplete; run model preparation')
    view = cache / 'model-yarn4'
    view.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'config.json').read_text())
    text = config['text_config']
    if limit != 262144:
        text['rope_parameters'].update(rope_type='yarn', factor=limit / 262144,
                                      original_max_position_embeddings=262144)
    text['max_position_embeddings'] = config['max_position_embeddings'] = limit
    for path in source.iterdir():
        if path.name == 'config.json' or not path.is_file():
            continue
        target = view / path.name
        if target.is_symlink():
            if target.readlink() == path:
                continue
            target.unlink()
        elif target.exists():
            raise RuntimeError(f'Unexpected file in runtime model view: {target.name}')
        target.symlink_to(path)
    temp = view / 'config.json.tmp'
    temp.write_text(json.dumps(config, indent=2))
    temp.replace(view / 'config.json')
    return view


def acquire_runtime(cache):
    """Own this recipe's cache before recovering interrupted extension builds."""
    import fcntl
    cache.mkdir(parents=True, exist_ok=True)
    handle = (cache / 'server.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another EXL3 TensorFold server owns this runtime cache') from None
    # The lifetime lock is released by the OS on exit, including SIGKILL.
    # This dedicated cache is not shared with other recipes or build workers.
    extensions = Path(os.environ.get('TORCH_EXTENSIONS_DIR', cache / 'torch-extensions'))
    if extensions.resolve() == (cache / 'torch-extensions').resolve():
        for lock in extensions.glob('*/lock'):
            lock.unlink(missing_ok=True)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18024)
    parser.add_argument('--name', default=MODEL)
    parser.add_argument('--kept-prefixes', type=int, choices=(1, 2, 8), default=1)
    parser.add_argument('--cache', type=Path, default=Path('/cache'))
    args = parser.parse_args()
    runtime_lock = acquire_runtime(args.cache)
    receipt = args.cache / 'runtime-profile.json'
    receipt.unlink(missing_ok=True)
    source = args.model
    args.model = prepare_runtime(source, args.cache)
    from tokenizers import Tokenizer
    from prepare_vision import prepare
    from tensorfold.cuda.server import App, serve
    from tensorfold.families.qwen4_exp.cuda import engine as engine_module
    from tensorfold.families.qwen4_exp.cuda.graphs import Graphs
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    import time
    engine_module.KEEP = args.kept_prefixes
    capture_metrics = {'captures': 0, 'seconds': 0.0}
    capture = Graphs._capture
    def measured_capture(graphs, fn):
        started = time.perf_counter()
        result = capture(graphs, fn)
        capture_metrics['captures'] += 1
        capture_metrics['seconds'] += time.perf_counter() - started
        return result
    Graphs._capture = measured_capture
    from tensorfold.families.qwen4_exp.cuda.weight_types import draft_token_ids
    os.environ['TENSORFOLD_VISION_WEIGHTS'] = str(prepare(source, args.cache))
    tok = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    ids = set(draft_token_ids('default').tolist())
    for token in range(tok.get_vocab_size()):
        if any('\uac00' <= c <= '\ud7a3' for c in tok.decode([token], skip_special_tokens=False)):
            ids.add(token)
    args.cache.mkdir(parents=True, exist_ok=True)
    vocab = args.cache / 'qwen38-exl3-ko-draft-vocab.txt'
    vocab.write_text(''.join(f'{token}\n' for token in sorted(ids)))
    from compact_experts import install as install_compaction
    from parallel_group import install as install_grouping
    from tensorfold.families.qwen4_exp.cuda import exl3_pack, exl3_mm
    install_compaction()
    exl3_pack.MOE_WINDOW = exl3_mm.MOE_WINDOW = 2048
    install_grouping()
    # Ordinary tensors can be used from HTTP worker threads as well as the
    # initialization thread. Do not allocate the engine inside inference_mode.
    engine = FlashNextEngine(args.model, depth=MTP_DEPTH, confidence=MTP_CONFIDENCE, draft_vocab=str(vocab),
                              max_len=1048576, context_explicit=True, kv_dtype='int8',
                              streams=2, graphs=True, vision=True, prefetch=False)
    # Warm the OS file cache without mlocking the 36 GiB n-gram table.
    # Pages can be reclaimed for a large KV cache or an auxiliary model.
    tables = {id(layer.ple.table): layer.ple.table for layer in engine.w.layers if layer.ple is not None}
    for table in tables.values():
        table.prefetch()
    print('[sparktalk] n-gram file cache warmed; pages remain reclaimable', flush=True)
    app = App(engine, args.model, args.name, default_thinking=False,
              sampling={'temperature': 0}, max_tokens=4096, context_window=1048576,
              vision_max_images=50, vision_image_tokens=16384)
    from tensorfold.cuda import health
    counters = health.of(app)
    snapshot = counters.snapshot
    def with_graph_metrics(app):
        return {**snapshot(app), 'cuda_graphs': dict(capture_metrics), 'kept_prefixes': args.kept_prefixes}
    counters.snapshot = with_graph_metrics
    receipt.write_text(json.dumps({'model': args.name, 'context': engine.context_window,
                                   'kv': engine.kv_dtype, 'depth': engine.depth,
                                   'vision': engine.vision is not None,
                                   'confidence': engine.confidence,
                                   'draft_vocabulary': len(ids),
                                   'expert_tiles': EXPERT_PROFILE,
                                   'moe_window': engine.w.x3.moe.rows,
                                   'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                     for name in ['compact_experts.py', 'parallel_group.py']}}, indent=2))
    print('SPARKTALK_QWEN_TENSORFOLD_READY context=1048576 kv=int8 mtp=5 confidence=0.4 expert_tiles=compact-parallel-v1 vision=1', flush=True)
    app.auth = None
    try:
        serve(app, args.host, args.port)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
