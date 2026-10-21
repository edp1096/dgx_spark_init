"""Serve the qualified text-only EXL3 trial through TensorFold's own HTTP API."""
import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tensorfold.cuda.server import App, serve
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.weight_types import draft_token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18024)
    parser.add_argument('--name', default='qwen38fn_exl3_tensorfold')
    parser.add_argument('--cache', type=Path, default=Path('/cache'))
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    ids = set(draft_token_ids('default').tolist())
    for token in range(tok.get_vocab_size()):
        if any('\uac00' <= c <= '\ud7a3' for c in tok.decode([token], skip_special_tokens=False)):
            ids.add(token)
    args.cache.mkdir(parents=True, exist_ok=True)
    vocab = args.cache / 'qwen38-exl3-ko-draft-vocab.txt'
    vocab.write_text(''.join(f'{token}\n' for token in sorted(ids)))
    # Ordinary tensors can be used from HTTP worker threads as well as the
    # initialization thread. Do not allocate the engine inside inference_mode.
    engine = FlashNextEngine(args.model, depth=3, confidence=0.0, draft_vocab=str(vocab),
                              max_len=262144, context_explicit=True, kv_dtype='int8',
                              streams=1, graphs=True, vision=False)
    app = App(engine, args.model, args.name, default_thinking=False,
              sampling={'temperature': 0}, max_tokens=4096, context_window=262144)
    app.auth = None
    try:
        serve(app, args.host, args.port)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
