"""CPU XGrammar masks for the native CUDA sampler; language weights stay native."""
import functools
import json
from pathlib import Path

import torch
import xgrammar as xgr
from transformers import PreTrainedTokenizerFast


@functools.lru_cache(maxsize=2)
def compiler(model):
    model = Path(model)
    cfg = json.loads((model / 'config.json').read_text())
    cfg = cfg.get('text_config', cfg)
    vocab = int(cfg['vocab_size'])
    tok = PreTrainedTokenizerFast(tokenizer_file=str(model / 'tokenizer.json'))
    stops = cfg.get('eos_token_id', [])
    if isinstance(stops, int):
        stops = [stops]
    for token in ('<|im_end|>', '<|endoftext|>'):
        value = tok.convert_tokens_to_ids(token)
        if value is not None and value not in stops:
            stops.append(value)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=stops)
    return xgr.GrammarCompiler(info), vocab


@functools.lru_cache(maxsize=32)
def compile_schema(model, kind, text):
    comp, _ = compiler(model)
    if kind in ('json', 'json_schema'):
        schema = {'type': 'object'} if kind == 'json' else text
        return comp.compile_json_schema(schema, max_whitespace_cnt=2)
    if kind == 'regex':
        return comp.compile_regex(text)
    if kind == 'choice':
        text = 'root ::= ' + ' | '.join(json.dumps(x, ensure_ascii=False) for x in json.loads(text))
    return comp.compile_grammar(text)


def create(model, kind, text):
    compiled = compile_schema(model, kind, text)
    _, vocab = compiler(model)
    return xgr.GrammarMatcher(compiled), xgr.allocate_token_bitmask(1, vocab)


def mask(state, tokens):
    matcher, bits = state
    for token in tokens:
        if not matcher.accept_token(token):
            raise ValueError(f'Native sampler committed a token outside its grammar: {token}')
    matcher.fill_next_token_bitmask(bits)
    assert bits.device.type == 'cpu' and bits.dtype == torch.int32
    return bits.numpy().tobytes()
