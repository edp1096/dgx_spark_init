"""Qualified MiniMax H3 text-to-video Sol-Attn hook for the resident worker."""
from contextlib import contextmanager
import os

import torch
import torch.nn.functional as F


MODE = os.getenv('MMH3_ATTENTION', 'sol')
if MODE not in ('sol', 'dense'):
    raise ValueError('MMH3_ATTENTION must be sol or dense')


def configuration():
    return {'attention': MODE, 'spectrum': False, 'fbc': False,
            'tau': 1.0, 'dense_first_steps': 4, 'dense_last_steps': 2,
            'dense_first_blocks': 2, 'dense_text_audio_queries': True}


class SolRun:
    def __init__(self, steps):
        from sol_attn import sol_attn, get_sol_attn_backend
        self.backend = get_sol_attn_backend()
        if self.backend != 'cute_sm121':
            raise RuntimeError(f'Sol-Attn requires cute_sm121; got {self.backend}')
        self.kernel = sol_attn
        self.steps = steps
        self.step = -1
        self.sigma = self.layout = None
        self.sol_calls = self.dense_calls = self.first_blocks = 0

    def first(self, args, extra):
        sigma = float(args['transformer_options']['sigmas'].max())
        if self.sigma is None or sigma != self.sigma:
            self.step += 1
            self.sigma = sigma
        self.layout = args['layout']
        self.first_blocks += 1
        return extra['original_block'](args)

    def attention(self, func, q, k, v, heads, mask=None, skip_reshape=False,
                  skip_output_reshape=False, transformer_options=None, **kwargs):
        opts = transformer_options or {}
        if (self.layout is None or not 4 <= self.step < self.steps - 2
                or opts.get('block_index', -1) < 2 or mask is not None
                or not skip_reshape or skip_output_reshape or q.ndim != 4
                or q.shape[2] != self.layout.seq_len or q.shape[-1] != 128
                or q.dtype != torch.bfloat16):
            self.dense_calls += 1
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)
        sinks = next(a for a, b, kind in self.layout.segments if kind == 'video')
        qb, kb, vb = [x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v)]
        out = self.kernel(qb, kb, vb, tau=1.0, thresh_type='diag',
                          sink_start=0, sink_tokens=sinks)
        # Dense prefix KV and dense text/audio queries, matching the measured run.
        prefix = F.scaled_dot_product_attention(q[:, :, :sinks], k, v)
        out[:, :sinks] = prefix.permute(0, 2, 1, 3)
        self.sol_calls += 1
        return out.reshape(q.shape[0], q.shape[2], heads * 128)

    def statistics(self):
        return {**configuration(), 'backend': self.backend, 'sol_calls': self.sol_calls,
                'dense_calls': self.dense_calls, 'first_blocks': self.first_blocks}

    @contextmanager
    def patch(self, model):
        saved = model.model_options
        options = {**saved}
        transformer = {**options.get('transformer_options', {})}
        replacements = {**transformer.get('patches_replace', {})}
        blocks = {**replacements.get('dit', {})}
        blocks[('double_block', 0)] = self.first
        replacements['dit'] = blocks
        transformer['patches_replace'] = replacements
        transformer['optimized_attention_override'] = self.attention
        options['transformer_options'] = transformer
        model.model_options = options
        try:
            yield self
            if self.sol_calls == 0 or self.first_blocks != self.steps:
                raise RuntimeError('Sol-Attn did not execute the qualified sampling path')
        finally:
            model.model_options = saved
            self.layout = None


@contextmanager
def sampling(model, steps):
    if MODE == 'dense':
        yield None
    else:
        with SolRun(steps).patch(model) as run:
            yield run
