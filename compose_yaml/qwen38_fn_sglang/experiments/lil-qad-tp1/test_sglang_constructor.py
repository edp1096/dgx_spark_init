from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import torch
from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.models import qwen4_exp as q
from ple_embedding import NVFP4PLEEmbedding

with tempfile.TemporaryDirectory() as directory:
    cfg=Qwen4ExpTextConfig(vocab_size=128,eos_token_id=2,ngram_vocab_size_base=101,
        ple_embed_dim=2560,ple_embedding_dtype='nvfp4',split_ngram_parts=128,
        ple_offload_embedding=True,ple_offload_backend='file',ple_offload_dir=directory)
    with patch.object(q,'get_tp_group',return_value=SimpleNamespace(world_size=1)),patch.object(q,'is_dp_attention_enabled',return_value=False):
        emb=q.Qwen4ExpNGramEmbedding(cfg,2560)
        assert isinstance(emb.ngram_embedding,NVFP4PLEEmbedding)
        assert emb.ngram_embedding.packed.dtype==torch.uint8
        assert emb.ngram_embedding.packed.shape[1]==80
        assert emb.ngram_embedding.scales.shape[1]==10
        print('Patched SGLang NGram constructor selects packed storage without BF16 table expansion')
    with patch.object(q,'get_tp_group',return_value=SimpleNamespace(world_size=2)),patch.object(q,'is_dp_attention_enabled',return_value=False):
        try:q.Qwen4ExpNGramEmbedding(cfg,2560)
        except ValueError as e:assert 'TP1' in str(e)
        else:raise AssertionError('TP2 must be rejected by this TP1 trial')
    print('TP2 remains explicitly excluded from the trial path')
