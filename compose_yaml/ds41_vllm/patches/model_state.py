# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any
import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.final_decoder_rows = os.environ.get('DSV41_FINAL_DECODER_ROWS', '0') == '1'
        self.dense_output_requests: set[str] = set()
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )
        self.disk_stager=None
        if os.environ.get('DSV41_ENGRAM_PRESTAGE')=='1':
            from engram_stager import EngramDiskStager
            self.disk_stager=EngramDiskStager(vllm_config,model)

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        params = new_req_data.sampling_params
        if (params is None or params.prompt_logprobs is not None
                or new_req_data.mm_features or new_req_data.prompt_embeds is not None):
            self.dense_output_requests.add(new_req_data.req_id)

    def remove_request(self, req_id):
        super().remove_request(req_id)
        self.dense_output_requests.discard(req_id)

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if os.environ.get('DSV41_SHORT_CONTEXT_GRAPHS') == '1':
            import streaming_graphs
            streaming_graphs.runtime_context_upper_bound = int(
                input_batch.seq_lens_cpu_upper_bound.max()) if input_batch.num_reqs else 0
        if os.environ.get('DSV41_BENCH_CONTROL')=='1':
            import cache_control
            cache_control.update()
        model_inputs = super().prepare_inputs(input_batch, req_states)
        select_final = self.final_decoder_rows
        if os.environ.get('DSV41_BENCH_CONTROL') == '1':
            select_final = cache_control.final_decoder_rows
        # Only actual single-request text prefills. Decode, profiling, prompt
        # logprobs and dense-output consumers retain the original path.
        if (select_final and input_batch.num_reqs == 1 and input_batch.has_prefill
                and input_batch.num_tokens > 16
                and not self.dense_output_requests.intersection(input_batch.req_ids)
                and 0 < input_batch.logits_indices.numel() < input_batch.num_tokens):
            model_inputs['final_output_rows'] = input_batch.logits_indices
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        if self.disk_stager is not None and input_batch.input_ids is not None:
            self.disk_stager.stage(input_batch.input_ids,input_batch.positions,
                input_batch.query_start_loc[:input_batch.num_reqs+1],window,
                input_batch.num_tokens)
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        if self.disk_stager is not None: self.disk_stager.dummy(num_tokens)
        return model_inputs
