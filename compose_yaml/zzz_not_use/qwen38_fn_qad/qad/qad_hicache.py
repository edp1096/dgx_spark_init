"""Experimental TP1/file HiCache payloads for QSA and PLE.

Only runs when HiCache is constructed; ordinary inference is unchanged.
The existing token-prefix lookup remains authoritative. No transcript rewriting.
"""
from __future__ import annotations

import hashlib
import json
import os
import logging
import zlib
from dataclasses import dataclass

import torch


ENVELOPE_BYTES = 44  # magic + schema SHA256 + payload CRC32


@dataclass
class ExtraTensor:
    name: str
    tensor: torch.Tensor  # slot is always axis zero (views are allowed)
    layer: int | None = None

    @property
    def row_bytes(self):
        return self.tensor[0].numel() * self.tensor.element_size()


def ple_tensors(pool):
    result = []
    for i, sibling in enumerate(getattr(pool, '_slot_siblings', ())):
        if hasattr(sibling, 'conv_state'):
            tensor = sibling.conv_state
            if tensor is not None:
                result.append(ExtraTensor(f'ple-conv-{i}', tensor.movedim(1, 0)))
        elif hasattr(sibling, 'context'):
            tensor = sibling.context
            if tensor is not None:
                result.append(ExtraTensor(f'ple-ngram-{i}', tensor))
        else:
            raise ValueError(f'Unregistered QAD slot state: {type(sibling).__name__}')
    return result


def qsa_tensors(pool, drafts=()):
    result = []
    target_layers = pool.full_kv_pool.layer_num
    for depth, source in enumerate((pool, *drafts)):
        buffers = getattr(source, 'qsa_compressed_k_buffer_pool', ())
        if not buffers:
            raise ValueError('QAD HiCache requires compressed QSA state for target and draft')
        if source.qsa_compress_ratio != pool.qsa_compress_ratio:
            raise ValueError('QAD target/draft QSA compression differs')
        if depth and len(buffers) != 1:
            raise ValueError('QAD HiCache supports one QSA layer per MTP draft pool')
        for layer, tensor in enumerate(buffers):
            host_layer = layer if depth == 0 else target_layers + depth - 1
            result.append(ExtraTensor(f'qsa-{depth}-{layer}', tensor, host_layer))
    return result


class CompleteHostPool:
    """Append dependent tensors to the same file page as their anchor state.

    Copies deliberately use synchronous Torch transfers for the first implementation:
    publication of a HiCache completion event must never precede the extra state.
    Native KV kernels are still used by the wrapped pool.
    """
    def __init__(self, base, extras, *, compression=1, label='qad'):
        self.base = base
        self.extras = tuple(extras)
        self.compression = compression
        self.page_size = base.page_size
        if self.page_size % compression:
            raise ValueError('QAD cache pages must end at a complete QSA group')
        self.rows_per_page = self.page_size // compression
        template = base.get_dummy_flat_data_page()
        self.base_page_bytes = template.numel() * template.element_size()
        schema = [label, self.page_size, compression, self.base_page_bytes,
                  [(e.name, list(e.tensor.shape[1:]), str(e.tensor.dtype), e.layer) for e in extras]]
        signature = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).digest()
        self.header = torch.tensor(list(b'QADHC002' + signature), dtype=torch.uint8)
        self.page_bytes = self.base_page_bytes + ENVELOPE_BYTES + sum(e.row_bytes for e in extras) * self.rows_per_page
        self.size_per_token = self.page_bytes / self.page_size
        self.can_use_write_back_jit = getattr(base, "can_use_write_back_jit", False)
        self.buffers = [torch.empty((base.size // compression, *e.tensor.shape[1:]),
                                    dtype=e.tensor.dtype, device='cpu', pin_memory=base.pin_memory)
                        for e in extras]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_size_per_token(self):
        return self.size_per_token

    def get_ksize_per_token(self):
        return self.size_per_token

    def _rows(self, indices):
        # HiCache transfers complete, potentially non-contiguous pages. Validate
        # page order before compression; never silently alias distinct token slots.
        ids = indices.to(device='cpu', dtype=torch.long).reshape(-1)
        if ids.numel() % self.page_size:
            raise ValueError('Partial QAD HiCache page')
        pages = ids.reshape(-1, self.page_size)
        if pages.numel() and (torch.any(pages[:, 0] % self.page_size) or
            not torch.equal(pages, pages[:, :1] + torch.arange(self.page_size))):
            raise ValueError('QAD HiCache indices must contain aligned contiguous pages')
        return ids[::self.compression] // self.compression

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        hi, di = self._rows(host_indices), self._rows(device_indices)
        if hi.numel() != di.numel():
            raise ValueError('QAD HiCache transfer lengths differ')
        self.base.backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
        for extra, host in zip(self.extras, self.buffers):
            values = extra.tensor.index_select(0, di.to(extra.tensor.device)).to('cpu')
            host.index_copy_(0, hi, values)

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices,
                                 layer_id, io_backend='kernel', *, is_draft=False):
        hi, di = self._rows(host_indices), self._rows(device_indices)
        if hi.numel() != di.numel():
            raise ValueError('QAD HiCache transfer lengths differ')
        self.base.load_to_device_per_layer(device_pool, host_indices, device_indices,
                                          layer_id, io_backend, is_draft=is_draft)
        for extra, host in zip(self.extras, self.buffers):
            # PLE is slot-level. Restore it with the first Mamba layer, before
            # that layer's completion is published. QSA follows its KV layer.
            if extra.layer != layer_id and not (extra.layer is None and layer_id == 0 and not is_draft):
                continue
            extra.tensor.index_copy_(0, di.to(extra.tensor.device),
                                     host.index_select(0, hi).to(extra.tensor.device))

    def _slice(self, index):
        if index < 0 or index % self.page_size or index + self.page_size > self.base.size:
            raise ValueError('Invalid QAD host page offset')
        start = index // self.compression
        return slice(start, start + self.rows_per_page)

    def get_data_page(self, index, flat=True):
        sl = self._slice(index)
        raw = torch.cat([self.header, self.base.get_data_page(index, flat=True).contiguous().view(torch.uint8).reshape(-1),
                          *[b[sl].contiguous().view(torch.uint8).reshape(-1) for b in self.buffers]])
        checksum = zlib.crc32(memoryview(raw.numpy())).to_bytes(4, 'little')
        return torch.cat([raw, torch.tensor(list(checksum), dtype=torch.uint8)])

    def get_dummy_flat_data_page(self):
        return torch.empty(self.page_bytes, dtype=torch.uint8, pin_memory=self.base.pin_memory)

    def set_from_flat_data_page(self, index, data_page):
        sl = self._slice(index)
        raw = data_page.contiguous().view(torch.uint8).reshape(-1)
        # Validate the entire envelope before mutating any live host cache state.
        if raw.numel() != self.page_bytes or not torch.equal(raw[:len(self.header)], self.header):
            raise ValueError('QAD HiCache payload size/layout version mismatch')
        expected_crc = int.from_bytes(bytes(raw[-4:].tolist()), 'little')
        if zlib.crc32(memoryview(raw[:-4].numpy())) != expected_crc:
            raise ValueError('QAD HiCache payload checksum mismatch')
        pos = len(self.header)
        template = self.base.get_dummy_flat_data_page()
        payload = raw[pos:pos+self.base_page_bytes].view(template.dtype).reshape(template.shape)
        self.base.set_from_flat_data_page(index, payload)
        pos += self.base_page_bytes
        for buffer in self.buffers:
            target = buffer[sl]
            size = target.numel() * target.element_size()
            target.copy_(raw[pos:pos+size].view(target.dtype).reshape(target.shape))
            pos += size

    def is_stride_page_aligned(self, page_size_bytes=4096):
        return False

    def get_page_buffer_meta(self, indices):
        raise NotImplementedError('QAD payload wrapper currently supports the file backend only')

    def destroy(self):
        self.buffers.clear()
        self.base.destroy()


def prepare(pool, params, server_args, storage_backend):
    """Called only from the hybrid Mamba HiCache constructor."""
    extras = ple_tensors(params.req_to_token_pool.mamba_pool)
    has_qsa = hasattr(pool, 'qsa_compressed_k_buffer_pool')
    if not has_qsa and not extras:
        return None
    if os.environ.get('SGLANG_QAD_HICACHE') != '1':
        raise ValueError('QSA/PLE HiCache requires experimental SGLANG_QAD_HICACHE=1')
    if storage_backend != 'file' or server_args.hicache_mem_layout != 'page_first':
        raise ValueError('QAD HiCache currently requires file storage and page_first layout')
    if server_args.tp_size != 1 or params.pp_size != 1 or params.attn_cp_size != 1:
        raise ValueError('QAD HiCache currently supports TP1/PP1/CP1 only')
    if server_args.hicache_size <= 0:
        raise ValueError('Set an explicit small --hicache-size for QAD unified memory')
    # PLE layer 2 in the qualified QAD model must not run before state restoration.
    if extras and min(params.req_to_token_pool.mamba_map) != 0:
        raise ValueError('QAD PLE restoration requires a Mamba layer at layer zero')
    mamba_pool = params.req_to_token_pool.mamba_pool
    if getattr(mamba_pool, 'enable_linear_replayssm', False) or getattr(mamba_pool, 'enable_linear_replayssm_spec', False):
        raise ValueError('ReplaySSM state is not supported by this QAD HiCache payload')
    drafts = params.mtp_draft_device_pools
    for draft in drafts:
        # EAGLE/MTP reuses the target request pool, including its PLE siblings.
        # Those slots are already serialized once by the target Mamba payload.
        if draft.mamba_pool is not mamba_pool and ple_tensors(draft.mamba_pool):
            raise ValueError('Independent MTP PLE state is not yet supported by QAD HiCache')
    qsa = qsa_tensors(pool, drafts) if has_qsa else []
    ratio = pool.qsa_compress_ratio if has_qsa else 1
    if params.page_size % ratio:
        raise ValueError('QAD page size must be divisible by QSA compression')
    return qsa, extras, ratio


def budgets(plan, kv_pool, mamba_pool, total_gb, page_size, drafts):
    """Fixed total budget including extra payloads and MTP layers (decimal GB)."""
    qsa, ple, ratio = plan
    kv_row = sum(t[0].numel()*t.element_size() for t in (*kv_pool.k_buffer, *kv_pool.v_buffer))
    for draft in drafts:
        kv_row += sum(t[0].numel()*t.element_size() for t in (*draft.full_kv_pool.k_buffer, *draft.full_kv_pool.v_buffer))
    state = mamba_pool.mamba_cache
    mamba_row = sum(t[:, 0].numel()*t.element_size() for t in (*state.conv, state.temporal))
    full_kv = kv_row + sum(e.row_bytes for e in qsa)/ratio + ENVELOPE_BYTES/page_size
    full_mamba = mamba_row + sum(e.row_bytes for e in ple) + ENVELOPE_BYTES
    kv_weight, mamba_weight = full_kv*kv_pool.size, full_mamba*mamba_pool.size
    # A single host Mamba slot can be pinned by the warmup/session checkpoint,
    # starving later write-through and disk prefetch. Reserve at least two
    # complete states while staying inside the requested total budget.
    total = total_gb * 1e9
    mamba_slots = max(2, int(total * mamba_weight / (kv_weight + mamba_weight) / full_mamba))
    kv_available = total - mamba_slots * full_mamba - full_kv * page_size
    if kv_available < full_kv * page_size:
        raise ValueError('QAD HiCache budget is too small for two states and KV pages')
    # Native allocators round up by one page/slot. Half a native state avoids
    # floating-point boundary errors and produces exactly mamba_slots slots.
    return kv_available * (kv_row / full_kv) / 1e9, (mamba_slots - 0.5) * mamba_row / 1e9


def read_file_page(backend, pool_name, key, host_pool, page_offset):
    """Bad/incomplete cache files are misses, never fatal to the IO worker.

    Validate before publishing host state. Remove the observed invalid file so
    the native atomic writer can replace it on recomputation. This uses the
    backend's existing single-process file-cache ownership model.
    """
    suffixed = backend._get_suffixed_key(backend._log_key(pool_name, key))
    path = os.path.join(backend.file_path, suffixed + '.bin')
    observed = None
    try:
        page = host_pool.get_dummy_flat_data_page()
        with open(path, 'rb', buffering=0) as stream:
            observed = os.fstat(stream.fileno())
            if observed.st_size != page.numel():
                raise ValueError('QAD HiCache file size mismatch')
            if stream.readinto(memoryview(page.numpy())) != page.numel():
                raise ValueError('QAD HiCache incomplete file read')
        host_pool.set_from_flat_data_page(page_offset, page)
        backend._evictor.touch(suffixed, path)
        if backend.metadata_cache is not None:
            backend.metadata_cache.add(suffixed)
        return True
    except (OSError, ValueError) as error:
        if observed is not None:
            try:
                current = os.stat(path)
                if (current.st_ino, current.st_size, current.st_mtime_ns) == (
                    observed.st_ino, observed.st_size, observed.st_mtime_ns
                ):
                    os.unlink(path)
                    backend._evictor.abort(suffixed)
            except OSError:
                pass
        if backend.metadata_cache is not None:
            backend.metadata_cache.remove(suffixed)
        logging.getLogger(__name__).warning('QAD cache page rejected (%s): %s', pool_name, error)
        return False


def guarded_page_get(controller, operation, hash_values, host_indices):
    """Cover the native generic KV IO path as well as v2 auxiliary-pool IO."""
    host = controller.mem_pool_host
    if hasattr(host, 'anchor_entry'):
        host = host.anchor_entry.host_pool
    if not isinstance(host, CompleteHostPool):
        return False
    from sglang.srt.mem_cache.hicache_storage import PoolName
    for i, key in enumerate(hash_values):
        if not read_file_page(controller.storage_backend, PoolName.KV, key, host,
                              int(host_indices[i * controller.page_size])):
            break
        if not operation.increment(controller.page_size):
            break
    return True


def admission_error(server_args, input_tokens, max_new_tokens, capacity):
    """Reject requests that can never pass the pinned scheduler's KV gate.

    This is a necessary bound, using the actual worker capacity, after SGLang
    normalizes/clips the output budget. In particular, clipping output to zero
    must not leave an unschedulable request spinning in the waiting queue.
    """
    if (os.environ.get('SGLANG_QAD_HICACHE') != '1' or
            not getattr(server_args, 'enable_hierarchical_cache', False)):
        return None
    clip = int(os.environ.get('SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION', '4096'))
    output_reserve = min(max(0, max_new_tokens), clip)
    page_reserve = server_args.page_size
    if input_tokens + output_reserve + page_reserve >= capacity:
        limit = max(0, capacity - output_reserve - page_reserve - 1)
        # The native output normalizer uses ceil_page(input), so the hint
        # must leave room after rounding instead of suggesting another zero-output request.
        limit -= limit % page_reserve
        return (
            f'Request cannot fit the KV admission budget: {input_tokens} input tokens, '
            f'{output_reserve} output-reserve tokens and {page_reserve} page-reserve '
            f'tokens require less than {capacity} total tokens. '
            f'Reduce the input to at most {limit} tokens for this output budget.'
        )
    return None
