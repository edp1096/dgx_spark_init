"""Stable parallel expert grouping for TensorFold EXL3.

Source: TensorFold db281878ddb836fd0df510d8771ecb7e0fe47d26 (Apache-2.0).
This local adapter replaces only integer routing; the grouped arithmetic stays upstream.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _initialize(IDS, COUNT, MEM, R: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(MEM + i, -1, i < E * R)
    tl.store(IDS + i, i, i < E)
    if tl.program_id(0) == 0:
        tl.store(COUNT, E)

@triton.jit
def _members(PICK, MEM, R: tl.constexpr, SLOTS: tl.constexpr, B: tl.constexpr):
    expert = tl.program_id(0)
    i = tl.arange(0, B)
    value = tl.load(PICK + i, i < R * SLOTS, other=-1)
    match = (i < R * SLOTS) & (value == expert)
    rank = tl.cumsum(match.to(tl.int32), 0) - 1
    code = (i // SLOTS) * 32 + (i % SLOTS)
    tl.store(MEM + expert * R + rank, code, match & (rank < R))


def install():
    from tensorfold.cuda.exl3 import experts
    extension = experts._ext()
    original = extension.group
    if getattr(original, '_sparktalk_parallel', False):
        return

    def group(pick, uids, count, members, R, slots, E):
        valid = (R >= 128 and 0 < slots <= 32 and 0 < E <= 4096
                 and pick.numel() >= R * slots and count.numel() >= 1
                 and uids.numel() >= E and members.ndim == 2
                 and members.shape[0] >= E and members.shape[1] == R
                 and all(t.is_cuda and t.dtype == torch.int32 and t.is_contiguous()
                         for t in (pick, uids, count, members)))
        if not valid:
            return original(pick, uids, count, members, R, slots, E)
        _initialize[(triton.cdiv(E * R, 1024),)](uids, count, members, R, E, 1024)
        _members[(E,)](pick, members, R, slots,
                       triton.next_power_of_2(R * slots), num_warps=8)

    group._sparktalk_parallel = True
    extension.group = group
