"""Compact empty EXL3 expert tiles without changing TensorFold's arithmetic.

Source: TensorFold db281878ddb836fd0df510d8771ecb7e0fe47d26 (Apache-2.0).
This local adapter retains the upstream grouping and grouped CUDA kernels.
"""
import torch
import triton
import triton.language as tl

OPTIMIZATION = 'expert-tile-compaction-v1'
MIN_ROWS = 128


@triton.jit
def _compact(IDS, COUNT, MEM, OUT_IDS, OUT_COUNT, OUT_MEM,
             R: tl.constexpr, TILES: tl.constexpr):
    u = tl.program_id(0)
    if u < tl.load(COUNT):
        tile = tl.arange(0, TILES)
        first = tl.load(MEM + u * R + tile * 16, tile * 16 < R, other=-1)
        live = first >= 0
        count = tl.sum(live.to(tl.int32), 0)
        base = tl.atomic_add(OUT_COUNT, count)
        rank = tl.cumsum(live.to(tl.int32), 0) - 1
        expert = tl.load(IDS + u)
        tl.store(OUT_IDS + base + rank, expert, live)
        lane = tl.arange(0, 16)
        source = tile[:, None] * 16 + lane[None, :]
        value = tl.load(MEM + u * R + source,
                        (source < R) & live[:, None], other=-1)
        tl.store(OUT_MEM + (base + rank[:, None]) * 16 + lane[None, :],
                 value, live[:, None])


class _ScratchView:
    def __init__(self, scratch, ids, count, members):
        self.__dict__.update(scratch.__dict__)
        self.compact_ids = ids
        self.count = count
        self.compact_members = members

    def window(self, rows):
        return self.compact_ids, self.compact_members


def install():
    from tensorfold.cuda.exl3 import experts as expert_module
    original = expert_module.routed
    if getattr(original, '_sparktalk_compact', False):
        return original._sparktalk_original

    def routed(x, pick, wts, ex, s, out, R, limit=float('inf'),
               act_mode=expert_module.ACT_F32, group=True):
        if R < MIN_ROWS or not group:
            return original(x, pick, wts, ex, s, out, R, limit, act_mode, group)
        ids, members = s.window(R)
        expert_module._ext().group(pick, ids, s.count, members, R, s.slots, ex.count)
        # Sum ceil(n_expert/16) <= ceil(total_members/16) + expert_count.
        # No CPU readback is needed to size the worklist or launch its kernels.
        bound = min(ids.numel() * triton.cdiv(R, 16),
                    triton.cdiv(R * s.slots, 16) + ids.numel())
        if not hasattr(s, '_compact_ids') or s._compact_ids.numel() < bound:
            s._compact_ids = torch.empty(bound, device=x.device, dtype=torch.int32)
            s._compact_count = torch.empty(1, device=x.device, dtype=torch.int32)
            s._compact_members = torch.empty((bound, 16), device=x.device, dtype=torch.int32)
        compact_ids = s._compact_ids[:bound]
        compact_members = s._compact_members[:bound]
        compact_count = s._compact_count
        compact_count.zero_()
        _compact[(ids.numel(),)](
            ids, s.count, members, compact_ids, compact_count, compact_members,
            R, triton.next_power_of_2(triton.cdiv(R, 16)), num_warps=4)
        view = _ScratchView(s, compact_ids, compact_count, compact_members)
        # Keep each original 16-member tile intact, including the member order.
        # Worklist order can vary; all tiles write disjoint output addresses.
        return original(x, pick, wts, ex, view, out, R, limit, act_mode, False)

    routed._sparktalk_compact = True
    routed._sparktalk_original = original
    expert_module.routed = routed
    return original
