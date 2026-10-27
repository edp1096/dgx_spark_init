# SPDX-License-Identifier: Apache-2.0
"""드라마3 의 16비트 남은 몫을 하다마드 회전 8비트(W8A8)로 (2026-10-20, 사용자 「하다마드 회전을 적용시켜봐」 → 「16비트 부분 → 회전 8비트」).

int4 판(drama-3-int4)은 라우티드 전문가만 4비트고 나머지 4.8GB 가 bf16 이다. 그중 셋을 회전 int8 로 바꿔 약 1.9GB 를 덜어낸다 —
3090 24GB 에 무게(20.8GB)를 올릴 자리를 만들려는 것.
  · lm_head [252288, 2048] · 백본 embed_tokens [252288, 2048] · 소리 디코더(코드북 탑)의 nn.Linear 전부(qkv·o·gate_up·down·eh_proj·코드북 머리).
  · 손대지 않는 것: 라우티드 전문가(이미 int4) · 공유 전문가(0.25GB, SGLang MoE 길 안이라 덜어낼 몫에 비해 품이 크다) ·
    디코더의 작은 임베딩(embedding_bag 셈, 0.39GB) · norm · 라우터.
셈은 미니맥스 H3 int8 판과 같다(`~/바탕화면/오디오북제작기/엔진/drama/h3_int8.py` — 3090 에서 돌려 본 셈판):
  줄마다 256칸씩 하다마드 회전(H256 = H16 ⊗ H16) 뒤 줄 비율 하나로 int8. 들어오는 활성값도 같은 회전을 걸어 int8 로 바꿔 int8 × int8 → int32.
  회전은 직교라 x·Wᵀ = (xR)(WR)ᵀ. 임베딩은 곱이 아니라 줄 찾기라, 찾은 줄을 풀고 같은 회전으로 되돌린다(R 은 대칭·직교 → 되돌림 = 같은 회전).
무게는 체크포인트의 bf16 을 **읽는 자리에서** GPU 로 한 장씩 올려 회전·양자화한다 — 16비트 사본을 통째로 카드에 올리지 않는다.
켜기: 환경변수 DRAMA3_HADAMARD8=1 (오디오북제작기 엔진/worker.py 가 서버를 띄울 때 건다). 끄면 원래 길 그대로.
🔴 이 파일과 sglang_model.py·audio_decoder.py 의 「하다마드8」 줄은 업스트림 밖 패치다 — `~/패치기록.md`. git pull 로 sglang-omni 를 새로 받으면 다시 넣을 것.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import nn


def 켜짐() -> bool:
    return os.environ.get("DRAMA3_HADAMARD8", "0") == "1"


# ── 셈판: FreeVideo freevideo_engine/int8_ops.py (Copyright FlashML-org, Apache License 2.0) 에서 옮김 — h3_int8.py 와 같은 글 ─────
GROUP = 256
_H16: dict[str, torch.Tensor] = {}


def hadamard16(device):
    """H4 (x) H4 / 4 with H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] (FP32)."""
    key = str(device)
    if key not in _H16:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        _H16[key] = (torch.kron(h4, h4) / 4).to(device).contiguous()
    return _H16[key]


@triton.jit
def _rotate(x, h16, BM: tl.constexpr):
    # Channel c = 16 a + b of a 256-wide group: y = H16 X H16 over (a, b).
    z = tl.dot(tl.reshape(x, (BM * 16, 16)), h16, input_precision='ieee')
    z = tl.permute(tl.reshape(z, (BM, 16, 16)), (0, 2, 1))
    z = tl.dot(tl.reshape(z, (BM * 16, 16)), h16, input_precision='ieee')
    return tl.reshape(tl.permute(tl.reshape(z, (BM, 16, 16)), (0, 2, 1)), (BM, 256))


@triton.jit
def _rotate_quantize(X, Q, S, H, M, K: tl.constexpr, SX, BM: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    live = rows < M
    lane = tl.arange(0, 256)
    pair = tl.arange(0, 16)
    h16 = tl.load(H + pair[:, None] * 16 + pair[None, :])
    largest = tl.zeros((BM,), tl.float32)
    for group in range(K // 256):
        x = tl.load(X + rows[:, None] * SX + group * 256 + lane[None, :], live[:, None], other=0.).to(tl.float32)
        largest = tl.maximum(largest, tl.max(tl.abs(_rotate(x, h16, BM)), 1))
    scale = tl.maximum(largest, 1e-12) / 127.
    tl.store(S + rows, scale, live)
    for group in range(K // 256):
        x = tl.load(X + rows[:, None] * SX + group * 256 + lane[None, :], live[:, None], other=0.).to(tl.float32)
        value = tl.extra.cuda.libdevice.rint(tl.fdiv(_rotate(x, h16, BM), scale[:, None], ieee_rounding=True))
        value = tl.minimum(tl.maximum(value, -127.), 127.)
        tl.store(Q + rows[:, None] * K + group * 256 + lane[None, :], value.to(tl.int8), live[:, None])


def rotate_quantize(value):
    """[..., K] BF16/FP16 -> (int8 [M, K], FP32 [M] row scales) after the ConvRot rotation."""
    rows = value.reshape(-1, value.shape[-1])
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    m, k = rows.shape
    if k % GROUP:
        raise ValueError('ConvRot activations need a multiple of 256 channels')
    quantized = torch.empty((m, k), device=rows.device, dtype=torch.int8)
    scales = torch.empty(m, device=rows.device, dtype=torch.float32)
    if m:
        block = 16
        _rotate_quantize[(triton.cdiv(m, block),)](rows, quantized, scales, hadamard16(rows.device), m, k, rows.stride(0),
                                                   BM=block, num_warps=4)
    return quantized, scales


@triton.autotune(configs=[
    triton.Config({'BM': 128, 'BN': 128, 'BK': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BM': 256, 'BN': 128, 'BK': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
], key=['ROWS', 'N', 'K'], cache_results=True)  # tuned once per machine, kept in the Triton cache
@triton.jit
def _matmul(A, W, SA, SW, B, O, M, N, K: tl.constexpr, SAM, SWN, SOM, ROWS,
            HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    blocks_m, blocks_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    per_group = GROUP_M * blocks_n
    first_m = (pid // per_group) * GROUP_M
    size_m = tl.minimum(blocks_m - first_m, GROUP_M)
    pid_m = first_m + (pid % per_group) % size_m
    pid_n = (pid % per_group) // size_m
    rows = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    cols = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    a_ptrs = A + rows[:, None] * SAM + inner[None, :]
    w_ptrs = W + cols[None, :] * SWN + inner[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(K // BK):
        a = tl.load(a_ptrs, rows[:, None] < M, other=0)
        w = tl.load(w_ptrs, cols[None, :] < N, other=0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
        a_ptrs += BK
        w_ptrs += BK
    sa = tl.load(SA + rows, rows < M, other=0.)
    sw = tl.load(SW + cols, cols < N, other=0.)
    out = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    if HAS_BIAS:
        out += tl.load(B + cols, cols < N, other=0.).to(tl.float32)[None, :]
    tl.store(O + rows[:, None] * SOM + cols[None, :], out.to(O.dtype.element_ty),
             (rows[:, None] < M) & (cols[None, :] < N))


def matmul(quantized, weight, weight_scale, bias=None, out_dtype=torch.bfloat16):
    """(int8 [M, K], FP32 [M]) x int8 [N, K] rows with FP32 [N] scales -> out_dtype [M, N]."""
    values, scales = quantized
    m, k = values.shape
    n, weight_k = weight.shape
    if k != weight_k or k % 128:
        raise ValueError('Int8 projection widths differ or are not multiples of 128')
    if weight.stride(1) != 1 or values.stride(1) != 1:
        raise ValueError('Int8 operands need unit column stride')
    weight_scale = weight_scale.reshape(-1)
    output = torch.empty((m, n), device=values.device, dtype=out_dtype)
    if not m:
        return output
    # Tune once per power-of-two row band, not for every chunk and remainder row count.
    rows = max(256, 1 << (m - 1).bit_length())
    _matmul[lambda meta: (triton.cdiv(m, meta['BM']) * triton.cdiv(n, meta['BN']),)](
        values, weight, scales, weight_scale, bias if bias is not None else weight_scale, output,
        m, n, k, values.stride(0), weight.stride(0), output.stride(0), rows, bias is not None)
    return output
# ── 셈판 끝 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def 되돌리기(값: torch.Tensor) -> torch.Tensor:
    """회전된 줄 [..., K] fp32 → 원래 꼴 — 256칸 묶음마다 H16 X H16(_rotate 와 같은 셈, 되돌림 = 같은 회전)."""
    h = hadamard16(값.device)
    앞, k = 값.shape[:-1], 값.shape[-1]
    return (h @ 값.reshape(*앞, k // 256, 16, 16) @ h).reshape(*앞, k)


@torch.no_grad()
def 줄채우기(무게칸: torch.Tensor, 비율칸: torch.Tensor, 원본: torch.Tensor) -> None:
    """체크포인트 bf16 줄 [n, K](CPU 도 됨) → 회전 int8 을 무게칸 [n, K]·비율칸 [n] 에 — 큰 판은 줄 묶음으로 나눠 GPU 자리를 아낀다."""
    if tuple(원본.shape) != tuple(무게칸.shape):
        raise ValueError(f"하다마드8: 모양이 다르다 {tuple(원본.shape)} ≠ {tuple(무게칸.shape)}")
    for 첫 in range(0, 원본.shape[0], 32768):
        조각 = 원본[첫:첫 + 32768].to(무게칸.device, torch.bfloat16)
        q, s = rotate_quantize(조각)
        무게칸[첫:첫 + 32768].copy_(q)
        비율칸[첫:첫 + 32768].copy_(s)


class Hadamard8Linear(nn.Module):
    """nn.Linear(bias 없음) 자리 — 회전 int8 무게 [N, K] + 줄 비율 [N]. 결과 dtype = 입력 dtype."""

    def __init__(self, in_features: int, out_features: int, device) -> None:
        super().__init__()
        if in_features % GROUP:
            raise ValueError(f"하다마드8: 입력 폭 {in_features} 이 256 의 배수가 아니다")
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float32, device=device))
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = matmul(rotate_quantize(x), self.weight_int8, self.weight_scale, None, out_dtype=x.dtype)
        return y.reshape(*x.shape[:-1], self.out_features)


class Hadamard8Embedding(nn.Module):
    """VocabParallelEmbedding(tp 1) 자리 — 회전 int8 줄을 찾아 풀고 회전을 되돌린다."""

    def __init__(self, num_rows: int, dim: int, dtype: torch.dtype, device) -> None:
        super().__init__()
        self.dim, self.dtype = dim, dtype
        self.register_buffer("weight_int8", torch.zeros(num_rows, dim, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.zeros(num_rows, dtype=torch.float32, device=device))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        값 = self.weight_int8[ids].float() * self.weight_scale[ids].unsqueeze(-1)
        return 되돌리기(값).to(self.dtype)


class Hadamard8LMHeadMethod:
    """LogitsProcessor 가 부르는 lm_head 셈 — `lm_head.weight` 는 회전 int8, `lm_head.weight_scale` 는 줄 비율."""

    def process_weights_after_loading(self, layer) -> None:
        """로더가 모든 quant_method 에 부른다 — 무게는 읽을 때 이미 회전 int8 로 채웠다."""

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        rows = x.reshape(-1, x.shape[-1])
        y = matmul(rotate_quantize(rows), layer.weight, layer.weight_scale, bias, out_dtype=x.dtype)
        return y.reshape(*x.shape[:-1], layer.weight.shape[0])


# ── 갈아 끼우기 · 읽기 ────────────────────────────────────────────────────────────────────────────────────────────────────────

def 갈아끼우기(model: nn.Module, 장치) -> None:
    """Drama3ForCausalLM 을 지은 바로 뒤 — lm_head(meta 에 지음)·탑(CPU 에 지음)을 회전 int8 자리로 카드에 (무게는 아직 비어 있다).
    embed_tokens 는 Drama3Backbone._build_embed_tokens 가 처음부터 Hadamard8Embedding 으로 짓는다."""
    줄수, 폭 = model.lm_head.weight.shape
    model.lm_head.weight = nn.Parameter(torch.zeros(줄수, 폭, dtype=torch.int8, device=장치), requires_grad=False)
    model.lm_head.register_buffer("weight_scale", torch.zeros(줄수, dtype=torch.float32, device=장치))
    model.lm_head.quant_method = Hadamard8LMHeadMethod()

    for 이름, 칸 in list(model.audio_decoder.named_modules()):
        if type(칸) is nn.Linear and 칸.bias is None:
            앞, _, 끝 = 이름.rpartition(".")
            어미 = model.audio_decoder.get_submodule(앞) if 앞 else model.audio_decoder
            setattr(어미, 끝, Hadamard8Linear(칸.in_features, 칸.out_features, 장치))
    model.audio_decoder.to(장치)        # 남은 작은 임베딩·norm·버퍼(0.39GB)만 bf16 으로 카드에
    model.audio_decoder.hadamard8 = True


def 양자화칸(모듈: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """그 모듈의 (int8 무게칸, 비율칸)."""
    if isinstance(모듈, (Hadamard8Linear, Hadamard8Embedding)):
        return 모듈.weight_int8, 모듈.weight_scale
    return 모듈.weight, 모듈.weight_scale          # lm_head


def 임베딩채우기(칸: Hadamard8Embedding, 원본: torch.Tensor) -> None:
    """패딩된 줄(어휘 넘는 줄)은 0 그대로 둔다."""
    무게, 비율 = 양자화칸(칸)
    n = 원본.shape[0]
    줄채우기(무게[:n], 비율[:n], 원본)
