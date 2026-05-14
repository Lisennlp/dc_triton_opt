"""Pure attention with head-serial loop — same structure as V3 but NO DC mixing.

Grid = (num_q_chunks, B * G)
Each block loops HPG heads: QK → softmax → PV → store OUT.

No s_acc, no a_acc, no pre/post mixing. Just attention.
This is the baseline for measuring DC mixing overhead.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['B', 'T', 'N', 'D', 'W', 'BM', 'KL', 'G', 'HPG'],
)
@triton.jit
def _attn_head_serial_kernel(
    Q, K, V, OUT, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_ob, stride_ot, stride_on, stride_od,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_c * BM
    head_start = g * HPG

    q_offs = q_start + tl.arange(0, BM)
    d_offs = tl.arange(0, D)
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))
    causal = k_offs[None, :] <= q_offs[:, None]
    win = (q_offs[:, None] - k_offs[None, :]) < W
    valid = causal & win & q_mask[:, None] & k_mask[None, :]

    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)

        # QK
        q_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                  + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q_blk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling
        qk = tl.where(valid, qk, float("-inf"))

        # Softmax (exp2)
        rm = tl.max(qk, axis=1)
        p = tl.exp2((qk - rm[:, None]) * LOG2E)
        p = tl.where(valid, p, 0.0)
        rs = tl.sum(p, axis=1)
        sl = tl.where(rs > 0.0, rs, 1.0)
        probs = p / sl[:, None]

        # PV
        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        pv = tl.dot(probs.to(v_blk.dtype), v_blk)

        # Store
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, pv, mask=q_mask[:, None])


class AttnHeadSerial:
    """Pure attention, head-serial. Same grid structure as DC V3."""

    @staticmethod
    def forward(q, k, v, scaling, window, seq_lens=None, G=1, chunk_size=16):
        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G
        assert N % G == 0

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)

        out = torch.empty_like(q)
        num_chunks = triton.cdiv(T, chunk_size)
        grid = (num_chunks, B * G)

        _attn_head_serial_kernel[grid](
            q, k, v, out, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL,
            G=G, HPG=HPG,
        )
        return out
