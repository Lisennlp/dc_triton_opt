"""Experimental single-kernel DC v5 for HPG=4.

This is intentionally narrower than v4:
- only the useful HPG=4 / KL<=128 path is specialized here;
- the four QK matrices are cached, optionally as fp16;
- valid attention mask is computed once and reused by all four heads;
- a_acc can optionally be kept in fp16 to reduce live register pressure.

Unsupported shapes fall back to v4 so callers can use it as a drop-in
experiment.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _load_qk(
    Q, K,
    b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    scaling,
):
    q = tl.load(
        Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
        + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )
    k = tl.load(
        K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
        + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
        mask=k_mask[:, None],
        other=0.0,
    )
    return tl.dot(q, tl.trans(k)) * scaling


@triton.jit
def _consume_qk_hpg4(
    qk, s_acc, a_acc, valid,
    b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
    PRE_W2, PRE_DD, POST_W1, POST_DD,
    V, OUT,
    stride_wb, stride_wt, stride_wn,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_ob, stride_ot, stride_on, stride_od,
    A_ACC_HALF: tl.constexpr,
):
    qk_f = qk.to(tl.float32)
    pw2 = tl.load(
        PRE_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    pdd = tl.load(
        PRE_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    score = (pdd + 1.0)[:, None] * qk_f + pw2[:, None] * s_acc
    score = tl.where(valid, score, float("-inf"))
    rm = tl.max(score, axis=1)
    p = tl.exp2((score - rm[:, None]) * LOG2E)
    p = tl.where(valid, p, 0.0)
    rs = tl.sum(p, axis=1)
    sl = tl.where(rs > 0.0, rs, 1.0)
    probs = p / sl[:, None]

    pw1p = tl.load(
        POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    if A_ACC_HALF:
        a_acc = (a_acc + (pw1p[:, None] * probs).to(tl.float16)).to(tl.float16)
    else:
        a_acc += pw1p[:, None] * probs

    v_ptrs = (
        V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
        + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd
    )
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    pv = tl.dot(probs.to(v_blk.dtype), v_blk)

    pddp = tl.load(
        POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    o_ptrs = (
        OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
        + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od
    )
    tl.store(o_ptrs, ((pddp[:, None] + 1.0) * pv), mask=q_mask[:, None])
    return a_acc


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=[
        "B", "T", "N", "D", "W", "BM", "KL", "G",
        "CACHE_QK_HALF", "A_ACC_HALF",
    ],
)
@triton.jit
def _dc_onekernel_hpg4(
    Q, K, V,
    PRE_W1, PRE_W2, PRE_DD,
    POST_W1, POST_W2, POST_DD,
    OUT, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr, G: tl.constexpr,
    CACHE_QK_HALF: tl.constexpr, A_ACC_HALF: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_c * BM
    head_start = g * 4

    q_offs = q_start + tl.arange(0, BM)
    d_offs = tl.arange(0, D)
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (
        (k_offs < T)
        & (k_offs >= 0)
        & (k_offs < seq_len)
        & (kl_offs < (BM + W - 1))
    )

    causal = k_offs[None, :] <= q_offs[:, None]
    win = (q_offs[:, None] - k_offs[None, :]) < W
    valid = causal & win & q_mask[:, None] & k_mask[None, :]

    n0 = head_start.to(tl.int64)
    n1 = (head_start + 1).to(tl.int64)
    n2 = (head_start + 2).to(tl.int64)
    n3 = (head_start + 3).to(tl.int64)

    s_acc = tl.zeros([BM, KL], dtype=tl.float32)

    qk0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_0 = tl.load(
        PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    s_acc += pw1_0[:, None] * qk0
    if CACHE_QK_HALF:
        qk0 = qk0.to(tl.float16)

    qk1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_1 = tl.load(
        PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    s_acc += pw1_1[:, None] * qk1
    if CACHE_QK_HALF:
        qk1 = qk1.to(tl.float16)

    qk2 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_2 = tl.load(
        PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n2 * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    s_acc += pw1_2[:, None] * qk2
    if CACHE_QK_HALF:
        qk2 = qk2.to(tl.float16)

    qk3 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_3 = tl.load(
        PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n3 * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    s_acc += pw1_3[:, None] * qk3
    if CACHE_QK_HALF:
        qk3 = qk3.to(tl.float16)

    if A_ACC_HALF:
        a_acc = tl.zeros([BM, KL], dtype=tl.float32).to(tl.float16)
    else:
        a_acc = tl.zeros([BM, KL], dtype=tl.float32)

    a_acc = _consume_qk_hpg4(
        qk0, s_acc, a_acc, valid,
        b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
        stride_wb, stride_wt, stride_wn,
        stride_vb, stride_vt, stride_vn, stride_vd,
        stride_ob, stride_ot, stride_on, stride_od,
        A_ACC_HALF,
    )
    a_acc = _consume_qk_hpg4(
        qk1, s_acc, a_acc, valid,
        b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
        stride_wb, stride_wt, stride_wn,
        stride_vb, stride_vt, stride_vn, stride_vd,
        stride_ob, stride_ot, stride_on, stride_od,
        A_ACC_HALF,
    )
    a_acc = _consume_qk_hpg4(
        qk2, s_acc, a_acc, valid,
        b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
        stride_wb, stride_wt, stride_wn,
        stride_vb, stride_vt, stride_vn, stride_vd,
        stride_ob, stride_ot, stride_on, stride_od,
        A_ACC_HALF,
    )
    a_acc = _consume_qk_hpg4(
        qk3, s_acc, a_acc, valid,
        b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
        stride_wb, stride_wt, stride_wn,
        stride_vb, stride_vt, stride_vn, stride_vd,
        stride_ob, stride_ot, stride_on, stride_od,
        A_ACC_HALF,
    )

    for h in range(4):
        ni = (head_start + h).to(tl.int64)
        v_ptrs = (
            V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
            + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd
        )
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        av = tl.dot(a_acc.to(v_blk.dtype), v_blk)
        pw2p = tl.load(
            POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        o_ptrs = (
            OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
            + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od
        )
        o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        tl.store(o_ptrs, (o_prev + pw2p[:, None] * av), mask=q_mask[:, None])


class TritonDCOneKernel:
    """V5 HPG=4 experiment; unsupported cases fall back to V4."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=8, chunk_size=16,
        cache_qk_half=True, a_acc_half=True,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        assert N % G == 0
        HPG = N // G

        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)
        if HPG != 4 or KL > 128:
            from triton_dc_onekernel_v4 import TritonDCOneKernel as V4

            return V4.forward(q, k, v, dc_weights, scaling, window, seq_lens, G=G, chunk_size=chunk_size)

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pre_w1 = pre_w1.contiguous()
        pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous()
        post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous()
        post_dd = post_dd.contiguous()

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)
        num_chunks = triton.cdiv(T, chunk_size)
        grid = (num_chunks, B * G)

        _dc_onekernel_hpg4[grid](
            q, k, v,
            pre_w1, pre_w2, pre_dd,
            post_w1, post_w2, post_dd,
            out, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G,
            CACHE_QK_HALF=bool(cache_qk_half), A_ACC_HALF=bool(a_acc_half),
        )
        return out
