
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════════════
# Direction 2 & 3: pre-mix in Q space
#
# q_agg[b,q,d] = Σ_s pw2_pre[s] * Q[b,q,s,d]
# score[b,j,q,k] = pw1_pre[j] * q_agg @ K[j]^T * scaling
#
# Direction 2: post in prob-space (original)
# Direction 3: post in output-space (combined with Direction 1)
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _k3_output_mix_kernel(
    ATTN_OUT, PW2_POST, PW1_POST, OUT,
    stride_ab, stride_at, stride_an, stride_ad,
    stride_pw2b, stride_pw2t, stride_pw2n,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_ob, stride_ot, stride_on, stride_od,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Output-space rank-1 post-mix.  Grid: (T/BM, B).

    out_agg[q,d] = Σ_j pw2_post[j] * attn_out[j,q,d]
    out[o,q,d]   = pw1_post[o] * out_agg[q,d]
    """
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    # Accumulate out_agg[BM, D] = Σ_j pw2_post[j] * attn_out[j, BM, D]
    out_agg = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for j in range(N):
        ji = j.to(tl.int64)
        pw2_ptrs = PW2_POST + b * stride_pw2b + m_offs.to(tl.int64) * stride_pw2t + ji * stride_pw2n
        pw2_j = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)

        a_ptrs = (ATTN_OUT + b * stride_ab
                  + m_offs[:, None].to(tl.int64) * stride_at
                  + ji * stride_an
                  + d_offs[None, :].to(tl.int64) * stride_ad)
        a_blk = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        out_agg += pw2_j[:, None] * a_blk

    # Broadcast: out[o,q,d] = pw1_post[o] * out_agg[q,d]
    for o in range(N):
        oi = o.to(tl.int64)
        pw1_ptrs = PW1_POST + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + oi * stride_pw1n
        pw1_o = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)

        o_ptrs = (OUT + b * stride_ob
                  + m_offs[:, None].to(tl.int64) * stride_ot
                  + oi * stride_on
                  + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, (pw1_o[:, None] * out_agg).to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _k0_qagg_kernel(
    Q, PW2_PRE, Q_AGG,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_pb, stride_pt, stride_pn,
    stride_ab, stride_at, stride_ad,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """q_agg[b,q,d] = Σ_s pw2_pre[s] * Q[b,q,s,d].  Grid: (T/BM, B)."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for s in range(N):
        si = s.to(tl.int64)
        pw2 = tl.load(PW2_PRE + b * stride_pb + m_offs.to(tl.int64) * stride_pt + si * stride_pn,
                       mask=m_mask, other=0.0).to(tl.float32)
        q_ptrs = (Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt
                  + si * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        acc += pw2[:, None] * q_blk

    a_ptrs = (Q_AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at
              + d_offs[None, :].to(tl.int64) * stride_ad)
    tl.store(a_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _k1_d3_perhead_qk_pv_kernel(
    Q_AGG, K, PW1_PRE, V, ATTN_OUT,
    stride_ab, stride_at, stride_ad,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_pb, stride_pt, stride_pn,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_ob, stride_ot, stride_on, stride_od,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Direction 3: per-head QK(q_agg) + online softmax + PV.  Grid: (T/BM, B*N)."""
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    q_ptrs = (Q_AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at
              + d_offs[None, :].to(tl.int64) * stride_ad)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    pw1_ptrs = PW1_PRE + b * stride_pb + m_offs.to(tl.int64) * stride_pt + n * stride_pn
    pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        logit = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        score = pw1[:, None] * logit
        score = tl.where(valid, score, float('-inf'))

        block_max = tl.max(score, axis=1)
        new_m = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(score - new_m[:, None])
        p = tl.where(valid, p, 0.0)
        new_l = l_i * alpha + tl.sum(p, axis=1)

        scale = tl.where(new_l > 0, (l_i * alpha) / new_l, 0.0)
        acc = acc * scale[:, None]
        p_norm = tl.where(new_l[:, None] > 0, p / new_l[:, None], 0.0)

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(p_norm.to(tl.float16), v_blk)

        m_i = new_m
        l_i = new_l

    o_ptrs = (ATTN_OUT + b * stride_ob + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])



def _launch_k0_qagg(q, pw2_pre, q_agg, block_m):
    B, T, N, D = q.shape
    grid = (triton.cdiv(T, block_m), B)
    _k0_qagg_kernel[grid](
        q, pw2_pre, q_agg,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        pw2_pre.stride(0), pw2_pre.stride(1), pw2_pre.stride(2),
        q_agg.stride(0), q_agg.stride(1), q_agg.stride(2),
        B=B, T=T, N=N, D=D, BLOCK_M=block_m,
        num_warps=8, num_stages=1,
    )


class TritonDCRank1PreQPostOut(object):
    """Direction 3: pre-mix in Q space, post-mix in output space.

    K0: q_agg = Σ_s pw2 * Q[s]                         grid(T/BM, B)   [Triton]
    K1: per-head QK(q_agg) + online softmax + PV        grid(T/BM, B*N) [Triton]
    K2: output-space mix                                grid(T/BM, B)   [Triton]
    """

    @staticmethod
    def forward(q, k, v, pre_w1, pre_w2, post_w1, post_w2,
                scaling, window, seq_lens, block_m=32, block_k=64):
        B, T, N, D = q.shape
        W = min(window, T)

        pw1_pre = pre_w1.squeeze(-1).contiguous()
        pw2_pre = pre_w2.squeeze(2).contiguous()
        pw1_post = post_w1.squeeze(-1).contiguous()
        pw2_post = post_w2.squeeze(2).contiguous()

        # K0: q_agg  (Triton)
        q_agg = torch.empty(B, T, D, device=q.device, dtype=torch.float16)
        _launch_k0_qagg(q, pw1_pre, q_agg, block_m)

        # K1: per-head QK + online softmax + PV  (Triton)
        attn_out = torch.empty(B, T, N, D, device=q.device, dtype=torch.float16)
        grid1 = (triton.cdiv(T, block_m), B * N)
        _k1_d3_perhead_qk_pv_kernel[grid1](
            q_agg, k, pw2_pre, v, attn_out,
            q_agg.stride(0), q_agg.stride(1), q_agg.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pw2_pre.stride(0), pw2_pre.stride(1), pw2_pre.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            scaling, B=B, T=T, N=N, D=D, W=W,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=2,
        )

        # K2: output-space mix  (Triton)
        out = torch.empty(B, T, N, D, device=q.device, dtype=torch.float16)
        grid2 = (triton.cdiv(T, block_m), B)
        _k3_output_mix_kernel[grid2](
            attn_out, pw1_post, pw2_post, out,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            pw1_post.stride(0), pw1_post.stride(1), pw1_post.stride(2),
            pw2_post.stride(0), pw2_post.stride(1), pw2_post.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, T=T, N=N, D=D, BLOCK_M=block_m,
            num_warps=2, num_stages=1,
        )
        return out