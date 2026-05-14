"""3-kernel DC residual: K0(BM=64) + K12f(BM=16, online softmax) + K3(BM=64).

Fuses K1+K2 from the 4-kernel baseline into a single K12f kernel using
FlashAttention-style online softmax.  Allows K0 to use independently
larger BM=64 for better compute intensity.

Eliminates m_buf/l_buf from HBM (stays in registers).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_dc_residual import _k0_preagg_only_kernel, _k3_final_kernel


@triton.jit
def _k12_fused_kernel(
    Q, K, V, S_BUF, PRE_W2, PRE_DD, POST_W1,
    O_BUF, A_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_sb, stride_st, stride_sw,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_at, stride_aw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Pass 1: online softmax + PV.  Pass 2: post_w1 atomic."""
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    pre_w2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_dd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                     mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = pre_dd + 1.0
    post_w1 = tl.load(POST_W1 + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + n * stride_pw1n,
                      mask=m_mask, other=0.0).to(tl.float32)

    q_ptrs = (Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T
    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    # ── Pass 1: online softmax + PV ──
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]
        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)
        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))

        block_max = tl.max(score, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        acc = acc * alpha[:, None]

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.float16), v_blk)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / safe_l[:, None]

    o_ptrs = (O_BUF + b * stride_ob + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])

    # ── Pass 2: post_w1 atomic ──
    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]
        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)
        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))
        probs = tl.exp(score - m_i[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)

        a_ptrs = (A_BUF + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at
                  + compact_k.to(tl.int64) * stride_aw)
        tl.atomic_add(a_ptrs, post_w1[:, None] * probs, mask=valid, sem="relaxed")


class TritonDCResidualFused:
    """3-kernel fused DC: K0 + K12f + K3. Independent BM per kernel."""

    @staticmethod
    def alloc_buffers(q, window):
        B, T, N, D = q.shape
        W = min(int(window), T)
        dev, dt = q.device, q.dtype
        return {
            "s_buf": torch.empty((B, T, W), device=dev, dtype=dt),
            "a_buf": torch.empty((B, T, W), device=dev, dtype=torch.float32),
            "o_buf": torch.empty((B, T, N, D), device=dev, dtype=dt),
            "out":   torch.empty((B, T, N, D), device=dev, dtype=dt),
        }

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None,
        bm_k0=64, bk_k0=64,
        bm_mid=16, bk_mid=32,
        bm_fin=64, bk_fin=32,
        buffers=None,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous(); pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous(); post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous(); post_dd = post_dd.contiguous()

        if buffers is None:
            buffers = TritonDCResidualFused.alloc_buffers(q, W)
        s_buf = buffers["s_buf"]
        a_buf = buffers["a_buf"]
        o_buf = buffers["o_buf"]
        out   = buffers["out"]
        a_buf.zero_()

        # K0: pre-agg (BM=64)
        _k0_preagg_only_kernel[(triton.cdiv(T, bm_k0), B)](
            q, k, pre_w1, s_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=bm_k0, BLOCK_K=bk_k0,
            num_warps=4, num_stages=2,
        )

        # K12f: fused online softmax + PV + post_w1 atomic
        _k12_fused_kernel[(triton.cdiv(T, bm_mid), B * N)](
            q, k, v, s_buf, pre_w2, pre_dd, post_w1,
            o_buf, a_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
            pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
            pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
            post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=bm_mid, BLOCK_K=bk_mid,
            num_warps=4, num_stages=2,
        )

        # K3: a_buf@V + final combine (BM=64)
        _k3_final_kernel[(triton.cdiv(T, bm_fin), B * N)](
            a_buf, v, o_buf, post_w2, post_dd, out, seq_lens,
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
            post_dd.stride(0), post_dd.stride(1), post_dd.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=bm_fin, BLOCK_K=bk_fin,
            num_warps=4, num_stages=2,
        )
        return out
