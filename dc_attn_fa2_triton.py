"""DC residual attention built on top of FlashAttention-2 Triton kernel.

Architecture:
  K0:  pre-agg (same as before)        grid = (C, B)
  K_dc: FA2-style kernel with DC mods  grid = (C, B*N)   ← BLOCK_M=128
  K3:  a_buf@V + final combine         grid = (C, B*N)

The K_dc kernel is a modified FA2 _fwd_kernel that:
  1. Loads s_buf and pre-mixes: score = (1+pre_dd)*qk*scale + pre_w2*s_buf
  2. Uses FA2's online softmax + PV accumulation (unchanged)
  3. Adds a second pass for post_w1 * probs → atomic a_buf
  
With BLOCK_M=128 (vs our old BM=16), compute density is ~64× better.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from triton_dc_residual import _k0_preagg_only_kernel, _k3_final_kernel


@triton.jit
def _dc_fwd_kernel(
    Q, K, V,
    S_BUF,
    PRE_W2, PRE_DD, POST_W1,
    Out, A_BUF,
    Lse, TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_sb, stride_sm, stride_sw,
    stride_ob, stride_oh, stride_om,
    stride_ab, stride_am, stride_aw,
    stride_wb, stride_wm, stride_wh,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    window_size,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    # Load per-head DC weights for this query chunk
    w_base = off_b * stride_wb + off_h * stride_wh
    pre_w2_vals = tl.load(PRE_W2 + w_base + offs_m * stride_wm,
                          mask=offs_m < seqlen_q, other=0.0).to(tl.float32)
    pre_dd_vals = tl.load(PRE_DD + w_base + offs_m * stride_wm,
                          mask=offs_m < seqlen_q, other=0.0).to(tl.float32)
    direct_scale = pre_dd_vals + 1.0
    post_w1_vals = tl.load(POST_W1 + w_base + offs_m * stride_wm,
                           mask=offs_m < seqlen_q, other=0.0).to(tl.float32)

    # Compute window bounds for this query chunk
    k_lo_per_q = tl.maximum(0, offs_m - window_size + 1)

    # Initialize softmax state
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # ═══ Main loop: iterate over K tiles (window-restricted) ═══
    # Only iterate keys within the causal window: [max(0, q_start - W + 1), q_end)
    start_n_lo = start_m * BLOCK_M - window_size + 1
    if start_n_lo < 0:
        start_n_lo = 0
    # Align to BLOCK_N
    start_n_lo = (start_n_lo // BLOCK_N) * BLOCK_N
    end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(start_n_lo, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K tile
        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )

        # QK computation (no masking yet — masking applied to final score)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Valid mask
        valid_mask = (offs_m[:, None] >= (start_n + offs_n)[None, :]) & \
                     ((offs_m[:, None] - (start_n + offs_n)[None, :]) < window_size) & \
                     ((start_n + offs_n)[None, :] < seqlen_k) & \
                     (offs_m[:, None] < seqlen_q)

        # Compact key indexing for s_buf
        k_abs = start_n + offs_n
        compact_k = k_abs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), window_size - 1)

        s_ptrs = (S_BUF + off_b * stride_sb
                  + offs_m[:, None] * stride_sm
                  + compact_k * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid_mask, other=0.0).to(tl.float32)

        # DC pre-mixed score, then apply mask
        score = direct_scale[:, None] * qk * softmax_scale + pre_w2_vals[:, None] * s_agg
        score = tl.where(valid_mask, score, float("-inf"))

        # Online softmax (FA2 style, with NaN guard for all-masked tiles)
        m_ij = tl.maximum(tl.max(score, 1), lse_i)
        p = tl.exp(score - m_ij[:, None])
        p = tl.where(m_ij[:, None] > float("-inf"), p, 0.0)
        l_ij = tl.sum(p, 1)

        # Scale accumulator
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o_scale = tl.where(m_i > float("-inf"), acc_o_scale, 0.0)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        # Load V and accumulate PV
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )
        acc_o += tl.dot(p.to(v.dtype), v)

        # Update statistics
        m_i = m_ij
        exp_lse_diff = tl.where(lse_i > float("-inf"), tl.exp(lse_i - m_ij), 0.0)
        l_i_new = exp_lse_diff + l_ij
        l_i_new = tl.where(l_i_new > 0.0, l_i_new, 1.0)
        lse_i = m_ij + tl.log(l_i_new)

    # Final normalization
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    # Store output
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    # Store LSE (needed for the post_w1 second pass)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    # ═══ Second pass: post_w1 atomic aggregation ═══
    # Reload Q (rematerialized for register savings)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    for start_n in range(start_n_lo, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        valid_mask = (offs_m[:, None] >= (start_n + offs_n)[None, :]) & \
                     ((offs_m[:, None] - (start_n + offs_n)[None, :]) < window_size) & \
                     ((start_n + offs_n)[None, :] < seqlen_k) & \
                     (offs_m[:, None] < seqlen_q)

        k_abs = start_n + offs_n
        compact_k = k_abs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), window_size - 1)

        s_ptrs = (S_BUF + off_b * stride_sb + offs_m[:, None] * stride_sm + compact_k * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid_mask, other=0.0).to(tl.float32)

        score = direct_scale[:, None] * qk * softmax_scale + pre_w2_vals[:, None] * s_agg
        score = tl.where(valid_mask, score, float("-inf"))

        # Normalized probs using stored lse_i
        probs = tl.exp(score - lse_i[:, None])
        probs = tl.where(valid_mask & (lse_i[:, None] > float("-inf")), probs, 0.0)
        probs = tl.where(probs != probs, 0.0, probs)  # nan → 0

        # Atomic post_w1 aggregation
        a_ptrs = (A_BUF + off_b * stride_ab + offs_m[:, None] * stride_am + compact_k * stride_aw)
        tl.atomic_add(a_ptrs, post_w1_vals[:, None] * probs, mask=valid_mask, sem="relaxed")


class DCAttentionFA2:
    """DC residual attention built on FA2 Triton kernel.

    K0:  pre-agg            grid=(C, B)     BM=16  (same as before)
    K_dc: FA2-style DC      grid=(C, B*N)   BM=128 (FA2's natural tile size)
    K3:  a_buf@V + combine  grid=(C, B*N)   BM=64
    """

    @staticmethod
    def alloc_buffers(q, window):
        B, T, N, D = q.shape
        W = min(int(window), T)
        dev, dt = q.device, q.dtype
        seqlen_q_rounded = math.ceil(T / 128) * 128
        return {
            "s_buf": torch.empty((B, T, W), device=dev, dtype=dt),
            "a_buf": torch.empty((B, T, W), device=dev, dtype=torch.float32),
            "o_buf": torch.empty((B, T, N, D), device=dev, dtype=dt),
            "out":   torch.empty((B, T, N, D), device=dev, dtype=dt),
            "lse":   torch.empty((B * N, seqlen_q_rounded), device=dev, dtype=torch.float32),
            "tmp":   torch.empty((B * N, seqlen_q_rounded), device=dev, dtype=torch.float32),
        }

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None,
        bm_k0=16, bk_k0=64,
        block_m=128, block_n=128,
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
            buffers = DCAttentionFA2.alloc_buffers(q, W)
        s_buf = buffers["s_buf"]
        a_buf = buffers["a_buf"]
        o_buf = buffers["o_buf"]
        out   = buffers["out"]
        lse   = buffers["lse"]
        tmp   = buffers["tmp"]
        a_buf.zero_()

        seqlen_q_rounded = math.ceil(T / 128) * 128
        BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)

        # K0: pre-aggregation
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

        # K_dc: FA2-style DC forward (per-head, BLOCK_M=128)
        # Q/K/V are [B,T,N,D]. FA2 expects strides in (batch, head, seq) order.
        num_warps = 4 if D <= 64 else 8
        grid = (triton.cdiv(T, block_m), B * N)
        _dc_fwd_kernel[grid](
            q, k, v,
            s_buf,
            pre_w2, pre_dd, post_w1,
            o_buf, a_buf,
            lse, tmp,
            scaling,
            q.stride(0), q.stride(2), q.stride(1),   # stride_qb, stride_qh, stride_qm
            k.stride(0), k.stride(2), k.stride(1),   # stride_kb, stride_kh, stride_kn
            v.stride(0), v.stride(2), v.stride(1),   # stride_vb, stride_vh, stride_vn
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),  # stride_sb, stride_sm, stride_sw
            o_buf.stride(0), o_buf.stride(2), o_buf.stride(1),  # stride_ob, stride_oh, stride_om
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),  # stride_ab, stride_am, stride_aw
            # DC weights strides: [B,T,N] → stride_wb=T*N, stride_wm=N, stride_wh=1
            pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
            N, T, T, seqlen_q_rounded, D, W,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )

        # K3: a_buf@V + final combine
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
