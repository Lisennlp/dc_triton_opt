from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════════════
# D0-3K Grouped: single-launch kernels — heads split into G groups.
# K1/K2b grid(T/BM, B*G); K3 grid(T/BM, B*N).  All groups in ONE launch.
# Inner loops: N → GROUP_SIZE (e.g., 32 → 8 for G=4), 4× less compute.
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _k1_grp_qk_stats_kernel(
    Q, K, PW1_PRE, PW2_PRE,
    S_BUF, M_BUF, L_BUF,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_pw2b, stride_pw2t, stride_pw2n,
    stride_sb, stride_sg, stride_st, stride_sk,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, G: tl.constexpr, GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (ceil(T/BLOCK_M), B*G). One program = one time-tile × (batch b, group g).
    # Global tensor shapes (logical; forward passes pw2_pre→PW1_PRE, pw1_pre→PW2_PRE):
    #   Q, K              [B, T, N, D]
    #   PW1_PRE, PW2_PRE  each [B, T, N]
    #   S_BUF             [B, G, T, W]   fp16 window-compressed scores (this group g)
    #   M_BUF, L_BUF      [B, T, N]      fp32 softmax running max / sum per head
    # Per-tile tensors (query rows m_offs = q_start + [0..BLOCK_M)):
    #   q_blk, k_blk      [BLOCK_M, D], [BLOCK_K, D]
    #   qk, s_acc         [BLOCK_M, BLOCK_K]
    #   pw2 (PW2_PRE)     [BLOCK_M] per head ni in inner loop
    #   pw1_j (PW1_PRE)   [BLOCK_M] per head ji in softmax loop
    #   score_j, p        [BLOCK_M, BLOCK_K]; m_j,l_j,new_m,new_l,block_max,alpha [BLOCK_M]
    pid_m = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_m * BLOCK_M
    h_start = g * GROUP_SIZE

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi

        s_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)  # tile [BM, BK]

        for h in range(GROUP_SIZE):
            ni = (h_start + h).to(tl.int64)
            q_ptrs = (Q + b * stride_qb
                      + m_offs[:, None].to(tl.int64) * stride_qt
                      + ni * stride_qn
                      + d_offs[None, :].to(tl.int64) * stride_qd)
            q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)  # [BLOCK_M, D]

            k_ptrs = (K + b * stride_kb
                      + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn
                      + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, D]

            qk = tl.dot(q_blk, tl.trans(k_blk))  # [BLOCK_M, BLOCK_K]

            pw2_ptrs = PW2_PRE + b * stride_pw2b + m_offs.to(tl.int64) * stride_pw2t + ni * stride_pw2n
            pw2 = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)  # [BLOCK_M]
            s_acc += pw2[:, None] * qk  # broadcast pw2 to [BLOCK_M, BLOCK_K]

        s_acc *= scaling

        causal = k_offs[None, :] <= m_offs[:, None]  # [BLOCK_M, BLOCK_K]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]
        s_acc = tl.where(valid, s_acc, float('-inf'))

        compact_k = tl.maximum(0, k_offs[None, :] - k_lo_per_q[:, None])  # [BLOCK_M, BLOCK_K]
        compact_k = tl.minimum(compact_k, W - 1)
        s_ptrs = (S_BUF + b * stride_sb + g * stride_sg
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sk)
        tl.store(s_ptrs, s_acc.to(tl.float16), mask=valid)

        for h in range(GROUP_SIZE):
            ji = (h_start + h).to(tl.int64)
            pw1_ptrs = PW1_PRE + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + ji * stride_pw1n
            pw1_j = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)  # [BLOCK_M]

            m_ptrs = M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + ji * stride_mn
            l_ptrs = L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + ji * stride_ln
            m_j = tl.load(m_ptrs, mask=m_mask, other=float('-inf'))  # [BLOCK_M]
            l_j = tl.load(l_ptrs, mask=m_mask, other=0.0)  # [BLOCK_M]

            score_j = pw1_j[:, None] * s_acc  # [BLOCK_M, BLOCK_K]
            score_j = tl.where(valid, score_j, float('-inf'))

            block_max = tl.max(score_j, axis=1)
            new_m = tl.maximum(m_j, block_max)
            alpha = tl.exp(m_j - new_m)
            p = tl.exp(score_j - new_m[:, None])
            p = tl.where(valid, p, 0.0)
            new_l = l_j * alpha + tl.sum(p, axis=1)

            tl.store(m_ptrs, new_m, mask=m_mask)
            tl.store(l_ptrs, new_l, mask=m_mask)


@triton.jit
def _k2b_grp_compute_u_kernel(
    S_BUF, M_BUF, L_BUF,
    PW1_PRE, PW2_POST, U_BUF,
    stride_sb, stride_sg, stride_st, stride_sk,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_pw2b, stride_pw2t, stride_pw2n,
    stride_ub, stride_ug, stride_ut, stride_uk,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    W: tl.constexpr, G: tl.constexpr, GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (ceil(T/BLOCK_M), B*G). Same tile partition as K1; sums GROUP_SIZE heads into U.
    # Global shapes (forward passes pw2_pre→PW1_PRE, pw1_post→PW2_POST):
    #   S_BUF        [B, G, T, W]   fp16 (same compact layout as K1 output)
    #   M_BUF,L_BUF  [B, T, N]
    #   PW1_PRE      [B, T, N]
    #   PW2_POST     [B, T, N]
    #   U_BUF        [B, G, T, W]   fp16 mixed probability accumulator (one row per group)
    # Tiles:
    #   s_blk        [BLOCK_M, BLOCK_K]
    #   u            [BLOCK_M, BLOCK_K]  Σ_h pw2_post[h]*prob_h
    #   pw1_j,pw2_j,m_j,l_j  [BLOCK_M]; prob_j [BLOCK_M, BLOCK_K]
    pid_m = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_m * BLOCK_M
    h_start = g * GROUP_SIZE

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = tl.maximum(0, k_offs[None, :] - k_lo_per_q[:, None])
        compact_k = tl.minimum(compact_k, W - 1)

        s_ptrs = (S_BUF + b * stride_sb + g * stride_sg
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sk)
        s_blk = tl.load(s_ptrs, mask=valid, other=float('-inf')).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        u = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for h in range(GROUP_SIZE):
            ji = (h_start + h).to(tl.int64)
            pw1_ptrs = PW1_PRE + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + ji * stride_pw1n
            pw1_j = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)  # [BLOCK_M]

            m_ptrs = M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + ji * stride_mn
            m_j = tl.load(m_ptrs, mask=m_mask, other=float('-inf'))  # [BLOCK_M]

            l_ptrs = L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + ji * stride_ln
            l_j = tl.load(l_ptrs, mask=m_mask, other=1.0)  # [BLOCK_M]

            pw2_ptrs = PW2_POST + b * stride_pw2b + m_offs.to(tl.int64) * stride_pw2t + ji * stride_pw2n
            pw2_j = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)  # [BLOCK_M]

            score_j = pw1_j[:, None] * s_blk  # [BLOCK_M, BLOCK_K]
            score_j = tl.where(valid, score_j, float('-inf'))
            safe_l = tl.where(l_j > 0, l_j, 1.0)
            prob_j = tl.exp(score_j - m_j[:, None]) / safe_l[:, None]  # [BLOCK_M, BLOCK_K]
            prob_j = tl.where(valid, prob_j, 0.0)

            u += pw2_j[:, None] * prob_j  # broadcast pw2_j to [BLOCK_M, BLOCK_K]

        u_ptrs = (U_BUF + b * stride_ub + g * stride_ug
                  + m_offs[:, None].to(tl.int64) * stride_ut
                  + compact_k.to(tl.int64) * stride_uk)
        tl.store(u_ptrs, u.to(tl.float16), mask=valid)


@triton.jit
def _k3_grp_pv_kernel(
    U_BUF, V, PW1_POST, OUT,
    stride_ub, stride_ug, stride_ut, stride_uk,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_pb, stride_pt, stride_pn,
    stride_ob, stride_ot, stride_on, stride_od,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (ceil(T/BLOCK_M), B*N). One program = one time-tile × (batch b, head n).
    # Global shapes (forward passes pw2_post→PW1_POST slot):
    #   U_BUF    [B, G, T, W]   fp16/fp32 mixed probs along compact key axis (G = group of head n)
    #   V        [B, T, N, D]
    #   PW1_POST [B, T, N]     post-mix expand weights per head
    #   OUT      [B, T, N, D]
    # Tiles:
    #   pw1      [BLOCK_M]  per head n for this program
    #   u_blk    [BLOCK_M, BLOCK_K]  weights over keys in window
    #   v_blk    [BLOCK_K, D]  one head n, key rows k_offs
    #   acc      [BLOCK_M, D]  dot-product accumulator across K tiles
    #   out_val  [BLOCK_M, D]  pw1[:,None] * acc
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    g = (n // GROUP_SIZE).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    pw1_ptrs = PW1_POST + b * stride_pb + m_offs.to(tl.int64) * stride_pt + n * stride_pn
    pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)  # [BLOCK_M]

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = tl.maximum(0, k_offs[None, :] - k_lo_per_q[:, None])
        compact_k = tl.minimum(compact_k, W - 1)

        u_ptrs = (U_BUF + b * stride_ub + g * stride_ug
                  + m_offs[:, None].to(tl.int64) * stride_ut
                  + compact_k.to(tl.int64) * stride_uk)
        u_blk = tl.load(u_ptrs, mask=valid, other=0.0)  # [BLOCK_M, BLOCK_K]

        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, D]

        acc += tl.dot(u_blk.to(tl.float16), v_blk)  # [BLOCK_M, D]

    out_val = pw1[:, None] * acc  # [BLOCK_M, D]
    o_ptrs = (OUT + b * stride_ob
              + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on
              + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, out_val.to(tl.float16), mask=m_mask[:, None])


class TritonDCRank1_3K_Grouped:
    """D0: grouped heads — single-launch kernels.

    K1 grid(T/BM, B*G), K2b grid(T/BM, B*G), K3 grid(T/BM, B*N).
    Inner loop: GROUP_SIZE instead of N.

    Tensor / tile shapes are documented at the top of each kernel:
    `_k1_grp_qk_stats_kernel`, `_k2b_grp_compute_u_kernel`, `_k3_grp_pv_kernel`.
    """

    @staticmethod
    def forward(q, k, v, pre_w1, pre_w2, post_w1, post_w2,
                scaling, window,
                n_groups=1, block_m=32, block_k=64,
                s_buf=None, m_buf=None, l_buf=None, u_buf=None, out=None):
        B, T, N, D = q.shape
        W = min(window, T)
        G = n_groups
        H = N // G
        assert N % G == 0

        if pre_w1.dim() == 4:
            pw1_pre = pre_w1.squeeze(-1).contiguous()
            pw2_pre = pre_w2.squeeze(2).contiguous()
            pw1_post = post_w1.squeeze(-1).contiguous()
            pw2_post = post_w2.squeeze(2).contiguous()
        else:
            pw1_pre, pw2_pre = pre_w1, pre_w2
            pw1_post, pw2_post = post_w1, post_w2

        if s_buf is None:
            s_buf = torch.empty(B, G, T, W, device=q.device, dtype=torch.float16)
        if m_buf is None:
            m_buf = torch.full((B, T, N), float('-inf'), device=q.device, dtype=torch.float32)
        else:
            m_buf.fill_(float('-inf'))
        if l_buf is None:
            l_buf = torch.zeros(B, T, N, device=q.device, dtype=torch.float32)
        else:
            l_buf.zero_()

        grid1 = (triton.cdiv(T, block_m), B * G)
        _k1_grp_qk_stats_kernel[grid1](
            q, k, pw2_pre, pw1_pre,
            s_buf, m_buf, l_buf,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pw2_pre.stride(0), pw2_pre.stride(1), pw2_pre.stride(2),
            pw1_pre.stride(0), pw1_pre.stride(1), pw1_pre.stride(2),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2), s_buf.stride(3),
            m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
            l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
            scaling,
            B=B, T=T, N=N, D=D, W=W, G=G, GROUP_SIZE=H,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=2,
        )

        if u_buf is None:
            u_buf = torch.empty(B, G, T, W, device=q.device, dtype=torch.float16)

        grid2 = (triton.cdiv(T, block_m), B * G)
        _k2b_grp_compute_u_kernel[grid2](
            s_buf, m_buf, l_buf,
            pw2_pre, pw1_post, u_buf,
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2), s_buf.stride(3),
            m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
            l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
            pw2_pre.stride(0), pw2_pre.stride(1), pw2_pre.stride(2),
            pw1_post.stride(0), pw1_post.stride(1), pw1_post.stride(2),
            u_buf.stride(0), u_buf.stride(1), u_buf.stride(2), u_buf.stride(3),
            B=B, T=T, N=N, W=W, G=G, GROUP_SIZE=H,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=1,
        )

        if out is None:
            out = torch.empty(B, T, N, D, device=q.device, dtype=torch.float16)

        grid3 = (triton.cdiv(T, block_m), B * N)
        _k3_grp_pv_kernel[grid3](
            u_buf, v, pw2_post, out,
            u_buf.stride(0), u_buf.stride(1), u_buf.stride(2), u_buf.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pw2_post.stride(0), pw2_post.stride(1), pw2_post.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, T=T, N=N, D=D, W=W, GROUP_SIZE=H,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=2,
        )
        return out
