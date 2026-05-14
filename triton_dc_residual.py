from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _k0_qk_preagg_kernel(
    Q, K, PRE_W1, QK_BUF, S_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_p1b, stride_p1t, stride_p1n,
    stride_qkb, stride_qkn, stride_qkt, stride_qkw,
    stride_sb, stride_st, stride_sw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        s_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for n in range(N):
            ni = n.to(tl.int64)
            q_ptrs = (Q + b * stride_qb
                      + m_offs[:, None].to(tl.int64) * stride_qt
                      + ni * stride_qn
                      + d_offs[None, :].to(tl.int64) * stride_qd)
            q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

            k_ptrs = (K + b * stride_kb
                      + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn
                      + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

            qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

            pw1_ptrs = PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + ni * stride_p1n
            pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            s_acc += pw1[:, None] * qk

            qk_ptrs = (QK_BUF + b * stride_qkb
                       + ni * stride_qkn
                       + m_offs[:, None].to(tl.int64) * stride_qkt
                       + compact_k.to(tl.int64) * stride_qkw)
            tl.store(qk_ptrs, qk.to(tl.float16), mask=valid)

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        tl.store(s_ptrs, s_acc.to(tl.float16), mask=valid)


@triton.jit
def _k0_qk_preagg_atomic_kernel(
    Q, K, PRE_W1, QK_BUF, S_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_p1b, stride_p1t, stride_p1n,
    stride_qkb, stride_qkn, stride_qkt, stride_qkw,
    stride_sb, stride_st, stride_sw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    pw1 = tl.load(PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + n * stride_p1n,
                  mask=m_mask, other=0.0).to(tl.float32)

    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        qk_ptrs = (QK_BUF + b * stride_qkb
                   + n * stride_qkn
                   + m_offs[:, None].to(tl.int64) * stride_qkt
                   + compact_k.to(tl.int64) * stride_qkw)
        tl.store(qk_ptrs, qk.to(tl.float16), mask=valid)

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        tl.atomic_add(s_ptrs, pw1[:, None] * qk, mask=valid, sem="relaxed")


@triton.jit
def _k0_preagg_only_atomic_kernel(
    Q, K, PRE_W1, S_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_p1b, stride_p1t, stride_p1n,
    stride_sb, stride_st, stride_sw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Like _k0_preagg_only_kernel but parallelizes over (B*N) instead of B.
    Each block handles one (b, n) pair for a tile of queries, uses atomic_add to s_buf.
    Better GPU utilization when B is small.
    """
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    pw1 = tl.load(PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + n * stride_p1n,
                  mask=m_mask, other=0.0).to(tl.float32)

    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        tl.atomic_add(s_ptrs, pw1[:, None] * qk, mask=valid, sem="relaxed")


@triton.jit
def _k0_preagg_only_kernel(
    Q, K, PRE_W1, S_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_p1b, stride_p1t, stride_p1n,
    stride_sb, stride_st, stride_sw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    pid_m = tl.program_id(0)
    pid_1 = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_1 // B
        pid_1 = pid_1 % B
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = pid_1.to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        s_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for n in range(N):
            ni = n.to(tl.int64)
            q_ptrs = (Q + b * stride_qb
                      + m_offs[:, None].to(tl.int64) * stride_qt
                      + ni * stride_qn
                      + d_offs[None, :].to(tl.int64) * stride_qd)
            q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

            k_ptrs = (K + b * stride_kb
                      + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn
                      + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

            pw1_ptrs = PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + ni * stride_p1n
            pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            s_acc += pw1[:, None] * qk

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        tl.store(s_ptrs, s_acc.to(tl.float16), mask=valid)


@triton.jit
def _k1_stats_kernel(
    QK_BUF, S_BUF, PRE_W2, PRE_DD, M_BUF, L_BUF, SEQ_LENS,
    stride_qkb, stride_qkn, stride_qkt, stride_qkw,
    stride_sb, stride_st, stride_sw,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    pre_w2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_dd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                     mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = pre_dd + 1.0

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        qk_ptrs = (QK_BUF + b * stride_qkb
                   + n * stride_qkn
                   + m_offs[:, None].to(tl.int64) * stride_qkt
                   + compact_k.to(tl.int64) * stride_qkw)
        qk = tl.load(qk_ptrs, mask=valid, other=0.0).to(tl.float32)

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))

        block_max = tl.max(score, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    m_ptrs = M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + n * stride_mn
    l_ptrs = L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + n * stride_ln
    tl.store(m_ptrs, m_i, mask=m_mask)
    tl.store(l_ptrs, l_i, mask=m_mask)


@triton.jit
def _k2_probs_out_postagg_kernel(
    QK_BUF, S_BUF, M_BUF, L_BUF, V, PRE_W2, PRE_DD, POST_W1, O_BUF, A_BUF, SEQ_LENS,
    stride_qkb, stride_qkn, stride_qkt, stride_qkw,
    stride_sb, stride_st, stride_sw,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_post1b, stride_post1t, stride_post1n,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_at, stride_aw,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
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
    post_w1 = tl.load(POST_W1 + b * stride_post1b + m_offs.to(tl.int64) * stride_post1t + n * stride_post1n,
                      mask=m_mask, other=0.0).to(tl.float32)

    m_i = tl.load(M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + n * stride_mn,
                  mask=m_mask, other=float("-inf")).to(tl.float32)
    l_i = tl.load(L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + n * stride_ln,
                  mask=m_mask, other=1.0).to(tl.float32)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)

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
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        qk_ptrs = (QK_BUF + b * stride_qkb
                   + n * stride_qkn
                   + m_offs[:, None].to(tl.int64) * stride_qkt
                   + compact_k.to(tl.int64) * stride_qkw)
        qk = tl.load(qk_ptrs, mask=valid, other=0.0).to(tl.float32)

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))
        probs = tl.exp(score - m_i[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)

        a_ptrs = (A_BUF + b * stride_ab
                  + m_offs[:, None].to(tl.int64) * stride_at
                  + compact_k.to(tl.int64) * stride_aw)
        tl.atomic_add(a_ptrs, post_w1[:, None] * probs, mask=valid, sem="relaxed")

        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(probs.to(tl.float16), v_blk)

    o_ptrs = (O_BUF + b * stride_ob
              + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on
              + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _k1_stats_recompute_kernel(
    Q, K, S_BUF, PRE_W2, PRE_DD, M_BUF, L_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_sb, stride_st, stride_sw,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_bn // (B * N)
        pid_bn = pid_bn % (B * N)
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    pre_w2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_dd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                     mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = pre_dd + 1.0

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))

        block_max = tl.max(score, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    m_ptrs = M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + n * stride_mn
    l_ptrs = L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + n * stride_ln
    tl.store(m_ptrs, m_i, mask=m_mask)
    tl.store(l_ptrs, l_i, mask=m_mask)


@triton.jit
def _k2_probs_out_postagg_recompute_kernel(
    Q, K, S_BUF, M_BUF, L_BUF, V, PRE_W2, PRE_DD, POST_W1, O_BUF, A_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_sb, stride_st, stride_sw,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_post1b, stride_post1t, stride_post1n,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_at, stride_aw,
    scaling,
    DO_POSTAGG: tl.constexpr,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_bn // (B * N)
        pid_bn = pid_bn % (B * N)
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    pre_w2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_dd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                     mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = pre_dd + 1.0
    post_w1 = tl.load(POST_W1 + b * stride_post1b + m_offs.to(tl.int64) * stride_post1t + n * stride_post1n,
                      mask=m_mask, other=0.0).to(tl.float32)

    m_i = tl.load(M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + n * stride_mn,
                  mask=m_mask, other=float("-inf")).to(tl.float32)
    l_i = tl.load(L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + n * stride_ln,
                  mask=m_mask, other=1.0).to(tl.float32)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)

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
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))
        probs = tl.exp(score - m_i[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)

        if DO_POSTAGG: # default true
            a_ptrs = (A_BUF + b * stride_ab
                      + m_offs[:, None].to(tl.int64) * stride_at
                      + compact_k.to(tl.int64) * stride_aw)
            tl.atomic_add(a_ptrs, post_w1[:, None] * probs, mask=valid, sem="relaxed")

        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(probs.to(tl.float16), v_blk)

    o_ptrs = (O_BUF + b * stride_ob
              + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on
              + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _k2_probs_out_storep_recompute_kernel(
    Q, K, S_BUF, M_BUF, L_BUF, V, PRE_W2, PRE_DD, POST_W1, O_BUF, P_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_sb, stride_st, stride_sw,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_post1b, stride_post1t, stride_post1n,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_pbb, stride_pbn, stride_pbt, stride_pbw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    pre_w2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_dd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                     mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = pre_dd + 1.0
    post_w1 = tl.load(POST_W1 + b * stride_post1b + m_offs.to(tl.int64) * stride_post1t + n * stride_post1n,
                      mask=m_mask, other=0.0).to(tl.float32)

    m_i = tl.load(M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + n * stride_mn,
                  mask=m_mask, other=float("-inf")).to(tl.float32)
    l_i = tl.load(L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + n * stride_ln,
                  mask=m_mask, other=1.0).to(tl.float32)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)

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
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))
        probs = tl.exp(score - m_i[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)

        p_ptrs = (P_BUF + b * stride_pbb
                  + n * stride_pbn
                  + m_offs[:, None].to(tl.int64) * stride_pbt
                  + compact_k.to(tl.int64) * stride_pbw)
        tl.store(p_ptrs, (post_w1[:, None] * probs).to(tl.float16), mask=valid)

        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(probs.to(tl.float16), v_blk)

    o_ptrs = (O_BUF + b * stride_ob
              + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on
              + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _reduce_post_probs_kernel(
    P_BUF, A_BUF, SEQ_LENS,
    stride_pbb, stride_pbn, stride_pbt, stride_pbw,
    stride_ab, stride_at, stride_aw,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for n in range(N):
            ni = n.to(tl.int64)
            p_ptrs = (P_BUF + b * stride_pbb
                      + ni * stride_pbn
                      + m_offs[:, None].to(tl.int64) * stride_pbt
                      + compact_k.to(tl.int64) * stride_pbw)
            acc += tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)

        a_ptrs = (A_BUF + b * stride_ab
                  + m_offs[:, None].to(tl.int64) * stride_at
                  + compact_k.to(tl.int64) * stride_aw)
        tl.store(a_ptrs, acc, mask=valid)


@triton.jit
def _k3_final_kernel(
    A_BUF, V, O_BUF, POST_W2, POST_DD, OUT, SEQ_LENS,
    stride_ab, stride_at, stride_aw,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_outb, stride_outt, stride_outn, stride_outd,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_bn // (B * N)
        pid_bn = pid_bn % (B * N)
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    post_w2 = tl.load(POST_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                      mask=m_mask, other=0.0).to(tl.float32)
    post_dd = tl.load(POST_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                      mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = post_dd + 1.0

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
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        a_ptrs = (A_BUF + b * stride_ab
                  + m_offs[:, None].to(tl.int64) * stride_at
                  + compact_k.to(tl.int64) * stride_aw)
        a_blk = tl.load(a_ptrs, mask=valid, other=0.0).to(tl.float32) # 没有乘以post_w2的score

        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a_blk.to(tl.float16), v_blk)

    o_ptrs = (O_BUF + b * stride_ob
              + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on
              + d_offs[None, :].to(tl.int64) * stride_od)
    # o_blk: 在第2个kernel中存储的probs * V
    o_blk = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    # direct_scale[:, None] * o_blk: 本来应该是(post_dd + residual) * V，但是这里直接存储了 residual * V，所以这里需要乘以direct_scale
    # post_w2[:, None] * acc # 经过了post_w1的probs * post_w2
    out = direct_scale[:, None] * o_blk + post_w2[:, None] * acc #前半部分是(dd + residual) 后半部分是probs * post

    out_ptrs = (OUT + b * stride_outb
                + m_offs[:, None].to(tl.int64) * stride_outt
                + n * stride_outn
                + d_offs[None, :].to(tl.int64) * stride_outd)
    tl.store(out_ptrs, out.to(tl.float16), mask=m_mask[:, None])


# =============================================================================
# BACKWARD KERNELS
# =============================================================================

@triton.jit
def _bwd_k3_final_kernel(
    DOUT, A_BUF, V, O_BUF, POST_W2, POST_DD, SEQ_LENS,
    DO_BUF, DA_BUF, DV,
    DPOST_W2, DPOST_DD,
    stride_doutb, stride_doutt, stride_doutn, stride_doutd,
    stride_ab, stride_at, stride_aw,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_dob, stride_dot, stride_don, stride_dod,
    stride_dab, stride_dat, stride_daw,
    stride_dvb, stride_dvt, stride_dvn, stride_dvd,
    stride_dp2b, stride_dp2t, stride_dp2n,
    stride_dpdb, stride_dpdt, stride_dpdn,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    """Backward of _k3_final_kernel.
    out[n,t] = (1+post_dd[t,n]) * o_buf[t,n,:] + post_w2[t,n] * sum_s a_buf[t,s] * v[s,n,:]
    Computes: d_o_buf, d_a_buf, d_v (atomic), d_post_w2, d_post_dd
    """
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_bn // (B * N)
        pid_bn = pid_bn % (B * N)
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    post_w2 = tl.load(POST_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                      mask=m_mask, other=0.0).to(tl.float32)
    post_dd = tl.load(POST_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                      mask=m_mask, other=0.0).to(tl.float32)
    direct_scale = post_dd + 1.0

    # Load dout [B,T,N,D]
    dout_ptrs = (DOUT + b * stride_doutb
                 + m_offs[:, None].to(tl.int64) * stride_doutt
                 + n * stride_doutn
                 + d_offs[None, :].to(tl.int64) * stride_doutd)
    dout_blk = tl.load(dout_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Load o_buf [B,T,N,D]
    o_ptrs = (O_BUF + b * stride_ob
              + m_offs[:, None].to(tl.int64) * stride_ot
              + n * stride_on
              + d_offs[None, :].to(tl.int64) * stride_od)
    o_blk = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # d_post_dd = sum_d dout * o_buf
    d_post_dd_val = tl.sum(dout_blk * o_blk, axis=1)

    # d_o_buf = direct_scale * dout
    d_o_blk = direct_scale[:, None] * dout_blk

    # Store d_o_buf
    do_ptrs = (DO_BUF + b * stride_dob
               + m_offs[:, None].to(tl.int64) * stride_dot
               + n * stride_don
               + d_offs[None, :].to(tl.int64) * stride_dod)
    tl.store(do_ptrs, d_o_blk.to(tl.float16), mask=m_mask[:, None])

    # For the a_buf*v branch:
    # acc_d = sum_s a_buf[t,s] * v[s,n,d] => d_a_buf[t,s] = sum_d dout_scaled[t,d] * v[s,n,d]
    # where dout_scaled = post_w2 * dout
    dout_scaled = post_w2[:, None] * dout_blk  # [BLOCK_M, D]

    # d_post_w2 = sum_d dout * acc_d  where acc_d = sum_s a_buf * v
    # We compute it as sum over key blocks
    d_post_w2_val = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        # Load a_buf [B,T,W]
        a_ptrs = (A_BUF + b * stride_ab
                  + m_offs[:, None].to(tl.int64) * stride_at
                  + compact_k.to(tl.int64) * stride_aw)
        a_blk = tl.load(a_ptrs, mask=valid, other=0.0).to(tl.float32)

        # Load v [B,T,N,D]
        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # d_a_buf[t,s] = sum_d dout_scaled[t,d] * v[s,n,d]
        d_a = tl.dot(dout_scaled.to(tl.float16), tl.trans(v_blk.to(tl.float16))).to(tl.float32)
        d_a = tl.where(valid, d_a, 0.0)

        # Accumulate d_a_buf (shared across heads, needs atomic)
        da_ptrs = (DA_BUF + b * stride_dab
                   + m_offs[:, None].to(tl.int64) * stride_dat
                   + compact_k.to(tl.int64) * stride_daw)
        tl.atomic_add(da_ptrs, d_a, mask=valid, sem="relaxed")

        # d_v[s,n,d] += sum_t (post_w2[t,n] * dout[t,d]) * a_buf[t,s]  (atomic)
        # = (dout_scaled^T @ a_blk)^T but we accumulate per-s
        # d_v[s,:] += a_blk^T @ dout_scaled => [BLOCK_K, D]
        d_v_blk = tl.dot(tl.trans(a_blk.to(tl.float16)), dout_scaled.to(tl.float16)).to(tl.float32)

        dv_ptrs = (DV + b * stride_dvb
                   + k_offs[:, None].to(tl.int64) * stride_dvt
                   + n * stride_dvn
                   + d_offs[None, :].to(tl.int64) * stride_dvd)
        tl.atomic_add(dv_ptrs, d_v_blk, mask=k_mask[:, None], sem="relaxed")

        # d_post_w2 += sum_d dout[t,d] * (a_buf[t,s] * v[s,n,d])
        # = sum_s a_blk * (dout @ v^T) but simpler: sum_s d_a * a_blk already computed differently
        # Actually: d_post_w2[t] = sum_d dout[t,d] * acc_d[t,d]
        # acc_d[t,d] = sum_s a_buf[t,s] * v[s,n,d]
        # So: d_post_w2[t] += sum_s a_buf[t,s] * sum_d dout[t,d] * v[s,n,d]
        # = sum_s a_blk[t,s] * (dout @ v^T)[t,s]
        dout_v = tl.dot(dout_blk.to(tl.float16), tl.trans(v_blk.to(tl.float16))).to(tl.float32)
        d_post_w2_val += tl.sum(tl.where(valid, a_blk * dout_v, 0.0), axis=1)

    # Store d_post_w2, d_post_dd
    dp2_ptrs = DPOST_W2 + b * stride_dp2b + m_offs.to(tl.int64) * stride_dp2t + n * stride_dp2n
    tl.atomic_add(dp2_ptrs, d_post_w2_val, mask=m_mask, sem="relaxed")

    dpd_ptrs = DPOST_DD + b * stride_dpdb + m_offs.to(tl.int64) * stride_dpdt + n * stride_dpdn
    tl.atomic_add(dpd_ptrs, d_post_dd_val, mask=m_mask, sem="relaxed")


@triton.jit
def _bwd_fused_mid_kernel(
    Q, K, S_BUF, M_BUF, L_BUF, V, PRE_W1, PRE_W2, PRE_DD, POST_W1,
    DO_BUF, DA_BUF, SEQ_LENS,
    DV, DPRE_W2, DPRE_DD, DPOST_W1,
    D_S_BUF, D_SAGG_BUF,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_sb, stride_st, stride_sw,
    stride_mb, stride_mt, stride_mn,
    stride_lb, stride_lt, stride_ln,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_p1b, stride_p1t, stride_p1n,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_post1b, stride_post1t, stride_post1n,
    stride_dob, stride_dot, stride_don, stride_dod,
    stride_dab, stride_dat, stride_daw,
    stride_dvb, stride_dvt, stride_dvn, stride_dvd,
    stride_dp2b, stride_dp2t, stride_dp2n,
    stride_dpdb, stride_dpdt, stride_dpdn,
    stride_dpo1b, stride_dpo1t, stride_dpo1n,
    stride_dsb, stride_dsn, stride_dst, stride_dsw,
    stride_dsab, stride_dsat, stride_dsaw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    """Fused backward for PV->post-agg->softmax->pre-mix->pre-agg.
    
    Per (b, n, tile_m): recomputes probs from saved stats, then:
    - PV backward: from do_buf compute d_probs contribution and d_v
    - Post-agg backward: d_probs += post_w1 * d_a_buf, d_post_w1
    - Softmax backward: d_score = probs * (d_probs - rowsum)
    - Pre-mix backward: d_logits, d_pre_dd, d_pre_w2
    - Pre-agg backward: d_pre_w1, final d_logits stored to D_S_BUF
    """
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_bn // (B * N)
        pid_bn = pid_bn % (B * N)
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    # Load per-query DC weights
    pre_w1 = tl.load(PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + n * stride_p1n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_w2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + n * stride_p2n,
                     mask=m_mask, other=0.0).to(tl.float32)
    pre_dd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + n * stride_pdn,
                     mask=m_mask, other=0.0).to(tl.float32)
    direct_scale_pre = pre_dd + 1.0

    post_w1 = tl.load(POST_W1 + b * stride_post1b + m_offs.to(tl.int64) * stride_post1t + n * stride_post1n,
                      mask=m_mask, other=0.0).to(tl.float32)

    # Load softmax stats
    m_i = tl.load(M_BUF + b * stride_mb + m_offs.to(tl.int64) * stride_mt + n * stride_mn,
                  mask=m_mask, other=float("-inf")).to(tl.float32)
    l_i = tl.load(L_BUF + b * stride_lb + m_offs.to(tl.int64) * stride_lt + n * stride_ln,
                  mask=m_mask, other=1.0).to(tl.float32)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)

    # Load do_buf [B,T,N,D]
    do_ptrs = (DO_BUF + b * stride_dob
               + m_offs[:, None].to(tl.int64) * stride_dot
               + n * stride_don
               + d_offs[None, :].to(tl.int64) * stride_dod)
    do_blk = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Load q for recomputing QK
    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    # Accumulators for DC weight gradients
    d_pre_w2_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    d_pre_dd_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    d_post_w1_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Pass 1: accumulate rowsum = sum_s(d_probs * probs) for softmax backward
    rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        # Recompute QK and probs
        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale_pre[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))
        probs = tl.exp(score - m_i[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)

        # d_probs from PV: do_buf @ v^T
        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        d_probs_pv = tl.dot(do_blk.to(tl.float16), tl.trans(v_blk)).to(tl.float32)
        d_probs_pv = tl.where(valid, d_probs_pv, 0.0)

        # d_probs from post-agg: post_w1 * d_a_buf
        da_ptrs = (DA_BUF + b * stride_dab
                   + m_offs[:, None].to(tl.int64) * stride_dat
                   + compact_k.to(tl.int64) * stride_daw)
        d_a_blk = tl.load(da_ptrs, mask=valid, other=0.0).to(tl.float32)
        d_probs_pa = post_w1[:, None] * d_a_blk

        d_probs = d_probs_pv + d_probs_pa
        rowsum += tl.sum(d_probs * probs, axis=1)

        # Accumulate d_v from PV path: probs^T @ do_buf
        d_v_blk = tl.dot(tl.trans(probs.to(tl.float16)), do_blk.to(tl.float16)).to(tl.float32)
        dv_ptrs = (DV + b * stride_dvb
                   + k_offs[:, None].to(tl.int64) * stride_dvt
                   + n * stride_dvn
                   + d_offs[None, :].to(tl.int64) * stride_dvd)
        tl.atomic_add(dv_ptrs, d_v_blk, mask=k_mask[:, None], sem="relaxed")

        # Accumulate d_post_w1
        d_post_w1_acc += tl.sum(d_a_blk * probs, axis=1)

    # Pass 2: compute d_score and propagate to d_logits
    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        # Recompute probs
        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        s_ptrs = (S_BUF + b * stride_sb
                  + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        score = direct_scale_pre[:, None] * qk + pre_w2[:, None] * s_agg
        score = tl.where(valid, score, float("-inf"))
        probs = tl.exp(score - m_i[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)

        # Recompute d_probs
        v_ptrs = (V + b * stride_vb
                  + k_offs[:, None].to(tl.int64) * stride_vt
                  + n * stride_vn
                  + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        d_probs_pv = tl.dot(do_blk.to(tl.float16), tl.trans(v_blk)).to(tl.float32)
        d_probs_pv = tl.where(valid, d_probs_pv, 0.0)

        da_ptrs = (DA_BUF + b * stride_dab
                   + m_offs[:, None].to(tl.int64) * stride_dat
                   + compact_k.to(tl.int64) * stride_daw)
        d_a_blk = tl.load(da_ptrs, mask=valid, other=0.0).to(tl.float32)
        d_probs_pa = post_w1[:, None] * d_a_blk

        d_probs = d_probs_pv + d_probs_pa

        # Softmax backward
        d_score = probs * (d_probs - rowsum[:, None])
        d_score = tl.where(valid, d_score, 0.0)

        # Pre-mix backward: score = (1+pre_dd)*qk + pre_w2*s_agg
        d_qk = d_score * direct_scale_pre[:, None]
        d_s_agg_local = d_score * pre_w2[:, None]

        d_pre_dd_acc += tl.sum(d_score * qk, axis=1)
        d_pre_w2_acc += tl.sum(d_score * s_agg, axis=1)

        # Store d_qk (per-head component without cross-head term) into D_S_BUF [B, N, T, W]
        ds_ptrs = (D_S_BUF + b * stride_dsb
                   + n * stride_dsn
                   + m_offs[:, None].to(tl.int64) * stride_dst
                   + compact_k.to(tl.int64) * stride_dsw)
        tl.store(ds_ptrs, d_qk.to(tl.float16), mask=valid)

        # Accumulate d_s_agg = pre_w2[t,n] * d_score[n,t,s] into D_SAGG_BUF [B,T,W] (atomic across heads)
        dsagg_ptrs = (D_SAGG_BUF + b * stride_dsab
                      + m_offs[:, None].to(tl.int64) * stride_dsat
                      + compact_k.to(tl.int64) * stride_dsaw)
        tl.atomic_add(dsagg_ptrs, d_s_agg_local, mask=valid, sem="relaxed")

    # Store per-head weight grads
    dp2_ptrs = DPRE_W2 + b * stride_dp2b + m_offs.to(tl.int64) * stride_dp2t + n * stride_dp2n
    tl.atomic_add(dp2_ptrs, d_pre_w2_acc, mask=m_mask, sem="relaxed")

    dpd_ptrs = DPRE_DD + b * stride_dpdb + m_offs.to(tl.int64) * stride_dpdt + n * stride_dpdn
    tl.atomic_add(dpd_ptrs, d_pre_dd_acc, mask=m_mask, sem="relaxed")

    dpo1_ptrs = DPOST_W1 + b * stride_dpo1b + m_offs.to(tl.int64) * stride_dpo1t + n * stride_dpo1n
    tl.atomic_add(dpo1_ptrs, d_post_w1_acc, mask=m_mask, sem="relaxed")


@triton.jit
def _bwd_qk_kernel(
    Q, K, D_S_BUF, D_SAGG_BUF, PRE_W1, SEQ_LENS,
    DQ, DK, DPRE_W1,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_dsb, stride_dsn, stride_dst, stride_dsw,
    stride_dsab, stride_dsat, stride_dsaw,
    stride_p1b, stride_p1t, stride_p1n,
    stride_dqb, stride_dqt, stride_dqn, stride_dqd,
    stride_dkb, stride_dkt, stride_dkn, stride_dkd,
    stride_dp1b, stride_dp1t, stride_dp1n,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr = 0,
):
    """BWD step 0: QK backward.
    d_logits_total[n,t,s] = D_S_BUF[n,t,s] + pre_w1[t,n] * D_SAGG_BUF[t,s]
    d_q[t,n,:] = scaling * sum_s d_logits_total * k[s,n,:]
    d_k[s,n,:] = scaling * sum_t d_logits_total * q[t,n,:]  (atomic)
    d_pre_w1[t,n] = sum_s d_sagg_buf[t,s] * logits[n,t,s]
    """
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    if TILES_PER_GROUP > 0:
        group_id = pid_bn // (B * N)
        pid_bn = pid_bn % (B * N)
        q_start = (group_id * TILES_PER_GROUP + pid_m) * BLOCK_M
    else:
        q_start = pid_m * BLOCK_M
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    # Load pre_w1 for this head
    pre_w1 = tl.load(PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + n * stride_p1n,
                     mask=m_mask, other=0.0).to(tl.float32)

    # Load q
    q_ptrs = (Q + b * stride_qb
              + m_offs[:, None].to(tl.int64) * stride_qt
              + n * stride_qn
              + d_offs[None, :].to(tl.int64) * stride_qd)
    q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    # Accumulate d_q and d_pre_w1
    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    d_pre_w1_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        # Load d_qk from D_S_BUF [B, N, T, W]
        ds_ptrs = (D_S_BUF + b * stride_dsb
                   + n * stride_dsn
                   + m_offs[:, None].to(tl.int64) * stride_dst
                   + compact_k.to(tl.int64) * stride_dsw)
        d_qk = tl.load(ds_ptrs, mask=valid, other=0.0).to(tl.float32)

        # Load d_sagg from D_SAGG_BUF [B, T, W]
        dsagg_ptrs = (D_SAGG_BUF + b * stride_dsab
                      + m_offs[:, None].to(tl.int64) * stride_dsat
                      + compact_k.to(tl.int64) * stride_dsaw)
        d_sagg = tl.load(dsagg_ptrs, mask=valid, other=0.0).to(tl.float32)

        # Complete d_logits = d_qk + pre_w1 * d_sagg
        d_logits = d_qk + pre_w1[:, None] * d_sagg

        # Load k
        k_ptrs = (K + b * stride_kb
                  + k_offs[:, None].to(tl.int64) * stride_kt
                  + n * stride_kn
                  + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        # d_q += scaling * d_logits @ k
        dq_acc += scaling * tl.dot(d_logits.to(tl.float16), k_blk)

        # d_k += scaling * d_logits^T @ q  (atomic)
        dk_blk = scaling * tl.dot(tl.trans(d_logits.to(tl.float16)), q_blk)
        dk_ptrs = (DK + b * stride_dkb
                   + k_offs[:, None].to(tl.int64) * stride_dkt
                   + n * stride_dkn
                   + d_offs[None, :].to(tl.int64) * stride_dkd)
        tl.atomic_add(dk_ptrs, dk_blk.to(tl.float32), mask=k_mask[:, None], sem="relaxed")

        # d_pre_w1[t,n] = sum_s d_sagg_total[t,s] * logits[n,t,s]
        # Recompute logits = scaling * q @ k^T
        logits = tl.dot(q_blk, tl.trans(k_blk)) * scaling
        d_pre_w1_acc += tl.sum(tl.where(valid, d_sagg * logits, 0.0), axis=1)

    # Store d_q
    dq_ptrs = (DQ + b * stride_dqb
               + m_offs[:, None].to(tl.int64) * stride_dqt
               + n * stride_dqn
               + d_offs[None, :].to(tl.int64) * stride_dqd)
    tl.store(dq_ptrs, dq_acc.to(tl.float16), mask=m_mask[:, None])

    # Store d_pre_w1
    dp1_ptrs = DPRE_W1 + b * stride_dp1b + m_offs.to(tl.int64) * stride_dp1t + n * stride_dp1n
    tl.atomic_add(dp1_ptrs, d_pre_w1_acc, mask=m_mask, sem="relaxed")



class TritonDCResidual:
    """Full rank-1 plus diagonal-residual DC attention for [B, T, N, D] tensors."""

    @staticmethod
    def alloc_buffers(
        q: torch.Tensor,
        window: int,
        store_qk: bool = False,
        store_post_probs: bool = False,
    ):
        B, T, N, D = q.shape
        W = min(int(window), T)
        device = q.device
        dtype = q.dtype
        buffers = {
            "s_buf": torch.empty((B, T, W), device=device, dtype=dtype),
            "m_buf": torch.empty((B, T, N), device=device, dtype=torch.float32),
            "l_buf": torch.empty((B, T, N), device=device, dtype=torch.float32),
            "a_buf": torch.empty((B, T, W), device=device, dtype=torch.float32),
            "o_buf": torch.empty((B, T, N, D), device=device, dtype=dtype),
            "out": torch.empty((B, T, N, D), device=device, dtype=dtype),
        }
        if store_qk:
            buffers["qk_buf"] = torch.empty((B, N, T, W), device=device, dtype=dtype)
        if store_post_probs:
            buffers["p_buf"] = torch.empty((B, N, T, W), device=device, dtype=dtype)
        return buffers

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dc_weights,
        scaling: float,
        window: int,
        seq_lens: torch.Tensor | None = None,
        block_m: int = 16,
        block_k: int = 64,
        block_m_final: int = 64,
        block_k_final: int = 32,
        atomic_preagg: bool = False,
        store_qk: bool = False,
        store_post_probs: bool = False,
        buffers: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        pre_w1 = pre_w1.contiguous()
        pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous()
        post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous()
        post_dd = post_dd.contiguous()

        if buffers is None:
            buffers = TritonDCResidual.alloc_buffers(
                q, W, store_qk=store_qk, store_post_probs=store_post_probs
            )
        qk_buf = buffers.get("qk_buf")
        s_buf = buffers["s_buf"]
        m_buf = buffers["m_buf"]
        l_buf = buffers["l_buf"]
        a_buf = buffers["a_buf"]
        p_buf = buffers.get("p_buf")
        o_buf = buffers["o_buf"] # probs * V，在第二个kernel中就计算了probs * V并存储在o_buf中
        out = buffers["out"]
        if not store_post_probs:
            a_buf.zero_()

        # Compute grid grouping: when B is small and T is large, split T tiles
        # into groups and move them to grid dim-1 to balance the 2D grid shape.
        # This improves GPU occupancy by making the CUDA block scheduler distribute
        # work more evenly across SMs.
        num_tiles = triton.cdiv(T, block_m)
        num_tiles_final = triton.cdiv(T, block_m_final)
        TARGET_DIM0 = 64  # target max tiles in grid dim-0

        # For K0: grid is (tiles, B). Only rebalance when B is very small (<=4)
        # and T is large enough to make the grid very lopsided.
        bn = B * N
        if num_tiles > TARGET_DIM0 and B <= 4:
            tiles_per_group_k0 = TARGET_DIM0
            num_groups_k0 = triton.cdiv(num_tiles, tiles_per_group_k0)
            grid_k0_grouped = (tiles_per_group_k0, B * num_groups_k0)
        else:
            tiles_per_group_k0 = 0
            grid_k0_grouped = None

        # For K1/K2: grid is (tiles, B*N)
        if num_tiles > TARGET_DIM0 and bn < num_tiles:
            tiles_per_group_bn = TARGET_DIM0
            num_groups_bn = triton.cdiv(num_tiles, tiles_per_group_bn)
            grid_bn_grouped = (tiles_per_group_bn, bn * num_groups_bn)
        else:
            tiles_per_group_bn = 0
            grid_bn_grouped = None

        # For K3: grid is (tiles_final, B*N), use smaller dim0 target
        TARGET_DIM0_FINAL = 16
        if num_tiles_final > TARGET_DIM0_FINAL and bn < num_tiles_final:
            tiles_per_group_final = TARGET_DIM0_FINAL
            num_groups_final = triton.cdiv(num_tiles_final, tiles_per_group_final)
            grid_final_grouped = (tiles_per_group_final, bn * num_groups_final)
        else:
            tiles_per_group_final = 0
            grid_final_grouped = None

        if store_qk and atomic_preagg:
            assert qk_buf is not None
            s_buf.zero_()
            grid_k0 = (triton.cdiv(T, block_m), B * N)
            _k0_qk_preagg_atomic_kernel[grid_k0](
                q, k, pre_w1, qk_buf, s_buf, seq_lens,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
                qk_buf.stride(0), qk_buf.stride(1), qk_buf.stride(2), qk_buf.stride(3),
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                scaling,
                B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                num_warps=4, num_stages=2,
            )
        elif store_qk:
            assert qk_buf is not None
            grid_b = (triton.cdiv(T, block_m), B)
            _k0_qk_preagg_kernel[grid_b](
                q, k, pre_w1, qk_buf, s_buf, seq_lens,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
                qk_buf.stride(0), qk_buf.stride(1), qk_buf.stride(2), qk_buf.stride(3),
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                scaling,
                B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                num_warps=4, num_stages=2,
            )
        else:
            grid_k0 = grid_k0_grouped if grid_k0_grouped is not None else (num_tiles, B)
            tpg_k0 = tiles_per_group_k0 if grid_k0_grouped is not None else 0
            _k0_preagg_only_kernel[grid_k0](
                q, k, pre_w1, s_buf, seq_lens,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                scaling,
                B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                TILES_PER_GROUP=tpg_k0,
                num_warps=4, num_stages=2,
            )

        grid_bn = grid_bn_grouped if grid_bn_grouped is not None else (num_tiles, bn)
        tpg_bn = tiles_per_group_bn if grid_bn_grouped is not None else 0

        if store_qk:
            assert qk_buf is not None
            _k1_stats_kernel[(num_tiles, bn)](
                qk_buf, s_buf, pre_w2, pre_dd, m_buf, l_buf, seq_lens,
                qk_buf.stride(0), qk_buf.stride(1), qk_buf.stride(2), qk_buf.stride(3),
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
                pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
                m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
                l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
                B=B, T=T, N=N, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                num_warps=4, num_stages=1,
            )

            _k2_probs_out_postagg_kernel[(num_tiles, bn)](
                qk_buf, s_buf, m_buf, l_buf, v, pre_w2, pre_dd, post_w1, o_buf, a_buf, seq_lens,
                qk_buf.stride(0), qk_buf.stride(1), qk_buf.stride(2), qk_buf.stride(3),
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
                l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
                pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
                post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
                o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
                a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                num_warps=4, num_stages=2,
            )
        else:
            _k1_stats_recompute_kernel[grid_bn](
                q, k, s_buf, pre_w2, pre_dd, m_buf, l_buf, seq_lens,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
                pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
                m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
                l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
                scaling,
                B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                TILES_PER_GROUP=tpg_bn,
                num_warps=4, num_stages=2,
            )

            if store_post_probs:
                assert p_buf is not None
                _k2_probs_out_storep_recompute_kernel[grid_bn](
                    q, k, s_buf, m_buf, l_buf, v, pre_w2, pre_dd, post_w1, o_buf, p_buf, seq_lens,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
                    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
                    pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
                    post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
                    o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
                    p_buf.stride(0), p_buf.stride(1), p_buf.stride(2), p_buf.stride(3),
                    scaling,
                    B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                    num_warps=4, num_stages=2,
                )
                grid_reduce = (triton.cdiv(T, block_m), B)
                _reduce_post_probs_kernel[grid_reduce](
                    p_buf, a_buf, seq_lens,
                    p_buf.stride(0), p_buf.stride(1), p_buf.stride(2), p_buf.stride(3),
                    a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                    B=B, T=T, N=N, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                    num_warps=4, num_stages=1,
                )
            else:
                _k2_probs_out_postagg_recompute_kernel[grid_bn](
                    q, k, s_buf, m_buf, l_buf, v, pre_w2, pre_dd, post_w1, o_buf, a_buf, seq_lens,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
                    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
                    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
                    pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
                    post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
                    o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
                    a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                    scaling,
                    DO_POSTAGG=True,
                    B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
                    TILES_PER_GROUP=tpg_bn,
                    num_warps=4, num_stages=2,
                )

        grid_f = grid_final_grouped if grid_final_grouped is not None else (num_tiles_final, bn)
        tpg_f = tiles_per_group_final if grid_final_grouped is not None else 0
        _k3_final_kernel[grid_f](
            a_buf, v, o_buf, post_w2, post_dd, out, seq_lens,
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
            post_dd.stride(0), post_dd.stride(1), post_dd.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m_final, BLOCK_K=block_k_final,
            TILES_PER_GROUP=tpg_f,
            num_warps=4, num_stages=2,
        )
        return out

    @staticmethod
    def backward(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dc_weights,
        scaling: float,
        window: int,
        s_buf: torch.Tensor,
        m_buf: torch.Tensor,
        l_buf: torch.Tensor,
        a_buf: torch.Tensor,
        o_buf: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        block_m: int = 16,
        block_k: int = 64,
        block_m_final: int = 64,
        block_k_final: int = 32,
    ):
        """Backward pass for DC residual attention.
        
        Args:
            dout: gradient of output [B, T, N, D]
            q, k, v: input tensors [B, T, N, D]
            dc_weights: tuple of (pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd) each [B, T, N]
            scaling: attention scaling factor
            window: window size
            s_buf: saved pre-aggregation buffer [B, T, W]
            m_buf: saved softmax max [B, T, N]
            l_buf: saved softmax sum [B, T, N]
            a_buf: saved post-aggregation buffer [B, T, W]
            o_buf: saved per-head output before post-mix [B, T, N, D]
            seq_lens: sequence lengths [B]
            
        Returns:
            dq, dk, dv: gradients for q, k, v [B, T, N, D]
            d_pre_w1, d_pre_w2, d_pre_dd, d_post_w1, d_post_w2, d_post_dd: [B, T, N]
        """
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        # Ensure contiguous
        if not dout.is_contiguous():
            dout = dout.contiguous()
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        pre_w1 = pre_w1.contiguous()
        pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous()
        post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous()
        post_dd = post_dd.contiguous()

        device = q.device
        dtype = q.dtype

        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.zeros((B, T, N, D), device=device, dtype=torch.float32)
        dv = torch.zeros((B, T, N, D), device=device, dtype=torch.float32)
        d_pre_w1 = torch.zeros((B, T, N), device=device, dtype=torch.float32)
        d_pre_w2 = torch.zeros((B, T, N), device=device, dtype=torch.float32)
        d_pre_dd = torch.zeros((B, T, N), device=device, dtype=torch.float32)
        d_post_w1 = torch.zeros((B, T, N), device=device, dtype=torch.float32)
        d_post_w2 = torch.zeros((B, T, N), device=device, dtype=torch.float32)
        d_post_dd = torch.zeros((B, T, N), device=device, dtype=torch.float32)

        # Intermediate buffers
        do_buf = torch.empty((B, T, N, D), device=device, dtype=dtype)
        da_buf = torch.zeros((B, T, W), device=device, dtype=torch.float32)
        d_s_buf = torch.empty((B, N, T, W), device=device, dtype=dtype)
        d_sagg_buf = torch.zeros((B, T, W), device=device, dtype=torch.float32)

        # Compute grid grouping for backward (same logic as forward)
        num_tiles = triton.cdiv(T, block_m)
        num_tiles_final = triton.cdiv(T, block_m_final)
        TARGET_DIM0 = 64
        TARGET_DIM0_FINAL = 16
        bn = B * N

        if num_tiles > TARGET_DIM0 and bn < num_tiles:
            tiles_per_group_bn = TARGET_DIM0
            num_groups_bn = triton.cdiv(num_tiles, tiles_per_group_bn)
            grid_bn = (tiles_per_group_bn, bn * num_groups_bn)
            tpg_bn = tiles_per_group_bn
        else:
            grid_bn = (num_tiles, bn)
            tpg_bn = 0

        if num_tiles_final > TARGET_DIM0_FINAL and bn < num_tiles_final:
            tiles_per_group_final = TARGET_DIM0_FINAL
            num_groups_final = triton.cdiv(num_tiles_final, tiles_per_group_final)
            grid_final = (tiles_per_group_final, bn * num_groups_final)
            tpg_final = tiles_per_group_final
        else:
            grid_final = (num_tiles_final, bn)
            tpg_final = 0

        # ===== BWD Step 6: Final kernel backward =====
        # out = (1+post_dd)*o_buf + post_w2 * sum_s(a_buf * v)
        _bwd_k3_final_kernel[grid_final](
            dout, a_buf, v, o_buf, post_w2, post_dd, seq_lens,
            do_buf, da_buf, dv,
            d_post_w2, d_post_dd,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
            post_dd.stride(0), post_dd.stride(1), post_dd.stride(2),
            do_buf.stride(0), do_buf.stride(1), do_buf.stride(2), do_buf.stride(3),
            da_buf.stride(0), da_buf.stride(1), da_buf.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            d_post_w2.stride(0), d_post_w2.stride(1), d_post_w2.stride(2),
            d_post_dd.stride(0), d_post_dd.stride(1), d_post_dd.stride(2),
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m_final, BLOCK_K=block_k_final,
            TILES_PER_GROUP=tpg_final,
            num_warps=4, num_stages=2,
        )

        # ===== BWD Steps 5-1: Fused mid kernel =====
        _bwd_fused_mid_kernel[grid_bn](
            q, k, s_buf, m_buf, l_buf, v, pre_w1, pre_w2, pre_dd, post_w1,
            do_buf, da_buf, seq_lens,
            dv, d_pre_w2, d_pre_dd, d_post_w1,
            d_s_buf, d_sagg_buf,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
            m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
            l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
            pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
            post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
            do_buf.stride(0), do_buf.stride(1), do_buf.stride(2), do_buf.stride(3),
            da_buf.stride(0), da_buf.stride(1), da_buf.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            d_pre_w2.stride(0), d_pre_w2.stride(1), d_pre_w2.stride(2),
            d_pre_dd.stride(0), d_pre_dd.stride(1), d_pre_dd.stride(2),
            d_post_w1.stride(0), d_post_w1.stride(1), d_post_w1.stride(2),
            d_s_buf.stride(0), d_s_buf.stride(1), d_s_buf.stride(2), d_s_buf.stride(3),
            d_sagg_buf.stride(0), d_sagg_buf.stride(1), d_sagg_buf.stride(2),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
            TILES_PER_GROUP=tpg_bn,
            num_warps=4, num_stages=2,
        )

        # ===== BWD Step 0: QK backward (also computes d_pre_w1) =====
        _bwd_qk_kernel[grid_bn](
            q, k, d_s_buf, d_sagg_buf, pre_w1, seq_lens,
            dq, dk, d_pre_w1,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            d_s_buf.stride(0), d_s_buf.stride(1), d_s_buf.stride(2), d_s_buf.stride(3),
            d_sagg_buf.stride(0), d_sagg_buf.stride(1), d_sagg_buf.stride(2),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            d_pre_w1.stride(0), d_pre_w1.stride(1), d_pre_w1.stride(2),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=block_m, BLOCK_K=block_k,
            TILES_PER_GROUP=tpg_bn,
            num_warps=4, num_stages=2,
        )

        # Cast accumulated gradients to output dtype
        dk = dk.to(dtype)
        dv = dv.to(dtype)
        d_pre_w1 = d_pre_w1.to(dtype)
        d_pre_w2 = d_pre_w2.to(dtype)
        d_pre_dd = d_pre_dd.to(dtype)
        d_post_w1 = d_post_w1.to(dtype)
        d_post_w2 = d_post_w2.to(dtype)
        d_post_dd = d_post_dd.to(dtype)

        return dq, dk, dv, d_pre_w1, d_pre_w2, d_pre_dd, d_post_w1, d_post_w2, d_post_dd


# # Forward-only benchmark: TritonDCResidual vs Torch vs torch.compile vs FA2
# N=32, D=128, window=256, chunk=256, dtype=torch.float16
# ====================================================================================================

#   B     T     B*T |   Triton    Torch  Compile |    FA2-w    FA2-f |  Tri/FA2w  Tor/FA2w  Cmp/FA2w  Tri/FA2f
# ------------------------------------------------------------------------------------------------------------
#   1  4096    4096 |     1424     7584     1601 |      204      812 |   6.97x  37.10x   7.83x   1.75x
#   4  4096   16384 |     5711    24199     8242 |      801     3220 |   7.13x  30.23x  10.30x   1.77x
#  16  4096   65536 |    21851    90408    30386 |     3420    12584 |   6.39x  26.44x   8.89x   1.74x
#  32  4096  131072 |    43574   177806    59957 |     6714    24988 |   6.49x  26.48x   8.93x   1.74x
#  64  4096  262144 |    87665   354881   118617 |    13399    49756 |   6.54x  26.48x   8.85x   1.76x
#   1 16384   16384 |     5841    29724     6463 |      768    11967 |   7.61x  38.72x   8.42x   0.49x
#   1 32768   32768 |    11518    60081    12928 |     1651    47332 |   6.98x  36.40x   7.83x   0.24x
#   1 65536   65536 |    22873   121356    25795 |     3417   192619 |   6.69x  35.51x   7.55x   0.12x
#   1 98304   98304 |    34380   182075    54285 |     5161   455724 |   6.66x  35.28x  10.52x   0.08x
#   1 131072  131072 |    46086   237973    51540 |     6844   846089 |   6.73x  34.77x   7.53x   0.05x
#   1 262144  262144 |    93269   487817   153639 |    13665  3682595 |   6.83x  35.70x  11.24x   0.03x


class TritonDCResidual4K:
    """Grouped DC via reshape: [B,T,N,D] → [B*G,T,HPG,D], single kernel launch."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=1, **kwargs,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        HPG = N // G

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        if G == 1:
            return TritonDCResidual.forward(q, k, v, dc_weights, scaling, window, seq_lens, **kwargs)

        # Reshape [B, T, N, D] → [B, T, G, HPG, D] → [B*G, T, HPG, D]
        q_r = q.view(B, T, G, HPG, D).permute(0, 2, 1, 3, 4).reshape(B * G, T, HPG, D).contiguous()
        k_r = k.view(B, T, G, HPG, D).permute(0, 2, 1, 3, 4).reshape(B * G, T, HPG, D).contiguous()
        v_r = v.view(B, T, G, HPG, D).permute(0, 2, 1, 3, 4).reshape(B * G, T, HPG, D).contiguous()
        pw1_r = pre_w1.view(B, T, G, HPG).permute(0, 2, 1, 3).reshape(B * G, T, HPG).contiguous()
        pw2_r = pre_w2.view(B, T, G, HPG).permute(0, 2, 1, 3).reshape(B * G, T, HPG).contiguous()
        pdd_r = pre_dd.view(B, T, G, HPG).permute(0, 2, 1, 3).reshape(B * G, T, HPG).contiguous()
        qw1_r = post_w1.view(B, T, G, HPG).permute(0, 2, 1, 3).reshape(B * G, T, HPG).contiguous()
        qw2_r = post_w2.view(B, T, G, HPG).permute(0, 2, 1, 3).reshape(B * G, T, HPG).contiguous()
        qdd_r = post_dd.view(B, T, G, HPG).permute(0, 2, 1, 3).reshape(B * G, T, HPG).contiguous()

        # seq_lens: [B] → [B*G] (repeat each element G times)
        sl_r = seq_lens.unsqueeze(1).expand(B, G).reshape(B * G).contiguous()

        ws_r = (pw1_r, pw2_r, pdd_r, qw1_r, qw2_r, qdd_r)

        # Single forward call with B*G "batches", HPG "heads"
        o_r = TritonDCResidual.forward(q_r, k_r, v_r, ws_r, scaling, window, sl_r, **kwargs)

        # Reshape back: [B*G, T, HPG, D] → [B, G, T, HPG, D] → [B, T, N, D]
        out = o_r.view(B, G, T, HPG, D).permute(0, 2, 1, 3, 4).reshape(B, T, N, D).contiguous()
        return out