from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


T_THRESH = 1024

@triton.jit
def _agg_finalise_kernel(
    AGG,
    stride_ab, stride_at, stride_ak,
    B: tl.constexpr, T: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = pid_m * BLOCK_M
    b = pid_b.to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi

        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
        agg_val = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        agg_val = tl.where(valid, agg_val, float('-inf'))
        tl.store(a_ptrs, agg_val, mask=m_mask[:, None] & k_mask[None, :])


# ── K1-sequential: grid(T/BM, B), loop N heads inside, fp16 output ──

@triton.jit
def _qk_agg_seq_kernel(
    Q, K, PW2, AGG,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_pb, stride_pt, stride_pn,
    stride_ab, stride_at, stride_ak,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = pid_m * BLOCK_M
    b = pid_b.to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi

        agg_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        for n in range(N):
            ni = n.to(tl.int64)
            q_ptrs = Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd
            q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
            k_ptrs = K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            qk = tl.dot(q_blk, tl.trans(k_blk))

            pw2_ptrs = PW2 + b * stride_pb + m_offs.to(tl.int64) * stride_pt + ni * stride_pn
            pw2 = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            agg_acc += pw2[:, None] * qk

        agg_acc = agg_acc * scaling

        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        full_mask = causal & win & m_mask[:, None] & k_mask[None, :]
        agg_acc = tl.where(full_mask, agg_acc, float('-inf'))

        a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
        tl.store(a_ptrs, agg_acc.to(tl.float16), mask=full_mask)


@triton.jit
def _qk_agg_atomic_kernel(
    Q, K, PW2, AGG,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_pb, stride_pt, stride_pn,
    stride_ab, stride_at, stride_ak,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    pw2_ptrs = PW2 + b * stride_pb + m_offs.to(tl.int64) * stride_pt + n * stride_pn
    pw2 = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_hi

        q_ptrs = Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt + n * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd
        q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        k_ptrs = K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt + n * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        qk = tl.dot(q_blk, tl.trans(k_blk))
        weighted = pw2[:, None] * qk * scaling

        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        full_mask = causal & win & m_mask[:, None] & k_mask[None, :]
        weighted = tl.where(full_mask, weighted, 0.0)

        a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
        tl.atomic_add(a_ptrs, weighted, mask=full_mask)


@triton.jit
def _pv_post_kernel(
    AGG_POST, V, PW1_POST, OUT,
    stride_apb, stride_apt, stride_apk,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_pb, stride_pt, stride_pn,
    stride_ob, stride_ot, stride_on, stride_od,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    AP_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)

    pw1_ptrs = PW1_POST + b * stride_pb + m_offs.to(tl.int64) * stride_pt + n * stride_pn
    pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < T)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        ap_ptrs = AGG_POST + b * stride_apb + m_offs[:, None].to(tl.int64) * stride_apt + k_offs[None, :].to(tl.int64) * stride_apk
        if AP_FP32:
            weights = tl.load(ap_ptrs, mask=valid, other=0.0).to(tl.float32)
        else:
            weights = tl.load(ap_ptrs, mask=valid, other=0.0).to(tl.float32)

        v_ptrs = V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt + n * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        acc += tl.dot(weights.to(tl.float16), v_blk)

    out = pw1[:, None] * acc
    o_ptrs = OUT + b * stride_ob + m_offs[:, None].to(tl.int64) * stride_ot + n * stride_on + d_offs[None, :].to(tl.int64) * stride_od
    tl.store(o_ptrs, out.to(tl.float16), mask=m_mask[:, None])



@triton.jit
def _softmax_postagg_seq_kernel(
    AGG, PW1, PW2_POST, AGG_POST,
    stride_ab, stride_at, stride_ak,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_pw2b, stride_pw2t, stride_pw2n,
    stride_apb, stride_apt, stride_apk,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = pid_m * BLOCK_M
    b = pid_b.to(tl.int64)

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    for n in range(N):
        ni = n.to(tl.int64)
        pw1_ptrs = PW1 + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + ni * stride_pw1n
        pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        pw2_ptrs = PW2_POST + b * stride_pw2b + m_offs.to(tl.int64) * stride_pw2t + ni * stride_pw2n
        pw2 = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)

        m_n = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_n = tl.zeros([BLOCK_M], dtype=tl.float32)

        for kb in range(k_lo, k_hi, BLOCK_K):
            k_offs = kb + tl.arange(0, BLOCK_K)
            k_mask = (k_offs < k_hi) & (k_offs < T)
            causal = k_offs[None, :] <= m_offs[:, None]
            win = (m_offs[:, None] - k_offs[None, :]) < W
            valid = causal & win & m_mask[:, None] & k_mask[None, :]

            a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
            agg_blk = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=float('-inf')).to(tl.float32)
            mixed = pw1[:, None] * agg_blk
            mixed = tl.where(valid, mixed, float('-inf'))

            m_new = tl.maximum(m_n, tl.max(mixed, axis=1))
            alpha = tl.exp(m_n - m_new)
            p = tl.exp(mixed - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            l_n = l_n * alpha + tl.sum(p, axis=1)
            m_n = m_new

        safe_l = tl.where(l_n > 0, l_n, 1.0)

        for kb in range(k_lo, k_hi, BLOCK_K):
            k_offs = kb + tl.arange(0, BLOCK_K)
            k_mask = (k_offs < k_hi) & (k_offs < T)
            causal = k_offs[None, :] <= m_offs[:, None]
            win = (m_offs[:, None] - k_offs[None, :]) < W
            valid = causal & win & m_mask[:, None] & k_mask[None, :]

            a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
            agg_blk = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=float('-inf')).to(tl.float32)
            mixed = pw1[:, None] * agg_blk
            mixed = tl.where(valid, mixed, float('-inf'))

            probs = tl.exp(mixed - m_n[:, None]) / safe_l[:, None]
            probs = tl.where(valid, probs, 0.0)

            ap_ptrs = AGG_POST + b * stride_apb + m_offs[:, None].to(tl.int64) * stride_apt + k_offs[None, :].to(tl.int64) * stride_apk
            ap_blk = tl.load(ap_ptrs, mask=valid, other=0.0).to(tl.float32)
            ap_blk += pw2[:, None] * probs
            tl.store(ap_ptrs, ap_blk.to(tl.float16), mask=valid)



@triton.jit
def _softmax_postagg_atomic_kernel(
    AGG, PW1, PW2_POST, AGG_POST,
    stride_ab, stride_at, stride_ak,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_pw2b, stride_pw2t, stride_pw2n,
    stride_apb, stride_apt, stride_apk,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    AGG_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    b = (pid_bn // N).to(tl.int64)
    n = (pid_bn % N).to(tl.int64)
    q_start = pid_m * BLOCK_M

    m_offs = q_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BLOCK_M
    if k_hi > T:
        k_hi = T

    pw1_ptrs = PW1 + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + n * stride_pw1n
    pw1 = tl.load(pw1_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    pw2_ptrs = PW2_POST + b * stride_pw2b + m_offs.to(tl.int64) * stride_pw2t + n * stride_pw2n
    pw2 = tl.load(pw2_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # Pass 1: online softmax
    m_n = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_n = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < T)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
        if AGG_FP32:
            agg_blk = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=float('-inf')).to(tl.float32)
        else:
            agg_blk = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=float('-inf')).to(tl.float32)
        mixed = pw1[:, None] * agg_blk
        mixed = tl.where(valid, mixed, float('-inf'))

        m_new = tl.maximum(m_n, tl.max(mixed, axis=1))
        alpha = tl.exp(m_n - m_new)
        p = tl.exp(mixed - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_n = l_n * alpha + tl.sum(p, axis=1)
        m_n = m_new

    # Pass 2: probs → atomic add into agg_post (fp32)
    safe_l = tl.where(l_n > 0, l_n, 1.0)

    for kb in range(k_lo, k_hi, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < k_hi) & (k_offs < T)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]

        a_ptrs = AGG + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at + k_offs[None, :].to(tl.int64) * stride_ak
        if AGG_FP32:
            agg_blk = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=float('-inf')).to(tl.float32)
        else:
            agg_blk = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=float('-inf')).to(tl.float32)
        mixed = pw1[:, None] * agg_blk
        mixed = tl.where(valid, mixed, float('-inf'))

        probs = tl.exp(mixed - m_n[:, None]) / safe_l[:, None]
        probs = tl.where(valid, probs, 0.0)
        contrib = pw2[:, None] * probs

        ap_ptrs = AGG_POST + b * stride_apb + m_offs[:, None].to(tl.int64) * stride_apt + k_offs[None, :].to(tl.int64) * stride_apk
        tl.atomic_add(ap_ptrs, contrib, mask=valid)


def _launch_k1(q, k, pw2, agg, scaling, W, block_m, block_k, use_atomic):
    B, T, N, D = q.shape
    if use_atomic:
        grid = (triton.cdiv(T, block_m), B * N)
        _qk_agg_atomic_kernel[grid](
            q, k, pw2, agg,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pw2.stride(0), pw2.stride(1), pw2.stride(2),
            agg.stride(0), agg.stride(1), agg.stride(2),
            scaling, B=B, T=T, N=N, D=D, W=W,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=2,
        )
        grid_fin = (triton.cdiv(T, block_m), B)
        _agg_finalise_kernel[grid_fin](
            agg, agg.stride(0), agg.stride(1), agg.stride(2),
            B=B, T=T, W=W,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=1,
        )
    else:
        grid = (triton.cdiv(T, block_m), B)
        _qk_agg_seq_kernel[grid](
            q, k, pw2, agg,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pw2.stride(0), pw2.stride(1), pw2.stride(2),
            agg.stride(0), agg.stride(1), agg.stride(2),
            scaling, B=B, T=T, N=N, D=D, W=W,
            BLOCK_M=block_m, BLOCK_K=block_k,
            num_warps=4, num_stages=2,
        )


class TritonDCRank1D0Atg:
    """D0 atomic pipeline: K1(qk-agg) -> K2(softmax+post-agg) -> K3(pv-post).

    Uses [B,T,T] buffers. T<=T_THRESH: atomic parallel, else sequential.
    """

    @staticmethod
    def forward(q, k, v, pw1_pre, pw2_pre, pw1_post, pw2_post,
                scaling, W, block_m=16, block_k=32,
                agg=None, agg_post=None, out=None):
        B, T, N, D = q.shape
        use_atomic = (T <= T_THRESH)

        if agg is None:
            if use_atomic:
                agg = torch.zeros(B, T, T, device=q.device, dtype=torch.float32)
            else:
                agg = torch.full((B, T, T), float('-inf'),
                                 device=q.device, dtype=q.dtype)
        else:
            if use_atomic:
                agg.zero_()
            else:
                agg.fill_(float('-inf'))

        if agg_post is None:
            if use_atomic:
                agg_post = torch.zeros(B, T, T, device=q.device,
                                       dtype=torch.float32)
            else:
                agg_post = torch.zeros(B, T, T, device=q.device, dtype=q.dtype)
        else:
            agg_post.zero_()

        if out is None:
            out = torch.empty(B, T, N, D, device=q.device, dtype=q.dtype)

        _launch_k1(q, k, pw1_pre, agg, scaling, W, block_m, block_k,
                    use_atomic)

        if use_atomic:
            grid2 = (triton.cdiv(T, block_m), B * N)
            _softmax_postagg_atomic_kernel[grid2](
                agg, pw2_pre, pw1_post, agg_post,
                agg.stride(0), agg.stride(1), agg.stride(2),
                pw2_pre.stride(0), pw2_pre.stride(1), pw2_pre.stride(2),
                pw1_post.stride(0), pw1_post.stride(1), pw1_post.stride(2),
                agg_post.stride(0), agg_post.stride(1), agg_post.stride(2),
                B=B, T=T, N=N, W=W,
                BLOCK_M=block_m, BLOCK_K=block_k,
                AGG_FP32=True,
                num_warps=4, num_stages=1,
            )
        else:
            grid2 = (triton.cdiv(T, block_m), B)
            _softmax_postagg_seq_kernel[grid2](
                agg, pw2_pre, pw1_post, agg_post,
                agg.stride(0), agg.stride(1), agg.stride(2),
                pw2_pre.stride(0), pw2_pre.stride(1), pw2_pre.stride(2),
                pw1_post.stride(0), pw1_post.stride(1), pw1_post.stride(2),
                agg_post.stride(0), agg_post.stride(1), agg_post.stride(2),
                B=B, T=T, N=N, W=W,
                BLOCK_M=block_m, BLOCK_K=block_k,
                num_warps=4, num_stages=1,
            )

        grid3 = (triton.cdiv(T, block_m), B * N)
        _pv_post_kernel[grid3](
            agg_post, v, pw2_post, out,
            agg_post.stride(0), agg_post.stride(1), agg_post.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pw2_post.stride(0), pw2_post.stride(1), pw2_post.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, T=T, N=N, D=D, W=W,
            BLOCK_M=block_m, BLOCK_K=block_k,
            AP_FP32=use_atomic,
            num_warps=4, num_stages=1,
        )
        return out