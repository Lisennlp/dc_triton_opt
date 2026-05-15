"""H100-oriented parallel full-DC attention experiment.

This variant deliberately gives up V4's single-kernel/register-only contract to
recover the head-level parallelism that Hopper kernels such as FA3 rely on.

Pipeline:
- K1: one CTA per (chunk, batch-group), serial over heads only to build S_BUF.
- K2: one CTA per (chunk, batch-group, head), recompute QK, apply pre-mixed
  softmax, store the direct post_dd * PV output, and write post_w1 * probs
  into a per-head D_BUF.
- K2b: one CTA per (chunk, batch-group), reduce D_BUF over heads into A_BUF.
- K3: one CTA per (chunk, batch-group, head-pair) for D=128/even HPG, or per
  head otherwise, computing A_BUF @ V and adding post_w2 * AV to OUT.

The intent is to test whether H100 prefers more CTAs/head parallelism over the
single-kernel V4/V3 group-serial structure for large KL=256 windows.
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
def _v4hp_k1_sbuf(
    Q, K, PRE_W1, S_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_wb, stride_wt, stride_wn,
    stride_sb, stride_sg, stride_sc, stride_sm, stride_sk,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    head_start = g * HPG
    q_start = pid_c * BM

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

    s_acc = tl.zeros([BM, KL], dtype=tl.float32)
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        qk = _load_qk(
            Q, K,
            b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
            stride_qb, stride_qt, stride_qn, stride_qd,
            stride_kb, stride_kt, stride_kn, stride_kd,
            scaling,
        )
        pw1 = tl.load(
            PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        s_acc += pw1[:, None] * qk

    s_ptrs = (
        S_BUF + b * stride_sb + g * stride_sg + pid_c * stride_sc
        + tl.arange(0, BM)[:, None].to(tl.int64) * stride_sm
        + kl_offs[None, :].to(tl.int64) * stride_sk
    )
    tl.store(s_ptrs, s_acc, mask=q_mask[:, None] & k_mask[None, :])


@triton.jit
def _v4hp_k2_head(
    Q, K, V,
    PRE_W2, PRE_DD,
    POST_W1, POST_DD,
    OUT, S_BUF, D_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_sb, stride_sg, stride_sc, stride_sm, stride_sk,
    stride_db, stride_dg, stride_dh, stride_dc, stride_dm, stride_dk,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    h = tl.program_id(2)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    ni = (g * HPG + h).to(tl.int64)
    q_start = pid_c * BM

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

    s_ptrs = (
        S_BUF + b * stride_sb + g * stride_sg + pid_c * stride_sc
        + tl.arange(0, BM)[:, None].to(tl.int64) * stride_sm
        + kl_offs[None, :].to(tl.int64) * stride_sk
    )
    s_acc = tl.load(s_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

    qk = _load_qk(
        Q, K,
        b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
        stride_qb, stride_qt, stride_qn, stride_qd,
        stride_kb, stride_kt, stride_kn, stride_kd,
        scaling,
    )

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
    score = (pdd + 1.0)[:, None] * qk + pw2[:, None] * s_acc
    score = tl.where(valid, score, float("-inf"))
    rm = tl.max(score, axis=1)
    p = tl.exp2((score - rm[:, None]) * LOG2E)
    p = tl.where(valid, p, 0.0)
    rs = tl.sum(p, axis=1)
    sl = tl.where(rs > 0.0, rs, 1.0)
    probs = p / sl[:, None]

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
    tl.store(o_ptrs, (pddp[:, None] + 1.0) * pv, mask=q_mask[:, None])

    pw1p = tl.load(
        POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    d_ptrs = (
        D_BUF + b * stride_db + g * stride_dg + h * stride_dh + pid_c * stride_dc
        + tl.arange(0, BM)[:, None].to(tl.int64) * stride_dm
        + kl_offs[None, :].to(tl.int64) * stride_dk
    )
    tl.store(d_ptrs, (pw1p[:, None] * probs).to(tl.float16), mask=q_mask[:, None] & k_mask[None, :])


@triton.jit
def _v4hp_k2b_reduce_delta(
    D_BUF, A_BUF, SEQ_LENS,
    stride_db, stride_dg, stride_dh, stride_dc, stride_dm, stride_dk,
    stride_ab, stride_ag, stride_ac, stride_am, stride_ak,
    B: tl.constexpr, T: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_c * BM

    q_offs = q_start + tl.arange(0, BM)
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))

    acc = tl.zeros([BM, KL], dtype=tl.float32)
    for h in range(HPG):
        d_ptrs = (
            D_BUF + b * stride_db + g * stride_dg + h * stride_dh + pid_c * stride_dc
            + tl.arange(0, BM)[:, None].to(tl.int64) * stride_dm
            + kl_offs[None, :].to(tl.int64) * stride_dk
        )
        acc += tl.load(d_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

    a_ptrs = (
        A_BUF + b * stride_ab + g * stride_ag + pid_c * stride_ac
        + tl.arange(0, BM)[:, None].to(tl.int64) * stride_am
        + kl_offs[None, :].to(tl.int64) * stride_ak
    )
    tl.store(a_ptrs, acc.to(tl.float16), mask=q_mask[:, None] & k_mask[None, :])


@triton.jit
def _v4hp_k3_final_av(
    V, POST_W2, OUT, A_BUF, SEQ_LENS,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_ag, stride_ac, stride_am, stride_ak,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    h = tl.program_id(2)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    ni = (g * HPG + h).to(tl.int64)
    q_start = pid_c * BM

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

    a_ptrs = (
        A_BUF + b * stride_ab + g * stride_ag + pid_c * stride_ac
        + tl.arange(0, BM)[:, None].to(tl.int64) * stride_am
        + kl_offs[None, :].to(tl.int64) * stride_ak
    )
    a_blk = tl.load(a_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

    v_ptrs = (
        V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
        + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd
    )
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    av = tl.dot(a_blk.to(v_blk.dtype), v_blk)

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
    tl.store(o_ptrs, o_prev + pw2p[:, None] * av, mask=q_mask[:, None])


@triton.jit
def _v4hp_k3_final_av_wide2(
    V, POST_W2, OUT, A_BUF, SEQ_LENS,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_ag, stride_ac, stride_am, stride_ak,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    pair_idx = tl.program_id(2)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    n0 = (g * HPG + pair_idx * 2).to(tl.int64)
    q_start = pid_c * BM

    q_offs = q_start + tl.arange(0, BM)
    d2_offs = tl.arange(0, 256)
    head_delta = d2_offs // 128
    d_pair = d2_offs - head_delta * 128
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))

    a_ptrs = (
        A_BUF + b * stride_ab + g * stride_ag + pid_c * stride_ac
        + tl.arange(0, BM)[:, None].to(tl.int64) * stride_am
        + kl_offs[None, :].to(tl.int64) * stride_ak
    )
    a_blk = tl.load(a_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

    n_pair = n0 + head_delta.to(tl.int64)
    v_ptrs = (
        V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
        + n_pair[None, :] * stride_vn + d_pair[None, :].to(tl.int64) * stride_vd
    )
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    av = tl.dot(a_blk.to(v_blk.dtype), v_blk)

    pw2_0 = tl.load(
        POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    pw2_1 = tl.load(
        POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + (n0 + 1) * stride_wn,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    pw2p = tl.where(head_delta[None, :] == 0, pw2_0[:, None], pw2_1[:, None])

    o_ptrs = (
        OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
        + n_pair[None, :] * stride_on + d_pair[None, :].to(tl.int64) * stride_od
    )
    o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    tl.store(o_ptrs, o_prev + pw2p * av, mask=q_mask[:, None])


class TritonDCOneKernel:
    """V4HP: H100 head-parallel full-DC experiment without atomics."""

    @staticmethod
    def make_buffers(q, G=16, chunk_size=16, window=256):
        B, T, N, _ = q.shape
        HPG = N // G
        W = min(int(window), T)
        KL = triton.next_power_of_2(chunk_size + W - 1)
        num_chunks = triton.cdiv(T, chunk_size)
        group_shape = (B, G, num_chunks, chunk_size, KL)
        delta_shape = (B, G, HPG, num_chunks, chunk_size, KL)
        return {
            "s_buf": torch.empty(group_shape, device=q.device, dtype=torch.float32),
            "d_buf": torch.empty(delta_shape, device=q.device, dtype=q.dtype),
            "a_buf": torch.empty(group_shape, device=q.device, dtype=q.dtype),
        }

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=16, chunk_size=16, buffers=None,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G
        assert N % G == 0
        assert HPG >= 1

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous(); pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous(); post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous(); post_dd = post_dd.contiguous()

        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)
        num_chunks = triton.cdiv(T, chunk_size)
        expected_shape = (B, G, num_chunks, chunk_size, KL)
        expected_delta_shape = (B, G, HPG, num_chunks, chunk_size, KL)

        if buffers is None or "s_buf" not in buffers or "d_buf" not in buffers or "a_buf" not in buffers:
            buffers = TritonDCOneKernel.make_buffers(q, G=G, chunk_size=chunk_size, window=W)
        s_buf = buffers["s_buf"]
        d_buf = buffers["d_buf"]
        a_buf = buffers["a_buf"]
        if (
            tuple(s_buf.shape) != expected_shape
            or tuple(a_buf.shape) != expected_shape
            or tuple(d_buf.shape) != expected_delta_shape
        ):
            raise ValueError(
                f"buffer shapes {tuple(s_buf.shape)}, {tuple(d_buf.shape)}, {tuple(a_buf.shape)} "
                f"!= {expected_shape}, {expected_delta_shape}, {expected_shape}"
            )
        if s_buf.dtype != torch.float32 or d_buf.dtype != q.dtype or a_buf.dtype != q.dtype:
            raise ValueError("s_buf must be float32; d_buf and a_buf must match q dtype")

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)

        grid_k1 = (num_chunks, B * G)
        grid_k2 = (num_chunks, B * G, HPG)
        use_wide2 = D == 128 and HPG % 2 == 0
        grid_k3 = (num_chunks, B * G, HPG // 2) if use_wide2 else (num_chunks, B * G, HPG)

        _v4hp_k1_sbuf[grid_k1](
            q, k, pre_w1, s_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2), s_buf.stride(3), s_buf.stride(4),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
            num_warps=4, num_stages=1,
        )

        _v4hp_k2_head[grid_k2](
            q, k, v,
            pre_w2, pre_dd,
            post_w1, post_dd,
            out, s_buf, d_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2), s_buf.stride(3), s_buf.stride(4),
            d_buf.stride(0), d_buf.stride(1), d_buf.stride(2), d_buf.stride(3), d_buf.stride(4), d_buf.stride(5),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
            num_warps=4, num_stages=1,
        )

        _v4hp_k2b_reduce_delta[grid_k1](
            d_buf, a_buf, seq_lens,
            d_buf.stride(0), d_buf.stride(1), d_buf.stride(2), d_buf.stride(3), d_buf.stride(4), d_buf.stride(5),
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2), a_buf.stride(3), a_buf.stride(4),
            B=B, T=T, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
            num_warps=4, num_stages=1,
        )

        if use_wide2:
            _v4hp_k3_final_av_wide2[grid_k3](
                v, post_w2, out, a_buf, seq_lens,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                a_buf.stride(0), a_buf.stride(1), a_buf.stride(2), a_buf.stride(3), a_buf.stride(4),
                B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
                num_warps=4, num_stages=1,
            )
        else:
            _v4hp_k3_final_av[grid_k3](
                v, post_w2, out, a_buf, seq_lens,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                a_buf.stride(0), a_buf.stride(1), a_buf.stride(2), a_buf.stride(3), a_buf.stride(4),
                B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
                num_warps=4, num_stages=1,
            )
        return out
