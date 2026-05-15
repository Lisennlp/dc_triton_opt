"""Single-kernel DC V4H: H100-oriented V4 experiment.

This is intentionally kept as a separate benchmark target instead of replacing
V4. It preserves the V4 one-kernel full-DC dataflow but tries choices that are
more plausible on H100:
- cached QK matrices are narrowed to fp16 after the fp32 pre-aggregation;
- the causal/window validity mask is hoisted and reused;
- HPG=8 / KL<=128 has a static cache8 path, avoiding Sweep-2 QK recompute;
- the final shared-a_acc AV pass uses a D=256 wide2 dot for D=128 even HPG;
- KL=256 is allowed through the fp16 cache4 path for an H100 wide-window trial.

Unsupported shapes fall back to V4 so this can be used as a drop-in benchmark
column.
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
    q = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                mask=q_mask[:, None], other=0.0)
    k = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                mask=k_mask[:, None], other=0.0)
    return tl.dot(q, tl.trans(k)) * scaling


@triton.jit
def _consume_qk(
    qk, s_acc, a_acc, valid,
    b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
    PRE_W2, PRE_DD, POST_W1, POST_DD,
    V, OUT,
    stride_wb, stride_wt, stride_wn,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_ob, stride_ot, stride_on, stride_od,
):
    qk_f = qk.to(tl.float32)
    pw2 = tl.load(PRE_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                   mask=q_mask, other=0.0).to(tl.float32)
    pdd = tl.load(PRE_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                  mask=q_mask, other=0.0).to(tl.float32)

    score = (pdd + 1.0)[:, None] * qk_f + pw2[:, None] * s_acc
    score = tl.where(valid, score, float("-inf"))
    rm = tl.max(score, axis=1)
    p = tl.exp2((score - rm[:, None]) * LOG2E)
    p = tl.where(valid, p, 0.0)
    rs = tl.sum(p, axis=1)
    sl = tl.where(rs > 0.0, rs, 1.0)
    probs = p / sl[:, None]

    pw1p = tl.load(POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                   mask=q_mask, other=0.0).to(tl.float32)
    a_acc += pw1p[:, None] * probs

    v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
              + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    pv = tl.dot(probs.to(v_blk.dtype), v_blk)

    pddp = tl.load(POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                   mask=q_mask, other=0.0).to(tl.float32)
    o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
              + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, ((pddp[:, None] + 1.0) * pv), mask=q_mask[:, None])

    return a_acc


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=1),
        # triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=["B", "T", "N", "D", "W", "BM", "KL", "G", "HPG"],
)
@triton.jit
def _dc_onekernel_cache4(
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

    if HPG == 8 and KL <= 128:
        n0 = head_start.to(tl.int64)
        n1 = (head_start + 1).to(tl.int64)
        n2 = (head_start + 2).to(tl.int64)
        n3 = (head_start + 3).to(tl.int64)
        n4 = (head_start + 4).to(tl.int64)
        n5 = (head_start + 5).to(tl.int64)
        n6 = (head_start + 6).to(tl.int64)
        n7 = (head_start + 7).to(tl.int64)

        s_acc8 = tl.zeros([BM, KL], dtype=tl.float32)

        qk0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_0[:, None] * qk0
        qk0 = qk0.to(tl.float16)

        qk1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_1[:, None] * qk1
        qk1 = qk1.to(tl.float16)

        qk2 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_2 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n2 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_2[:, None] * qk2
        qk2 = qk2.to(tl.float16)

        qk3 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_3 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n3 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_3[:, None] * qk3
        qk3 = qk3.to(tl.float16)

        qk4 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n4,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_4 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n4 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_4[:, None] * qk4
        qk4 = qk4.to(tl.float16)

        qk5 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n5,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_5 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n5 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_5[:, None] * qk5
        qk5 = qk5.to(tl.float16)

        qk6 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n6,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_6 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n6 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_6[:, None] * qk6
        qk6 = qk6.to(tl.float16)

        qk7 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n7,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_7 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n7 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1_7[:, None] * qk7
        qk7 = qk7.to(tl.float16)

        a_acc8 = tl.zeros([BM, KL], dtype=tl.float32)
        a_acc8 = _consume_qk(qk0, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk1, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk2, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk3, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk4, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n4,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk5, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n5,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk6, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n6,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)
        a_acc8 = _consume_qk(qk7, s_acc8, a_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n7,
                             PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                             stride_wb, stride_wt, stride_wn,
                             stride_vb, stride_vt, stride_vn, stride_vd,
                             stride_ob, stride_ot, stride_on, stride_od)

        a_acc8_h = a_acc8.to(tl.float16)
        if D == 128:
            d2_offs = tl.arange(0, 256)
            head_delta = d2_offs // 128
            d_pair = d2_offs - head_delta * 128
            for pair_idx in range(4):
                n0p = (head_start + pair_idx * 2).to(tl.int64)
                n_pair = n0p + head_delta.to(tl.int64)
                v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                          + n_pair[None, :] * stride_vn + d_pair[None, :].to(tl.int64) * stride_vd)
                v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
                av = tl.dot(a_acc8_h, v_blk)

                pw2_0 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0p * stride_wn,
                                mask=q_mask, other=0.0).to(tl.float32)
                pw2_1 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + (n0p + 1) * stride_wn,
                                mask=q_mask, other=0.0).to(tl.float32)
                pw2p = tl.where(head_delta[None, :] == 0, pw2_0[:, None], pw2_1[:, None])

                o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                          + n_pair[None, :] * stride_on + d_pair[None, :].to(tl.int64) * stride_od)
                o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
                tl.store(o_ptrs, o_prev + pw2p * av, mask=q_mask[:, None])
        else:
            for h in range(8):
                ni = (head_start + h).to(tl.int64)
                v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                          + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
                v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
                av = tl.dot(a_acc8_h, v_blk)
                pw2p = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                                mask=q_mask, other=0.0).to(tl.float32)
                o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                          + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
                o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
                tl.store(o_ptrs, (o_prev + pw2p[:, None] * av), mask=q_mask[:, None])
        return

    num_pairs = HPG // 2
    s_acc = tl.zeros([BM, KL], dtype=tl.float32)

    for pair_idx in range(num_pairs - 2):
        ni0 = (head_start + pair_idx * 2).to(tl.int64)
        ni1 = (head_start + pair_idx * 2 + 1).to(tl.int64)
        qk = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni0,
                      stride_qb, stride_qt, stride_qn, stride_qd,
                      stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni0 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc += pw1_0[:, None] * qk

        qk = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni1,
                      stride_qb, stride_qt, stride_qn, stride_qd,
                      stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1_1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni1 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc += pw1_1[:, None] * qk

    # Cache the final two pairs. They stay live through sweep 2.
    n_c0 = (head_start + (num_pairs - 2) * 2).to(tl.int64)
    n_c1 = (head_start + (num_pairs - 2) * 2 + 1).to(tl.int64)
    n_l0 = (head_start + (num_pairs - 1) * 2).to(tl.int64)
    n_l1 = (head_start + (num_pairs - 1) * 2 + 1).to(tl.int64)

    qk_c0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_c0,
                     stride_qb, stride_qt, stride_qn, stride_qd,
                     stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    qk_c1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_c1,
                     stride_qb, stride_qt, stride_qn, stride_qd,
                     stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    qk_l0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_l0,
                     stride_qb, stride_qt, stride_qn, stride_qd,
                     stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    qk_l1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_l1,
                     stride_qb, stride_qt, stride_qn, stride_qd,
                     stride_kb, stride_kt, stride_kn, stride_kd, scaling)

    pw1_c0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_c0 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_c0[:, None] * qk_c0
    qk_c0 = qk_c0.to(tl.float16)
    pw1_c1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_c1 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_c1[:, None] * qk_c1
    qk_c1 = qk_c1.to(tl.float16)
    pw1_l0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_l0 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_l0[:, None] * qk_l0
    qk_l0 = qk_l0.to(tl.float16)
    pw1_l1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_l1 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_l1[:, None] * qk_l1
    qk_l1 = qk_l1.to(tl.float16)

    a_acc = tl.zeros([BM, KL], dtype=tl.float32)
    a_acc = _consume_qk(qk_l0, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_l0,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
    a_acc = _consume_qk(qk_l1, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_l1,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
    a_acc = _consume_qk(qk_c0, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_c0,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
    a_acc = _consume_qk(qk_c1, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n_c1,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)

    for pair_idx in range(num_pairs - 2):
        ni0 = (head_start + pair_idx * 2).to(tl.int64)
        ni1 = (head_start + pair_idx * 2 + 1).to(tl.int64)
        qk = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni0,
                      stride_qb, stride_qt, stride_qn, stride_qd,
                      stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        a_acc = _consume_qk(qk, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni0,
                            PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                            stride_wb, stride_wt, stride_wn,
                            stride_vb, stride_vt, stride_vn, stride_vd,
                            stride_ob, stride_ot, stride_on, stride_od)
        qk = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni1,
                      stride_qb, stride_qt, stride_qn, stride_qd,
                      stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        a_acc = _consume_qk(qk, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni1,
                            PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                            stride_wb, stride_wt, stride_wn,
                            stride_vb, stride_vt, stride_vn, stride_vd,
                            stride_ob, stride_ot, stride_on, stride_od)

    a_acc_h = a_acc.to(tl.float16)
    if D == 128 and HPG % 2 == 0:
        d2_offs = tl.arange(0, 256)
        head_delta = d2_offs // 128
        d_pair = d2_offs - head_delta * 128
        for pair_idx in range(HPG // 2):
            n0p = (head_start + pair_idx * 2).to(tl.int64)
            n_pair = n0p + head_delta.to(tl.int64)
            v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                      + n_pair[None, :] * stride_vn + d_pair[None, :].to(tl.int64) * stride_vd)
            v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            av = tl.dot(a_acc_h, v_blk)

            pw2_0 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0p * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)
            pw2_1 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + (n0p + 1) * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)
            pw2p = tl.where(head_delta[None, :] == 0, pw2_0[:, None], pw2_1[:, None])

            o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                      + n_pair[None, :] * stride_on + d_pair[None, :].to(tl.int64) * stride_od)
            o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
            tl.store(o_ptrs, o_prev + pw2p * av, mask=q_mask[:, None])
    else:
        for h in range(HPG):
            ni = (head_start + h).to(tl.int64)
            v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                      + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
            v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            av = tl.dot(a_acc_h, v_blk)
            pw2p = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)
            o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                      + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
            o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
            tl.store(o_ptrs, (o_prev + pw2p[:, None] * av), mask=q_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["B", "T", "N", "D", "W", "BM", "KL", "G", "HPG"],
)
@triton.jit
def _dc_onekernel_w128_hpg4(
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
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr, FULL_LEN: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_c * BM
    head_start = g * 4

    q_offs = q_start + tl.arange(0, BM)
    d_offs = tl.arange(0, 128)
    kl_offs = tl.arange(0, 128)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    if FULL_LEN:
        q_mask = tl.full([BM], True, tl.int1)
        k_mask = kl_offs < (BM + W - 1)
    else:
        seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
        q_mask = (q_offs < T) & (q_offs < seq_len)
        k_mask = (k_offs < T) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))
    rel = q_offs[:, None] - k_offs[None, :]
    valid = (rel >= 0) & (rel < W) & q_mask[:, None] & k_mask[None, :]

    n0 = head_start.to(tl.int64)
    n1 = (head_start + 1).to(tl.int64)
    n2 = (head_start + 2).to(tl.int64)
    n3 = (head_start + 3).to(tl.int64)

    s_acc = tl.zeros([BM, 128], dtype=tl.float32)

    qk0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                    mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_0[:, None] * qk0
    qk0 = qk0.to(tl.float16)

    qk1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
                    mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_1[:, None] * qk1
    qk1 = qk1.to(tl.float16)

    qk2 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_2 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n2 * stride_wn,
                    mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_2[:, None] * qk2
    qk2 = qk2.to(tl.float16)

    qk3 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd, scaling)
    pw1_3 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n3 * stride_wn,
                    mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_3[:, None] * qk3
    qk3 = qk3.to(tl.float16)

    a_acc = tl.zeros([BM, 128], dtype=tl.float32)
    a_acc = _consume_qk(qk0, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
    a_acc = _consume_qk(qk1, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
    a_acc = _consume_qk(qk2, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
    a_acc = _consume_qk(qk3, s_acc, a_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                        PRE_W2, PRE_DD, POST_W1, POST_DD, V, OUT,
                        stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)

    for h in range(4):
        ni = (head_start + h).to(tl.int64)
        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        av = tl.dot(a_acc.to(v_blk.dtype), v_blk)
        pw2p = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        tl.store(o_ptrs, o_prev + pw2p[:, None] * av, mask=q_mask[:, None])


class TritonDCOneKernel:
    """V4H: H100-oriented V4 experiment; unsupported cases fall back to V4."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=16, chunk_size=16,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G
        assert N % G == 0
        assert HPG >= 2 and HPG % 2 == 0, f"HPG={HPG} must be even and >= 2"

        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)
        if G > 8 or HPG < 4 or HPG > 8 or KL > 256:
            from triton_dc_onekernel_v4 import TritonDCOneKernel as V4
            return V4.forward(q, k, v, dc_weights, scaling, window, seq_lens, G=G, chunk_size=chunk_size)

        full_len_fast = seq_lens is None and T % chunk_size == 0 and T >= chunk_size + W

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous(); pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous(); post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous(); post_dd = post_dd.contiguous()
        seq_lens_arg = seq_lens
        if seq_lens_arg is None:
            if full_len_fast and D == 128 and HPG == 4 and KL == 128 and chunk_size + W == 128:
                seq_lens_arg = pre_w1
            else:
                seq_lens_arg = torch.full((B,), T, device=q.device, dtype=torch.int32)

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)
        num_chunks = triton.cdiv(T, chunk_size)
        grid = (num_chunks, B * G)

        if D == 128 and HPG == 4 and KL == 128 and chunk_size + W == 128:
            _dc_onekernel_w128_hpg4[grid](
                q, k, v,
                pre_w1, pre_w2, pre_dd,
                post_w1, post_w2, post_dd,
                out, seq_lens_arg,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                scaling,
                B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
                FULL_LEN=full_len_fast,
            )
            return out

        _dc_onekernel_cache4[grid](
            q, k, v,
            pre_w1, pre_w2, pre_dd,
            post_w1, post_w2, post_dd,
            out, seq_lens_arg,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
        )
        return out
