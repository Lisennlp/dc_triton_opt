"""Single-kernel pre-only DC attention.

This variant removes the whole post-DC path:
- no post_w1 / post_w2 / post_dd
- keep pre_w1 / pre_w2 / pre_dd logits mixing
- output is PV from the pre-mixed softmax

Fast path:
- HPG=4 or HPG=8 with KL<=128 caches all QK matrices as fp16 after they
  contribute to the fp32 pre aggregation.

Generic path:
- all other HPG/KL shapes use two passes: build s_acc, then recompute QK for
  pre-mixed softmax + PV.
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
def _consume_pre_qk(
    qk, s_acc, valid,
    b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
    PRE_W2, PRE_DD, V, OUT,
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

    v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
              + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    pv = tl.dot(probs.to(v_blk.dtype), v_blk)

    o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
              + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
    tl.store(o_ptrs, pv, mask=q_mask[:, None])


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
def _pre_only_onekernel(
    Q, K, V,
    PRE_W1, PRE_W2, PRE_DD,
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

    if HPG == 4 and KL <= 128:
        n0 = head_start.to(tl.int64)
        n1 = (head_start + 1).to(tl.int64)
        n2 = (head_start + 2).to(tl.int64)
        n3 = (head_start + 3).to(tl.int64)
        s_acc4 = tl.zeros([BM, KL], dtype=tl.float32)

        qk0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc4 += pw1[:, None] * qk0
        qk0 = qk0.to(tl.float16)

        qk1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc4 += pw1[:, None] * qk1
        qk1 = qk1.to(tl.float16)

        qk2 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n2 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc4 += pw1[:, None] * qk2
        qk2 = qk2.to(tl.float16)

        qk3 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n3 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc4 += pw1[:, None] * qk3
        qk3 = qk3.to(tl.float16)

        _consume_pre_qk(qk0, s_acc4, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk1, s_acc4, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk2, s_acc4, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk3, s_acc4, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        return

    if HPG == 8 and KL <= 128:
        s_acc8 = tl.zeros([BM, KL], dtype=tl.float32)
        n0 = head_start.to(tl.int64)
        n1 = (head_start + 1).to(tl.int64)
        n2 = (head_start + 2).to(tl.int64)
        n3 = (head_start + 3).to(tl.int64)
        n4 = (head_start + 4).to(tl.int64)
        n5 = (head_start + 5).to(tl.int64)
        n6 = (head_start + 6).to(tl.int64)
        n7 = (head_start + 7).to(tl.int64)

        qk0 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk0
        qk0 = qk0.to(tl.float16)

        qk1 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk1
        qk1 = qk1.to(tl.float16)

        qk2 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n2 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk2
        qk2 = qk2.to(tl.float16)

        qk3 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n3 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk3
        qk3 = qk3.to(tl.float16)

        qk4 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n4,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n4 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk4
        qk4 = qk4.to(tl.float16)

        qk5 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n5,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n5 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk5
        qk5 = qk5.to(tl.float16)

        qk6 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n6,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n6 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk6
        qk6 = qk6.to(tl.float16)

        qk7 = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, n7,
                       stride_qb, stride_qt, stride_qn, stride_qd,
                       stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n7 * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc8 += pw1[:, None] * qk7
        qk7 = qk7.to(tl.float16)

        _consume_pre_qk(qk0, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n0,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk1, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n1,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk2, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n2,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk3, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n3,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk4, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n4,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk5, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n5,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk6, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n6,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        _consume_pre_qk(qk7, s_acc8, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, n7,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)
        return

    s_acc = tl.zeros([BM, KL], dtype=tl.float32)
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        qk = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
                      stride_qb, stride_qt, stride_qn, stride_qd,
                      stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc += pw1[:, None] * qk

    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        qk = _load_qk(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
                      stride_qb, stride_qt, stride_qn, stride_qd,
                      stride_kb, stride_kt, stride_kn, stride_kd, scaling)
        _consume_pre_qk(qk, s_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
                        PRE_W2, PRE_DD, V, OUT, stride_wb, stride_wt, stride_wn,
                        stride_vb, stride_vt, stride_vn, stride_vd,
                        stride_ob, stride_ot, stride_on, stride_od)


class TritonDCOneKernel:
    """Pre-only DC: keep pre logits mixing, remove post probability mixing."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=16, chunk_size=16,
    ):
        if len(dc_weights) == 6:
            pre_w1, pre_w2, pre_dd, _, _, _ = dc_weights
        elif len(dc_weights) == 3:
            pre_w1, pre_w2, pre_dd = dc_weights
        else:
            raise ValueError("dc_weights must be either 6 full weights or 3 pre-only weights")

        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G
        assert N % G == 0
        assert HPG >= 1, f"HPG={HPG} must be >= 1"

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous()
        pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous()

        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)
        num_chunks = triton.cdiv(T, chunk_size)
        grid = (num_chunks, B * G)

        _pre_only_onekernel[grid](
            q, k, v,
            pre_w1, pre_w2, pre_dd,
            out, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
        )
        return out
