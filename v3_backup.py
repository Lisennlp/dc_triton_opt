"""Single-kernel DC v1 — fused sweep1+2, last pair QK cached, register s_acc/a_acc.

Optimizations vs v0:
- exp2 instead of exp (native GPU instruction)
- autotune over num_warps and num_stages
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['B', 'T', 'N', 'D', 'W', 'BM', 'KL', 'G', 'HPG'],
)
@triton.jit
def _dc_onekernel(
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
    # KSPAN = BM + W - 1; KL >= KSPAN (padded to power-of-2)
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))
    causal = k_offs[None, :] <= q_offs[:, None]
    win = (q_offs[:, None] - k_offs[None, :]) < W
    valid = causal & win & q_mask[:, None] & k_mask[None, :]

    num_pairs = HPG // 2

    # ═══ Sweep 1: all heads QK → s_acc. Cache last pair. ═══
    s_acc = tl.zeros([BM, KL], dtype=tl.float32)

    for pair_idx in range(num_pairs - 1):
        n0 = (head_start + pair_idx * 2).to(tl.int64)
        n1 = (head_start + pair_idx * 2 + 1).to(tl.int64)

        q0 = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                      + n0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                      mask=q_mask[:, None], other=0.0)
        k0 = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + n0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                      mask=k_mask[:, None], other=0.0)
        qk0 = tl.dot(q0, tl.trans(k0)) * scaling

        q1 = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                      + n1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                      mask=q_mask[:, None], other=0.0)
        k1 = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + n1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                      mask=k_mask[:, None], other=0.0)
        qk1 = tl.dot(q1, tl.trans(k1)) * scaling

        pw1_0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        pw1_1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc += pw1_0[:, None] * qk0 + pw1_1[:, None] * qk1

    # Last pair: keep QK
    n_last0 = (head_start + (num_pairs - 1) * 2).to(tl.int64)
    n_last1 = (head_start + (num_pairs - 1) * 2 + 1).to(tl.int64)

    q_l0 = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                    + n_last0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                    mask=q_mask[:, None], other=0.0)
    k_l0 = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                    + n_last0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                    mask=k_mask[:, None], other=0.0)
    qk_l0 = tl.dot(q_l0, tl.trans(k_l0)) * scaling

    q_l1 = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                    + n_last1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                    mask=q_mask[:, None], other=0.0)
    k_l1 = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                    + n_last1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                    mask=k_mask[:, None], other=0.0)
    qk_l1 = tl.dot(q_l1, tl.trans(k_l1)) * scaling

    pw1_l0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_last0 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    pw1_l1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_last1 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_l0[:, None] * qk_l0 + pw1_l1[:, None] * qk_l1

    # ═══ Sweep 2: score → exp2-softmax → PV → a_acc ═══
    a_acc = tl.zeros([BM, KL], dtype=tl.float32)

    # Last pair (QK cached)
    for lh in range(2):
        ni = n_last0 if lh == 0 else n_last1
        qk = qk_l0 if lh == 0 else qk_l1

        pw2 = tl.load(PRE_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        pdd = tl.load(PRE_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        pw1p = tl.load(POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        pddp = tl.load(POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)

        score = (pdd + 1.0)[:, None] * qk + pw2[:, None] * s_acc
        score = tl.where(valid, score, float("-inf"))

        # exp2 softmax: exp2(x * log2e) == exp(x)
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
        a_acc += pw1p[:, None] * probs

        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, ((pddp[:, None] + 1.0) * pv), mask=q_mask[:, None])

    # Remaining pairs (re-read Q/K)
    for pair_idx in range(num_pairs - 1):
        n0 = (head_start + pair_idx * 2).to(tl.int64)
        n1 = (head_start + pair_idx * 2 + 1).to(tl.int64)

        q0 = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                      + n0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                      mask=q_mask[:, None], other=0.0)
        k0 = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + n0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                      mask=k_mask[:, None], other=0.0)
        qk0 = tl.dot(q0, tl.trans(k0)) * scaling

        q1 = tl.load(Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                      + n1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd,
                      mask=q_mask[:, None], other=0.0)
        k1 = tl.load(K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + n1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd,
                      mask=k_mask[:, None], other=0.0)
        qk1 = tl.dot(q1, tl.trans(k1)) * scaling

        for lh in range(2):
            ni = n0 if lh == 0 else n1
            qk = qk0 if lh == 0 else qk1

            pw2 = tl.load(PRE_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                           mask=q_mask, other=0.0).to(tl.float32)
            pdd = tl.load(PRE_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                           mask=q_mask, other=0.0).to(tl.float32)
            pw1p = tl.load(POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)
            pddp = tl.load(POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)

            score = (pdd + 1.0)[:, None] * qk + pw2[:, None] * s_acc
            score = tl.where(valid, score, float("-inf"))

            rm = tl.max(score, axis=1)
            p = tl.exp2((score - rm[:, None]) * LOG2E) # <=> exp(score - max)
            p = tl.where(valid, p, 0.0)
            rs = tl.sum(p, axis=1)
            sl = tl.where(rs > 0.0, rs, 1.0)
            probs = p / sl[:, None]

            v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                      + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
            v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            pv = tl.dot(probs.to(v_blk.dtype), v_blk)
            a_acc += pw1p[:, None] * probs

            o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                      + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
            tl.store(o_ptrs, ((pddp[:, None] + 1.0) * pv), mask=q_mask[:, None])

    # ═══ Sweep 3: a_acc @ V + final ═══
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        pw2p = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        av = tl.dot(a_acc.to(v_blk.dtype), v_blk)
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        tl.store(o_ptrs, (o_prev + pw2p[:, None] * av), mask=q_mask[:, None])


class TritonDCOneKernel:
    """V1: fused sweep1+2, exp2 softmax, autotune."""

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

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous(); pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous(); post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous(); post_dd = post_dd.contiguous()

        import triton
        # KL = chunk_size + W
        # assert KL & (KL - 1) == 0, f"BM+W={KL} must be power of 2."
        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)
        num_chunks = triton.cdiv(T, chunk_size)
        grid = (num_chunks, B * G)

        _dc_onekernel[grid](
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
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL,
            G=G, HPG=HPG,
        )
        return out
