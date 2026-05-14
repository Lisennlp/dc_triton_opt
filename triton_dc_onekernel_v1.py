"""Single-kernel DC residual attention — no key tiling, no HBM intermediates.

Grid = (num_q_chunks, B * G)

Supports any HPG via a general loop:
  Sweep 1: loop all HPG heads → QK → s_acc (register). Last 2 heads' QK cached.
  Sweep 2: process last 2 heads using cached QK (no re-read).
            Then re-read remaining heads in pairs of 2 → QK → score → softmax → PV.
  Sweep 3: a_acc @ V for all heads.

For HPG=2: zero Q/K re-reads (fully fused).
For HPG=4: only 2 heads re-read (50% saving vs full re-read).
For HPG=8: 6 heads re-read (25% saving).
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _load_qk_pair(Q, K, b, q_offs, k_offs, q_mask, k_mask, d_offs,
                   n0, n1,
                   stride_qb, stride_qt, stride_qn, stride_qd,
                   stride_kb, stride_kt, stride_kn, stride_kd,
                   scaling):
    """Load Q,K for 2 heads and compute QK. Returns qk0, qk1."""
    q0_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
               + n0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
    q0 = tl.load(q0_ptrs, mask=q_mask[:, None], other=0.0)
    k0_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
               + n0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
    k0 = tl.load(k0_ptrs, mask=k_mask[:, None], other=0.0)
    qk0 = tl.dot(q0, tl.trans(k0)) * scaling

    q1_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
               + n1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
    q1 = tl.load(q1_ptrs, mask=q_mask[:, None], other=0.0)
    k1_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
               + n1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
    k1 = tl.load(k1_ptrs, mask=k_mask[:, None], other=0.0)
    qk1 = tl.dot(q1, tl.trans(k1)) * scaling
    return qk0, qk1


@triton.jit
def _process_head(qk, s_acc, valid, b, q_offs, k_offs, q_mask, k_mask, d_offs, ni,
                  PRE_W2, PRE_DD, POST_W1, POST_DD,
                  V, OUT, a_acc,
                  stride_wb, stride_wt, stride_wn,
                  stride_vb, stride_vt, stride_vn, stride_vd,
                  stride_ob, stride_ot, stride_on, stride_od,
                  BM: tl.constexpr, KL: tl.constexpr, D: tl.constexpr):
    """Score → softmax → PV → a_acc accumulation → store direct PV."""
    pw2 = tl.load(PRE_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                   mask=q_offs < tl.load(OUT + 0).to(tl.int64) * 0 + 99999999, other=0.0).to(tl.float32)
    # Simpler: just always load with q_mask
    return a_acc  # placeholder


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
    q_mask = q_offs < T
    d_offs = tl.arange(0, D)
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len)
    causal = k_offs[None, :] <= q_offs[:, None]
    win = (q_offs[:, None] - k_offs[None, :]) < W
    valid = causal & win & q_mask[:, None] & k_mask[None, :]

    # ═══ Sweep 1: compute all HPG heads' QK, accumulate s_acc ═══
    # Process in pairs of 2. Cache the LAST pair's QK.
    s_acc = tl.zeros([BM, KL], dtype=tl.float32)
    num_pairs = HPG // 2

    # All pairs except the last: load QK, accumulate s_acc, discard QK
    for pair_idx in range(num_pairs - 1):
        n0 = (head_start + pair_idx * 2).to(tl.int64)
        n1 = (head_start + pair_idx * 2 + 1).to(tl.int64)

        q0_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                   + n0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q0 = tl.load(q0_ptrs, mask=q_mask[:, None], other=0.0)
        k0_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                   + n0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k0 = tl.load(k0_ptrs, mask=k_mask[:, None], other=0.0)
        qk0 = tl.dot(q0, tl.trans(k0)) * scaling

        q1_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                   + n1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q1 = tl.load(q1_ptrs, mask=q_mask[:, None], other=0.0)
        k1_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                   + n1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k1 = tl.load(k1_ptrs, mask=k_mask[:, None], other=0.0)
        qk1 = tl.dot(q1, tl.trans(k1)) * scaling

        pw1_0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        pw1_1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n1 * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        s_acc += pw1_0[:, None] * qk0 + pw1_1[:, None] * qk1

    # Last pair: load QK, accumulate s_acc, KEEP QK in registers
    n_last0 = (head_start + (num_pairs - 1) * 2).to(tl.int64)
    n_last1 = (head_start + (num_pairs - 1) * 2 + 1).to(tl.int64)

    q_last0_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                    + n_last0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
    q_last0 = tl.load(q_last0_ptrs, mask=q_mask[:, None], other=0.0)
    k_last0_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                    + n_last0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
    k_last0 = tl.load(k_last0_ptrs, mask=k_mask[:, None], other=0.0)
    qk_last0 = tl.dot(q_last0, tl.trans(k_last0)) * scaling

    q_last1_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                    + n_last1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
    q_last1 = tl.load(q_last1_ptrs, mask=q_mask[:, None], other=0.0)
    k_last1_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                    + n_last1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
    k_last1 = tl.load(k_last1_ptrs, mask=k_mask[:, None], other=0.0)
    qk_last1 = tl.dot(q_last1, tl.trans(k_last1)) * scaling

    pw1_l0 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_last0 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    pw1_l1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n_last1 * stride_wn,
                     mask=q_mask, other=0.0).to(tl.float32)
    s_acc += pw1_l0[:, None] * qk_last0 + pw1_l1[:, None] * qk_last1
    # s_acc is now COMPLETE. qk_last0, qk_last1 still in registers.

    # ═══ Sweep 2: score → softmax → PV → a_acc ═══
    a_acc = tl.zeros([BM, KL], dtype=tl.float32)

    # Helper: process one head given its QK
    # (Inlined as a macro-like pattern since Triton JIT functions are tricky with mutable state)

    # Process last pair FIRST (QK already in registers, zero re-read)
    for local_h in range(2):
        if local_h == 0:
            ni = n_last0
            qk = qk_last0
        else:
            ni = n_last1
            qk = qk_last1

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
        p = tl.exp(score - rm[:, None])
        p = tl.where(valid, p, 0.0)
        rs = tl.sum(p, axis=1)
        sl = tl.where(rs > 0.0, rs, 1.0)
        probs = p / sl[:, None]

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        pv = tl.dot(probs.to(tl.float16), v_blk)
        a_acc += pw1p[:, None] * probs

        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, ((pddp[:, None] + 1.0) * pv).to(tl.float16), mask=q_mask[:, None])

    # qk_last0, qk_last1 can be freed now

    # Process remaining pairs (need to re-read Q, K)
    for pair_idx in range(num_pairs - 1):
        n0 = (head_start + pair_idx * 2).to(tl.int64)
        n1 = (head_start + pair_idx * 2 + 1).to(tl.int64)

        q0_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                   + n0 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q0 = tl.load(q0_ptrs, mask=q_mask[:, None], other=0.0)
        k0_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                   + n0 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k0 = tl.load(k0_ptrs, mask=k_mask[:, None], other=0.0)
        qk0 = tl.dot(q0, tl.trans(k0)) * scaling

        q1_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                   + n1 * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q1 = tl.load(q1_ptrs, mask=q_mask[:, None], other=0.0)
        k1_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                   + n1 * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k1 = tl.load(k1_ptrs, mask=k_mask[:, None], other=0.0)
        qk1 = tl.dot(q1, tl.trans(k1)) * scaling

        for local_h in range(2):
            if local_h == 0:
                ni = n0; qk = qk0
            else:
                ni = n1; qk = qk1

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
            p = tl.exp(score - rm[:, None])
            p = tl.where(valid, p, 0.0)
            rs = tl.sum(p, axis=1)
            sl = tl.where(rs > 0.0, rs, 1.0)
            probs = p / sl[:, None]

            v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                      + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
            v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            pv = tl.dot(probs.to(tl.float16), v_blk)
            a_acc += pw1p[:, None] * probs

            o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                      + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
            tl.store(o_ptrs, ((pddp[:, None] + 1.0) * pv).to(tl.float16), mask=q_mask[:, None])

    # ═══ Sweep 3: a_acc @ V + final combine ═══
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        pw2p = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                        mask=q_mask, other=0.0).to(tl.float32)
        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        av = tl.dot(a_acc.to(tl.float16), v_blk)
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        tl.store(o_ptrs, (o_prev + pw2p[:, None] * av).to(tl.float16), mask=q_mask[:, None])


class TritonDCOneKernel:
    """Single kernel DC. Supports any even HPG."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=16, chunk_size=16,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G
        assert HPG >= 2 and HPG % 2 == 0, f"HPG={HPG} must be even and >= 2"

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous(); pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous(); post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous(); post_dd = post_dd.contiguous()

        import triton
        KL = chunk_size + W
        assert KL & (KL - 1) == 0, f"BM+W={KL} must be power of 2."

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
            num_warps=4, num_stages=1,
        )
        return out
