"""Single-kernel DC residual attention — no key tiling, no HBM intermediates.

Grid = (num_q_chunks, B * G)
Each block: HPG = N/G heads, key dim = KL (one shot).

s_acc and a_acc live entirely in registers. No s_buf or a_buf HBM traffic.

Sweep 1: loop HPG heads → load Q,K → QK → accumulate s_acc in registers
Sweep 2: loop HPG heads → load Q,K,V → QK → score(register s_acc) → softmax → PV
          + accumulate a_acc in registers → store direct PV to OUT
Sweep 3: loop HPG heads → load V → a_acc@V → load/update OUT

Total HBM: Q read 2×, K read 2×, V read 2×, OUT write 2× (vs FA2: Q 1×, K 1×, V 1×, OUT 1×)
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl


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

    # ═══════════════ Sweep 1: pre-agg → s_acc (REGISTER only) ═══════════════
    s_acc = tl.zeros([BM, KL], dtype=tl.float32)
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        q_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                  + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q_blk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling
        pw1 = tl.load(PRE_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        s_acc += pw1[:, None] * qk

    # s_acc [BM, KL] now complete in registers — no HBM write!

    # ═══════════════ Sweep 2: score → softmax → PV + post-agg → a_acc (REGISTER) ═══════════════
    a_acc = tl.zeros([BM, KL], dtype=tl.float32)
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        pw2 = tl.load(PRE_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        pdd = tl.load(PRE_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        ds = pdd + 1.0
        pw1_post = tl.load(POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)
        pdd_post = tl.load(POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)

        # Reload Q, K for QK (2nd read)
        q_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                  + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q_blk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

        # Score using register-resident s_acc
        score = ds[:, None] * qk + pw2[:, None] * s_acc
        score = tl.where(valid, score, float("-inf"))

        # Direct softmax
        row_max = tl.max(score, axis=1)
        p = tl.exp(score - row_max[:, None])
        p = tl.where(valid, p, 0.0)
        row_sum = tl.sum(p, axis=1)
        safe_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
        probs = p / safe_sum[:, None]

        # PV
        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        pv = tl.dot(probs.to(tl.float16), v_blk)

        # Accumulate a_acc in registers
        a_acc += pw1_post[:, None] * probs

        # Store (1+post_dd) * pv → OUT
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, ((pdd_post[:, None] + 1.0) * pv).to(tl.float16), mask=q_mask[:, None])

    # a_acc [BM, KL] now complete in registers — no HBM write!

    # ═══════════════ Sweep 3: a_acc @ V + final combine ═══════════════
    # a_acc is in registers; only need to load V and update OUT
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)
        pw2_post = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                            mask=q_mask, other=0.0).to(tl.float32)

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        av = tl.dot(a_acc.to(tl.float16), v_blk)

        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        tl.store(o_ptrs, (o_prev + pw2_post[:, None] * av).to(tl.float16), mask=q_mask[:, None])


class TritonDCOneKernel:
    """Single kernel DC, no HBM intermediates. Grid = (C, B*G)."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=1, chunk_size=16,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G

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
